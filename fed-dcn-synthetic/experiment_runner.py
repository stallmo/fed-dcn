"""Optuna hyperparameter optimization runner for fed-dcn-synthetic.

Each trial:
  1. Samples hyperparameters from the search space.
  2. Writes a temporary TOML override file.
  3. Launches `flwr run` as a subprocess with the overrides.
  4. Parses the resulting run directory from stdout.
  5. Reads final metrics and logs everything to W&B.
  6. Returns the objective value (accuracy or db_latent) to Optuna.

Usage:
    uv run python experiment_runner.py \\
        --n-trials 20 \\
        --objective accuracy \\
        --early-stopping-metric accuracy \\
        --superlink local-simulation \\
        --num-supernodes 5 \\
        --wandb-project fed-dcn-synthetic-hpo \\
        [--study-name my-study] \\
        [--study-storage sqlite:///optuna.db] \\
        [--reuse-pretraining]
"""

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path

import optuna
import tomllib
import wandb

# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

def _cleanup_stale_ray_sessions() -> None:
    """Delete Ray session directories whose parent process is no longer alive.

    Each failed ray.shutdown() leaves its session dir behind, which accumulates
    GBs of disk usage and causes the next run to fail the same way. This removes
    only sessions whose originating process is confirmed dead.
    """
    ray_tmp = Path(os.environ.get("RAY_TMPDIR", "/tmp")) / "ray"
    if not ray_tmp.exists():
        return
    pattern = re.compile(r"^session_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_\d+_(\d+)$")
    cleaned = 0
    for entry in sorted(ray_tmp.iterdir()):
        if not entry.is_dir():
            continue
        m = pattern.match(entry.name)
        if not m:
            continue
        pid = int(m.group(1))
        try:
            os.kill(pid, 0)  # signal 0: check existence without killing
        except ProcessLookupError:
            shutil.rmtree(entry, ignore_errors=True)
            cleaned += 1
        except PermissionError:
            pass  # process alive, owned by another user
    if cleaned:
        print(f"  [cleanup] Removed {cleaned} stale Ray session(s) from {ray_tmp}")


HIDDEN_DIMS_CHOICES = [
    "[128]",
    "[256, 128]",
    "[512, 256]",
    "[512, 256, 128]",
    "[1024, 512, 256, 128, 64]",
]


def _suggest_params(trial: optuna.Trial) -> dict:
    hidden_dims = trial.suggest_categorical("hidden_dims", HIDDEN_DIMS_CHOICES)
    bottleneck_dim = trial.suggest_int("bottleneck_dim", 3, 16)
    alpha_geom = trial.suggest_float("alpha_geom", 1e-4, 1.0, log=True)
    alpha_geom_federated = trial.suggest_float("alpha_geom_federated", 1e-4, 1.0, log=True)
    clust_weight = trial.suggest_float("clust_weight", 1e-4, 1.0, log=True)
    recon_weight = trial.suggest_float("recon_weight", 0.1, 10.0, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    return {
        "hidden_dims": hidden_dims,
        "bottleneck_dim": bottleneck_dim,
        "alpha_geom": alpha_geom,
        "alpha_geom_federated": alpha_geom_federated,
        "clust_weight": clust_weight,
        "recon_weight": recon_weight,
        "learning_rate": learning_rate,
    }


_DATASET_INPUT_DIM = {"mnist": 784, "fashion-mnist": 784, "usps": 256}


def _params_to_config_overrides(params: dict, early_stopping_metric: str, phase2a_epochs: int, phase2b_epochs: int,
                                dataset: str) -> dict:
    """Map Optuna params to pyproject.toml config keys.

    alpha_geom is applied to Phases 1, 2a, and 2b; alpha_geom_federated controls Phase 3.
    A single learning_rate is applied to all phase-specific LR keys.
    A single clust_weight is applied to both Phase 2b and Phase 3 cluster loss.
    """
    lr = params["learning_rate"]
    ag = params["alpha_geom"]
    cw = params["clust_weight"]
    return {
        "hidden-dims": params["hidden_dims"],
        "bottleneck-dim": params["bottleneck_dim"],
        # umap-n-components must equal bottleneck-dim; geometry loss computes
        # MSELoss(encoder_output[B, bottleneck], umap_targets[B, n_components])
        "umap-n-components": params["bottleneck_dim"],
        "alpha-geom": ag,
        "alpha-geom-warmstart": ag,
        "alpha-geom-federated": params["alpha_geom_federated"],
        "alpha-geom-local": ag,
        "lambda-clust-loss": cw,
        "beta-clust-warmstart": cw,
        "alpha-reconstruction-loss": params["recon_weight"],
        "learning-rate": lr,
        "learning-rate-local": lr,
        "learning-rate-pretrain": lr,
        "learning-rate-federated": lr,
        "early-stopping-metric": early_stopping_metric,
        "phase2a-epochs": phase2a_epochs,
        "phase2b-epochs": phase2b_epochs,
        "dataset": dataset,
        "input-dim": _DATASET_INPUT_DIM[dataset],
    }


# ---------------------------------------------------------------------------
# TOML override file
# ---------------------------------------------------------------------------

def _write_override_toml(path: Path, overrides: dict) -> None:
    lines = []
    for k, v in overrides.items():
        if isinstance(v, str):
            lines.append(f'{k} = "{v}"')
        elif isinstance(v, bool):
            lines.append(f"{k} = {str(v).lower()}")
        else:
            lines.append(f"{k} = {v}")
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Run directory detection
# ---------------------------------------------------------------------------

def _find_run_dirs(app_dir: Path, dataset: str) -> set[Path]:
    base = app_dir / "fitted_models" / dataset
    if not base.exists():
        return set()
    return {p for p in base.iterdir() if p.is_dir()}


def _parse_run_dir_from_output(output: str, app_dir: Path) -> Path | None:
    for line in output.splitlines():
        if "Run output:" in line:
            # Extract the path segment after "Run output: "
            idx = line.index("Run output:") + len("Run output:")
            candidate = line[idx:].strip()
            # Could be relative — resolve against app_dir
            p = Path(candidate)
            if not p.is_absolute():
                p = app_dir / p
            if p.exists():
                return p
    return None


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _read_best_metrics(run_dir: Path, best_round: int) -> dict:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return {}
    with open(metrics_path) as f:
        history = json.load(f)
    # Find entry for best_round
    for entry in history:
        if entry.get("round") == best_round:
            return entry
    # Fallback: last entry
    return history[-1] if history else {}


def _read_run_config(run_dir: Path) -> dict:
    cfg_path = run_dir / "run_config.json"
    if not cfg_path.exists():
        return {}
    with open(cfg_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Trial objective
# ---------------------------------------------------------------------------

def make_objective(
    app_dir: Path,
    base_cfg: dict,
    superlink: str,
    num_supernodes: int,
    early_stopping_metric: str,
    wandb_project: str,
    reuse_pretraining: bool,
    tmp_dir: Path,
    phase2a_epochs: int,
    phase2b_epochs: int,
    dataset: str,
    trial_timeout: int = 3600,
):
    flwr_bin = app_dir / ".venv" / "bin" / "flwr"

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial)
        overrides = _params_to_config_overrides(params=params, early_stopping_metric=early_stopping_metric,
                                                phase2a_epochs=phase2a_epochs,
                                                phase2b_epochs=phase2b_epochs, dataset=dataset)
        if reuse_pretraining:
            overrides["reuse-pretraining"] = True

        override_path = tmp_dir / f"trial_{trial.number}.toml"
        _write_override_toml(override_path, overrides)

        #dataset = str(base_cfg.get("dataset", "fashion-mnist"))
        existing_dirs = _find_run_dirs(app_dir, dataset)

        log_path = tmp_dir / f"trial_{trial.number}.log"
        cmd = [
            str(flwr_bin), "run", ".",
            superlink,
            "--run-config", str(override_path),
            "--federation-config", f"num-supernodes={num_supernodes}",
            "--stream",
        ]

        _cleanup_stale_ray_sessions()

        ## every 10th trial, restart supernode
        if trial.number % 10 == 0:
            print(f"\n[Trial {trial.number}] Restarting supernode")
            result = subprocess.run(
                ["pkill", "-f", "flower-superlink|flower-supernode|flower-superexec|flwr"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )

        print(f"\n[Trial {trial.number}] Running: {' '.join(cmd)}")
        print(f"[Trial {trial.number}] Log: {log_path}  (timeout: {trial_timeout}s)")

        timed_out = False
        with open(log_path, "w") as log_file:
            proc = subprocess.Popen(
                cmd,
                cwd=str(app_dir),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # own process group so we can kill all children
            )
            try:
                proc.wait(timeout=trial_timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                print(
                    f"[Trial {trial.number}] Timeout after {trial_timeout}s — "
                    "killing process group and moving to next trial."
                )
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    proc.wait(timeout=15)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                _cleanup_stale_ray_sessions()

        result_returncode = proc.returncode
        stdout = log_path.read_text()

        if timed_out:
            raise optuna.exceptions.TrialPruned(f"Trial exceeded timeout of {trial_timeout}s")

        if result_returncode != 0:
            print(
                f"[Trial {trial.number}] flwr run exited with code {result_returncode}. "
                "Checking whether outputs are valid before pruning..."
            )

        # Locate run directory
        run_dir = _parse_run_dir_from_output(stdout, app_dir)
        if run_dir is None:
            new_dirs = _find_run_dirs(app_dir, dataset) - existing_dirs
            if new_dirs:
                run_dir = max(new_dirs, key=lambda p: p.stat().st_mtime)
            else:
                print(f"[Trial {trial.number}] Could not locate run directory.")
                if result.returncode != 0:
                    print(f"[Trial {trial.number}] Last 3000 chars of log:")
                    print(stdout[-3000:])
                raise optuna.exceptions.TrialPruned("Run directory not found")

        # Verify outputs exist before trusting a non-zero exit code.
        # Flower 1.30+ may report finished:failed even when the ServerApp completed
        # successfully (e.g. Ray shutdown fails due to disk pressure), so we treat
        # the run as successful if the required output files are present.
        run_cfg_path = run_dir / "run_config.json"
        if run_cfg_path.exists():
            print(
                f"[Trial {trial.number}] flwr run exited {result_returncode} but "
                f"run_config.json exists — treating as completed. "
                f"(Likely Flower/Ray shutdown issue, not a training failure.)"
            )
        else:
            print(f"[Trial {trial.number}] flwr run failed (exit {result_returncode}), last 3000 chars of log:")
            print(stdout[-3000:])
            raise optuna.exceptions.TrialPruned(f"flwr run exited with code {result_returncode}")

        print(f"[Trial {trial.number}] Run dir: {run_dir}")

        run_cfg = _read_run_config(run_dir)
        best_round = int(run_cfg.get("best_round", 0))
        best_metrics = _read_best_metrics(run_dir, best_round)

        # Train metrics — used for early stopping and HPO objective
        acc        = float(best_metrics.get("train_acc",       run_cfg.get("best_acc", 0.0)))
        db_latent  = float(best_metrics.get("train_db_latent", run_cfg.get("best_db_latent", float("inf"))))
        nmi        = float(best_metrics.get("train_nmi", 0.0))
        ari        = float(best_metrics.get("train_ari", 0.0))
        db_observed = float(best_metrics.get("train_db_observed", float("inf")))
        # Test metrics — logged to W&B for analysis only
        test_acc        = float(best_metrics.get("test_acc", 0.0))
        test_db_latent  = float(best_metrics.get("test_db_latent", float("inf")))
        test_nmi        = float(best_metrics.get("test_nmi", 0.0))
        test_ari        = float(best_metrics.get("test_ari", 0.0))
        test_db_observed = float(best_metrics.get("test_db_observed", float("inf")))
        stopped_early = run_cfg.get("stopped_early", "False").lower() == "true"

        # W&B logging
        wandb_config = {
            **{k: str(v) for k, v in base_cfg.items()},
            **overrides,
            "run_dir_name": run_dir.name,
            "num_supernodes": num_supernodes,
            "optuna_trial": trial.number,
        }

        wandb.init(
            project=wandb_project,
            name=f"trial-{trial.number}",
            config=wandb_config,
            reinit=True,
        )
        wandb.log({
            "train_acc": acc, "train_nmi": nmi, "train_ari": ari,
            "train_db_latent": db_latent, "train_db_observed": db_observed,
            "test_acc": test_acc, "test_nmi": test_nmi, "test_ari": test_ari,
            "test_db_latent": test_db_latent, "test_db_observed": test_db_observed,
            "best_round": best_round,
            "stopped_early": int(stopped_early),
        })
        wandb.finish()

        shutil.rmtree(run_dir, ignore_errors=True)
        print(f"[Trial {trial.number}] Cleaned up run dir: {run_dir}")

        objective_value = acc if early_stopping_metric == "accuracy" else db_latent
        print(
            f"[Trial {trial.number}] train_acc={acc:.4f}  train_nmi={nmi:.4f}  "
            f"train_db_latent={db_latent:.4f}  test_acc={test_acc:.4f}  "
            f"best_round={best_round}  objective={objective_value:.4f}"
        )
        return objective_value

    return objective


# ---------------------------------------------------------------------------
# Base config loading
# ---------------------------------------------------------------------------

def _load_base_cfg(app_dir: Path) -> dict:
    toml_path = app_dir / "pyproject.toml"
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)
    return data.get("tool", {}).get("flwr", {}).get("app", {}).get("config", {})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna HPO runner for fed-dcn-synthetic")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument(
        "--objective", choices=["accuracy", "db_latent"], default="db_latent",
        help="Optuna optimization direction target",
    )
    parser.add_argument(
        "--early-stopping-metric", choices=["accuracy", "db_latent"], default="db_latent",
        help="Metric used for early stopping inside each run",
    )
    parser.add_argument(
        "--superlink", default="local-simulation",
        help="SuperLink connection name passed to `flwr run . <superlink>`",
    )
    parser.add_argument("--num-supernodes", type=int, default=5)
    parser.add_argument("--wandb-project", default="fed-dcn-synthetic-hpo")
    parser.add_argument("--study-name", default="fed-dcn-hpo")
    parser.add_argument("--study-storage", default=None, help="Optuna storage URL (e.g. sqlite:///optuna.db)")
    parser.add_argument(
        "--reuse-pretraining", action="store_true",
        help="Pass reuse-pretraining=true to all trials",
    )
    parser.add_argument('--dataset', default='fashion-mnist', help='Dataset to use (default: fashion-mnist')
    parser.add_argument('--phase2a-epochs', type=int, default=10, help='Number of epochs for Phase 2a (default: 10)')
    parser.add_argument('--phase2b-epochs', type=int, default=10, help='Number of epochs for Phase 2b (default: 10)')
    parser.add_argument(
        '--trial-timeout', type=int, default=3600,
        help='Kill a trial and prune it if it runs longer than this many seconds (default: 3600)',
    )
    args = parser.parse_args()

    app_dir = Path(__file__).parent.resolve()
    base_cfg = _load_base_cfg(app_dir)
    print(f"Base config loaded from {app_dir / 'pyproject.toml'}")

    direction = "maximize" if args.objective == "accuracy" else "minimize"

    with tempfile.TemporaryDirectory(prefix="fed_dcn_hpo_") as tmp_str:
        tmp_dir = Path(tmp_str)

        objective_fn = make_objective(
            app_dir=app_dir,
            base_cfg=base_cfg,
            superlink=args.superlink,
            num_supernodes=args.num_supernodes,
            early_stopping_metric=args.early_stopping_metric,
            wandb_project=args.wandb_project,
            reuse_pretraining=args.reuse_pretraining,
            tmp_dir=tmp_dir,
            phase2a_epochs=args.phase2a_epochs,
            phase2b_epochs=args.phase2b_epochs,
            dataset=args.dataset,
            trial_timeout=args.trial_timeout,
        )

        optuna.logging.set_verbosity(optuna.logging.INFO)
        study = optuna.create_study(
            study_name=args.study_name,
            storage=args.study_storage,
            direction=direction,
            load_if_exists=True,
        )
        study.optimize(objective_fn, n_trials=args.n_trials)

    print("\n=== Optuna study complete ===")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best value ({args.objective}): {study.best_trial.value:.4f}")
    print("Best params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    results = {
        "best_trial": study.best_trial.number,
        "best_value": study.best_trial.value,
        "objective": args.objective,
        "best_params": study.best_trial.params,
        "n_trials": len(study.trials),
    }
    out_path = app_dir / "optuna_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
