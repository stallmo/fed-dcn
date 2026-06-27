"""Re-run the best W&B trial N times with different random seeds.

Queries a W&B project for the best run (by train_acc or train_db_latent),
extracts its full config, and launches N federated runs with identical
hyperparameters but varying seeds. Each run is logged to W&B with the
tag "repeat_run".

Usage:
    uv run python repeat_runner.py \\
        --project fed-dcn-synthetic-hpo-iid-mnist-accuracy \\
        --objective accuracy \\
        --n-runs 10 \\
        --num-supernodes 5
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import wandb

from experiment_runner import (
    _cleanup_stale_ray_sessions,
    _find_run_dirs,
    _parse_run_dir_from_output,
    _read_best_metrics,
    _read_run_config,
    _write_override_toml,
)

# Keys present in the W&B config that are runner metadata, not Flower config,
# or that should never be inherited by repeat runs (e.g. cache flags that were
# only enabled during HPO to speed up trials).
_META_KEYS = {"run_dir_name", "num_supernodes", "optuna_trial", "reuse-pretraining"}


def _normalize(v):
    """Coerce a W&B config value (may be a string) back to its natural Python type."""
    if not isinstance(v, str):
        return v
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def _fetch_best_run(project: str, entity: str | None, objective: str):
    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    order = "-summary_metrics.train_acc" if objective == "accuracy" else "+summary_metrics.train_db_latent"
    runs = api.runs(
        path=path,
        filters={"tags": {"$nin": ["repeat_run"]}},
        order=order,
    )
    try:
        return next(iter(runs))
    except StopIteration:
        print(f"No runs found in project '{path}' (excluding repeat_run-tagged runs).")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-run best W&B trial N times with different seeds")
    parser.add_argument("--project", required=True, help="W&B project to query for best run")
    parser.add_argument("--objective", choices=["accuracy", "db_latent"], required=True)
    parser.add_argument("--n-runs", type=int, required=True)
    parser.add_argument("--num-supernodes", type=int, required=True)
    parser.add_argument("--superlink", default="local-simulation")
    parser.add_argument("--entity", default=None, help="W&B entity (defaults to logged-in user)")
    parser.add_argument(
        "--wandb-project", default=None,
        help="W&B project for logging repeat runs (defaults to --project)",
    )
    parser.add_argument(
        "--base-seed", type=int, default=1000,
        help="First seed value; run i uses base-seed+i (default: 1000)",
    )
    parser.add_argument(
        "--override", metavar="KEY=VALUE", action="append", default=[],
        help="Override a config key from the best run (e.g. --override dirichlet-alpha=0.1). "
             "Can be repeated. Applied after the W&B config is fetched, before each run.",
    )
    args = parser.parse_args()

    wandb_project = args.wandb_project or args.project

    # --- Step 1: Find best run ---
    print(f"Querying W&B project '{args.project}' for best run by {args.objective}...")
    best_run = _fetch_best_run(args.project, args.entity, args.objective)

    metric_key = "train_acc" if args.objective == "accuracy" else "train_db_latent"
    best_value = best_run.summary.get(metric_key, "N/A")
    print(f"Best run: {best_run.name} (id={best_run.id}), {metric_key}={best_value}")

    # --- Step 2: Extract full config ---
    params = {
        k: _normalize(v)
        for k, v in best_run.config.items()
        if k not in _META_KEYS
    }
    print(f"Extracted {len(params)} config keys from best run.")

    # Parse --override KEY=VALUE flags and merge them over the W&B config.
    cli_overrides: dict = {}
    for item in args.override:
        if "=" not in item:
            print(f"Warning: ignoring malformed --override '{item}' (expected KEY=VALUE)")
            continue
        k, _, v = item.partition("=")
        cli_overrides[k.strip()] = _normalize(v.strip())
    if cli_overrides:
        print(f"Applying {len(cli_overrides)} CLI override(s): {cli_overrides}")
        params.update(cli_overrides)

    params.setdefault("augment-federated", True)

    dataset = str(params.get("dataset", "mnist"))
    app_dir = Path(__file__).parent.resolve()
    flwr_bin = app_dir / ".venv" / "bin" / "flwr"

    # --- Step 3: Repeat runs ---
    with tempfile.TemporaryDirectory(prefix="fed_dcn_repeat_") as tmp_str:
        tmp_dir = Path(tmp_str)

        for i in range(args.n_runs):
            seed = args.base_seed + i
            print(f"\n[Repeat {i}] seed={seed}")

            run_params = {**params, "random-seed": seed}
            override_path = tmp_dir / f"repeat_{i}.toml"
            _write_override_toml(override_path, run_params)

            existing_dirs = _find_run_dirs(app_dir, dataset)
            log_path = tmp_dir / f"repeat_{i}.log"

            cmd = [
                str(flwr_bin), "run", ".",
                args.superlink,
                "--run-config", str(override_path),
                "--federation-config", f"num-supernodes={args.num_supernodes}",
                "--stream",
            ]

            _cleanup_stale_ray_sessions()

            print(f"[Repeat {i}] Running: {' '.join(cmd)}")
            print(f"[Repeat {i}] Log: {log_path}")

            with open(log_path, "w") as log_file:
                result = subprocess.run(
                    cmd,
                    cwd=str(app_dir),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )

            stdout = log_path.read_text()

            if result.returncode != 0:
                print(
                    f"[Repeat {i}] flwr run exited with code {result.returncode}. "
                    "Checking whether outputs are valid before skipping..."
                )

            run_dir = _parse_run_dir_from_output(stdout, app_dir)
            if run_dir is None:
                new_dirs = _find_run_dirs(app_dir, dataset) - existing_dirs
                if new_dirs:
                    run_dir = max(new_dirs, key=lambda p: p.stat().st_mtime)
                else:
                    print(f"[Repeat {i}] Could not locate run directory. Skipping.")
                    print(f"[Repeat {i}] Last 3000 chars of log:")
                    print(stdout[-3000:])
                    continue

            run_cfg_path = run_dir / "run_config.json"
            if not run_cfg_path.exists():
                print(f"[Repeat {i}] run_config.json not found — training likely failed. Skipping.")
                print(f"[Repeat {i}] Last 3000 chars of log:")
                print(stdout[-3000:])
                continue

            if result.returncode != 0:
                print(
                    f"[Repeat {i}] flwr run exited {result.returncode} but "
                    "run_config.json exists — treating as completed. "
                    "(Likely Flower/Ray shutdown issue, not a training failure.)"
                )

            print(f"[Repeat {i}] Run dir: {run_dir}")

            run_cfg = _read_run_config(run_dir)
            best_round = int(run_cfg.get("best_round", 0))
            best_metrics = _read_best_metrics(run_dir, best_round)

            acc              = float(best_metrics.get("train_acc",          run_cfg.get("best_acc", 0.0)))
            db_latent        = float(best_metrics.get("train_db_latent",    run_cfg.get("best_db_latent", float("inf"))))
            nmi              = float(best_metrics.get("train_nmi",          0.0))
            ari              = float(best_metrics.get("train_ari",          0.0))
            db_observed      = float(best_metrics.get("train_db_observed",  float("inf")))
            test_acc         = float(best_metrics.get("test_acc",           0.0))
            test_db_latent   = float(best_metrics.get("test_db_latent",     float("inf")))
            test_nmi         = float(best_metrics.get("test_nmi",           0.0))
            test_ari         = float(best_metrics.get("test_ari",           0.0))
            test_db_observed = float(best_metrics.get("test_db_observed",   float("inf")))
            stopped_early    = run_cfg.get("stopped_early", "False").lower() == "true"

            wandb_config = {
                **{k: str(v) for k, v in run_params.items()},
                "source_run_id":   best_run.id,
                "source_run_name": best_run.name,
                "num_supernodes":  args.num_supernodes,
                "repeat_index":    i,
            }

            wandb.init(
                project=wandb_project,
                name=f"repeat-{i}",
                tags=["repeat_run"],
                config=wandb_config,
                reinit=True,
            )
            wandb.log({
                "train_acc": acc,         "train_nmi": nmi,          "train_ari": ari,
                "train_db_latent": db_latent, "train_db_observed": db_observed,
                "test_acc": test_acc,     "test_nmi": test_nmi,      "test_ari": test_ari,
                "test_db_latent": test_db_latent, "test_db_observed": test_db_observed,
                "best_round": best_round,
                "stopped_early": int(stopped_early),
            })
            wandb.finish()

            shutil.rmtree(run_dir, ignore_errors=True)
            print(f"[Repeat {i}] Cleaned up run dir: {run_dir}")

            print(
                f"[Repeat {i}] train_acc={acc:.4f}  train_nmi={nmi:.4f}  "
                f"train_db_latent={db_latent:.4f}  test_acc={test_acc:.4f}  "
                f"best_round={best_round}"
            )

    print("\n=== Repeat runs complete ===")


if __name__ == "__main__":
    main()
