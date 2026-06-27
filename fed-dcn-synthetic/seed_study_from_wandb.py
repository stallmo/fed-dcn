"""Seed an Optuna study from previous runs logged in Weights & Biases.

Queries a W&B project for completed HPO trials, reverse-maps the Flower config
keys back to Optuna parameter names, and inserts them into a persistent Optuna
study via study.add_trial(). This lets you resume HPO from prior results without
re-running any experiments.

Usage:
    uv run python seed_study_from_wandb.py \\
        --wandb-project fed-dcn-synthetic-hpo-non-iid-mnist-accuracy \\
        --study-name    fed-dcn-hpo-non-iid-mnist \\
        --study-storage sqlite:///optuna_non_iid_mnist.db \\
        --objective     accuracy
"""

import argparse
import sys

import optuna
import wandb
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)

from experiment_runner import HIDDEN_DIMS_CHOICES
from repeat_runner import _normalize

# Distributions must match the search space in experiment_runner._suggest_params exactly.
_DISTRIBUTIONS = {
    "hidden_dims":    CategoricalDistribution(HIDDEN_DIMS_CHOICES),
    "bottleneck_dim": IntDistribution(3, 16),
    "alpha_geom":           FloatDistribution(1e-4, 1.0, log=True),
    "alpha_geom_federated": FloatDistribution(1e-4, 1.0, log=True),
    "clust_weight":         FloatDistribution(1e-4, 1.0, log=True),
    "recon_weight":   FloatDistribution(0.1, 10.0, log=True),
    "learning_rate":  FloatDistribution(1e-5, 1e-2, log=True),
}

# Maps the W&B config key (Flower naming) to the Optuna parameter name.
# Where multiple Flower keys carry the same value (all alpha-geom-* phases,
# all learning-rate-* phases), the canonical key is listed here.
_WANDB_KEY_TO_OPTUNA = {
    "hidden-dims":              "hidden_dims",
    "bottleneck-dim":           "bottleneck_dim",
    "alpha-geom":               "alpha_geom",
    "alpha-geom-federated":     "alpha_geom_federated",
    "lambda-clust-loss":        "clust_weight",
    "alpha-reconstruction-loss": "recon_weight",
    "learning-rate":            "learning_rate",
}

_METRIC_KEY = {
    "accuracy":  "train_acc",
    "db_latent": "train_db_latent",
}


def _extract_params(run_config: dict) -> dict | None:
    """Extract and coerce the 6 Optuna params from a W&B run config.

    Returns None (and prints a reason) if any required key is missing.
    """
    params = {}
    for wandb_key, optuna_name in _WANDB_KEY_TO_OPTUNA.items():
        raw = run_config.get(wandb_key)
        if raw is None:
            print(f"    skip: missing config key '{wandb_key}'")
            return None
        params[optuna_name] = _normalize(raw)
    return params


def _in_bounds(params: dict) -> bool:
    """Return True if all params fall within their declared distributions."""
    for name, dist in _DISTRIBUTIONS.items():
        v = params[name]
        if isinstance(dist, CategoricalDistribution):
            if v not in dist.choices:
                print(f"    skip: '{name}={v}' not in categorical choices {dist.choices}")
                return False
        elif isinstance(dist, IntDistribution):
            if not (dist.low <= int(v) <= dist.high):
                print(f"    skip: '{name}={v}' out of int range [{dist.low}, {dist.high}]")
                return False
        elif isinstance(dist, FloatDistribution):
            if not (dist.low <= float(v) <= dist.high):
                print(f"    skip: '{name}={v}' out of float range [{dist.low}, {dist.high}]")
                return False
    return True


def seed_study(
    wandb_project: str,
    study_name: str,
    study_storage: str,
    objective: str,
    entity: str | None,
) -> None:
    api = wandb.Api()
    path = f"{entity}/{wandb_project}" if entity else wandb_project
    metric_key = _METRIC_KEY[objective]
    direction = "maximize" if objective == "accuracy" else "minimize"

    print(f"Querying W&B project '{path}' (excluding repeat_run-tagged runs)...")
    try:
        runs = list(api.runs(path, filters={"tags": {"$nin": ["repeat_run"]}}))
    except (wandb.errors.CommError, ValueError):
        print(f"W&B project '{path}' not found — no trials loaded.")
        return
    print(f"Found {len(runs)} candidate run(s).")

    if not runs:
        print("No W&B runs to seed — nothing to do.")
        return

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        optuna.delete_study(study_name=study_name, storage=study_storage)
        print(f"Dropped existing study '{study_name}' from storage (repopulating from W&B).")
    except KeyError:
        pass

    study = optuna.create_study(
        study_name=study_name,
        storage=study_storage,
        direction=direction,
        load_if_exists=False,
    )

    seeded = 0
    skipped = 0
    for run in runs:
        print(f"  run '{run.name}' (id={run.id})")

        objective_value = run.summary.get(metric_key)
        if objective_value is None:
            print(f"    skip: no summary metric '{metric_key}' (run may have failed)")
            skipped += 1
            continue

        params = _extract_params(run.config)
        if params is None:
            skipped += 1
            continue

        if not _in_bounds(params):
            skipped += 1
            continue

        # Coerce types to match distributions.
        params["bottleneck_dim"] = int(params["bottleneck_dim"])
        params["alpha_geom"]           = float(params["alpha_geom"])
        params["alpha_geom_federated"] = float(params["alpha_geom_federated"])
        params["clust_weight"]   = float(params["clust_weight"])
        params["recon_weight"]   = float(params["recon_weight"])
        params["learning_rate"]  = float(params["learning_rate"])

        trial = optuna.trial.create_trial(
            params=params,
            distributions=_DISTRIBUTIONS,
            value=float(objective_value),
        )
        study.add_trial(trial)
        seeded += 1
        print(f"    seeded  {metric_key}={float(objective_value):.4f}  params={params}")

    print(f"\nDone. Seeded {seeded} trial(s), skipped {skipped}.")
    print(f"Study now has {len(study.trials)} trial(s) total.")
    if study.trials:
        best = study.best_trial
        print(f"Best trial: #{best.number}  {metric_key}={best.value:.4f}  params={best.params}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed an Optuna study from previous W&B HPO runs"
    )
    parser.add_argument("--wandb-project", required=True, help="W&B project to query")
    parser.add_argument("--study-name",    required=True, help="Optuna study name")
    parser.add_argument(
        "--study-storage", required=True,
        help="Optuna storage URL (e.g. sqlite:///optuna.db). In-memory storage is not supported.",
    )
    parser.add_argument("--objective", choices=["accuracy", "db_latent"], required=True)
    parser.add_argument("--entity", default=None, help="W&B entity (defaults to logged-in user)")
    args = parser.parse_args()

    if not args.study_storage:
        print("Error: --study-storage must be a persistent URL (e.g. sqlite:///optuna.db).")
        sys.exit(1)

    seed_study(
        wandb_project=args.wandb_project,
        study_name=args.study_name,
        study_storage=args.study_storage,
        objective=args.objective,
        entity=args.entity,
    )


if __name__ == "__main__":
    main()
