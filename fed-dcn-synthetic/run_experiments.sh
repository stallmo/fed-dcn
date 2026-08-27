#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Experiment configuration — edit these values
# =============================================================================

# Dataset to use for training and evaluation: "mnist", "fashion-mnist", or "usps"
# For usps, input-dim is automatically set to 256 by experiment_runner.py.
DATASET="usps"

# Number of Optuna trials to run
N_TRIALS=50

PRETRAIN_EPOCHS=50
WARMSTART_EPOCHS=25

# Optimization goal for Optuna: "accuracy" (maximize) or "db_latent" (minimize)
OBJECTIVE="accuracy"

# Metric used for early stopping inside each run: "accuracy" or "db_latent"
EARLY_STOPPING_METRIC=$OBJECTIVE

# Flower SuperLink to submit runs to (local simulation or a running SuperLink address)
SUPERLINK="local-simulation"

# Number of simulated clients
NUM_SUPERNODES=20

# Weights & Biases project name
WANDB_PROJECT="fed-dcn-synthetic-hpo-iid-$DATASET-$OBJECTIVE-20-clients"

# Optuna study name (used to resume a previous study when combined with STUDY_STORAGE)
STUDY_NAME=$WANDB_PROJECT
#"fed-dcn-synthetic-hpo-iid-$DATASET-$OBJECTIVE-balanced-clusters"

# Optuna storage URL — leave empty to use in-memory storage (results are not persisted)
# Example for SQLite persistence: "sqlite:///optuna.db"
STUDY_STORAGE="sqlite:///optuna_iid_20_clients_$DATASET.db"

# Set to 1 to skip Phase 1+2 if a pretraining cache exists for the current architecture
REUSE_PRETRAINING=0

# Kill and prune a trial if it runs longer than this many hours
TRIAL_TIMEOUT_HOURS=1

# W&B entity (leave empty to use the currently logged-in user)
ENTITY=""

# Optional: import previous W&B runs into the Optuna study before starting HPO.
# Set to the source W&B project name to import from, or leave empty to skip.
# Requires STUDY_STORAGE to be set (in-memory storage cannot be seeded).
SEED_FROM_WANDB_PROJECT=$WANDB_PROJECT

# =============================================================================
# Launch — no edits needed below this line
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ARGS=(
    --dataset           "$DATASET"
    --n-trials          "$N_TRIALS"
    --objective         "$OBJECTIVE"
    --early-stopping-metric "$EARLY_STOPPING_METRIC"
    --phase2a-epochs       "$PRETRAIN_EPOCHS"
    --phase2b-epochs       "$WARMSTART_EPOCHS"
    --superlink         "$SUPERLINK"
    --num-supernodes    "$NUM_SUPERNODES"
    --wandb-project     "$WANDB_PROJECT"
    --study-name        "$STUDY_NAME"
)

if [[ -n "$STUDY_STORAGE" ]]; then
    ARGS+=(--study-storage "$STUDY_STORAGE")
fi

if [[ "$REUSE_PRETRAINING" -eq 1 ]]; then
    ARGS+=(--reuse-pretraining)
fi

ARGS+=(--trial-timeout "$(( TRIAL_TIMEOUT_HOURS * 3600 ))")

echo "Starting experiment runner with:"
printf "  %-26s %s\n" "n-trials:"              "$N_TRIALS"
printf "  %-26s %s\n" "objective:"             "$OBJECTIVE"
printf "  %-26s %s\n" "early-stopping-metric:" "$EARLY_STOPPING_METRIC"
printf "  %-26s %s\n" "superlink:"             "$SUPERLINK"
printf "  %-26s %s\n" "num-supernodes:"        "$NUM_SUPERNODES"
printf "  %-26s %s\n" "wandb-project:"         "$WANDB_PROJECT"
printf "  %-26s %s\n" "entity:"                "${ENTITY:-<logged-in user>}"
printf "  %-26s %s\n" "study-name:"            "$STUDY_NAME"
printf "  %-26s %s\n" "study-storage:"         "${STUDY_STORAGE:-<in-memory>}"
printf "  %-26s %s\n" "reuse-pretraining:"     "$REUSE_PRETRAINING"
printf "  %-26s %s\n" "trial-timeout:"         "${TRIAL_TIMEOUT_HOURS}h"
printf "  %-26s %s\n" "seed-from-wandb:"       "${SEED_FROM_WANDB_PROJECT:-<none>}"
echo ""

cd "$SCRIPT_DIR"

# Pre-download USPS data in the main process so Ray workers find it on disk.
# (torchvision's download needs an SSL workaround on Python 3.14; the patch lives
# inside _usps_tensors and only applies during the one-time download.)
if [[ "$DATASET" == "usps" ]]; then
    echo "Pre-fetching USPS dataset..."
    uv run python -c "
from fed_dcn_synthetic_app.task import _usps_tensors
_usps_tensors(True)
_usps_tensors(False)
print('USPS data ready.')
"
    echo ""
fi

if [[ -n "$SEED_FROM_WANDB_PROJECT" ]]; then
    if [[ -z "$STUDY_STORAGE" ]]; then
        echo "Error: SEED_FROM_WANDB_PROJECT requires STUDY_STORAGE to be set (in-memory storage cannot be seeded)."
        exit 1
    fi
    SEED_ARGS=(
        --wandb-project "$SEED_FROM_WANDB_PROJECT"
        --study-name    "$STUDY_NAME"
        --study-storage "$STUDY_STORAGE"
        --objective     "$OBJECTIVE"
    )
    [[ -n "$ENTITY" ]] && SEED_ARGS+=(--entity "$ENTITY")
    echo "Seeding Optuna study from W&B project '$SEED_FROM_WANDB_PROJECT'..."
    uv run python seed_study_from_wandb.py "${SEED_ARGS[@]}"
    echo ""
fi

uv run python experiment_runner.py "${ARGS[@]}"
