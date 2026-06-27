#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Configuration — edit these values
# =============================================================================

# W&B project to query for the best run
PROJECT="fed-dcn-synthetic-hpo-non-iid-mnist-accuracy"

# Optimization objective used to rank runs:
#   "accuracy"  → picks run with highest train_acc
#   "db_latent" → picks run with lowest train_db_latent
OBJECTIVE="accuracy"

# Number of repeat runs (each uses a different random seed)
N_RUNS=5

# Number of simulated clients
NUM_SUPERNODES=5

# Flower SuperLink to submit runs to
SUPERLINK="local-simulation"

# First seed value; run i uses BASE_SEED+i
BASE_SEED=1005

# W&B project to log repeat runs into (leave empty to use PROJECT)
WANDB_PROJECT=""

# W&B entity (leave empty to use the currently logged-in user)
ENTITY=""

# =============================================================================
# Ablation overrides — edit these to pin specific values over the best-run config.
# Leave a variable empty ("") to inherit the value from the best W&B run.
# =============================================================================

# Non-IID impact study: lower α = more heterogeneous partitioning
DIRICHLET_ALPHA=""

# Geometry (UMAP) loss weights — set all four together for a clean ablation
# (0 disables the geometry loss in that phase; leave empty to use best-run values)
ALPHA_GEOM_LOCAL=""             # Phase 1 local DCN training
ALPHA_GEOM=""                   # Phase 2a UMAP geometry pretraining
ALPHA_GEOM_WARMSTART=""         # Phase 2b warm-start
ALPHA_GEOM_FEDERATED=""         # Phase 3 federated rounds

# Synthetic data
SYNTHETIC_SAMPLES_TOTAL=""        # total synthetic samples, proportionally allocated (e.g. 5000, 10000)
# SYNTHETIC_SAMPLES_PER_CLUSTER="" # legacy: fixed count per non-empty cluster
AUGMENT_FEDERATED=""              # set to "false" to disable synthetic augmentation during Phase 3 federation

# Federated training
NUM_SERVER_ROUNDS=""            # total federated rounds (e.g. 50, 100)

# =============================================================================
# Launch — no edits needed below this line
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ARGS=(
    --project        "$PROJECT"
    --objective      "$OBJECTIVE"
    --n-runs         "$N_RUNS"
    --num-supernodes "$NUM_SUPERNODES"
    --superlink      "$SUPERLINK"
    --base-seed      "$BASE_SEED"
)

if [[ -n "$WANDB_PROJECT" ]]; then
    ARGS+=(--wandb-project "$WANDB_PROJECT")
fi

if [[ -n "$ENTITY" ]]; then
    ARGS+=(--entity "$ENTITY")
fi

# Append ablation overrides for any non-empty variable.
[[ -n "$DIRICHLET_ALPHA"               ]] && ARGS+=(--override "dirichlet-alpha=$DIRICHLET_ALPHA")
[[ -n "$ALPHA_GEOM_LOCAL"              ]] && ARGS+=(--override "alpha-geom-local=$ALPHA_GEOM_LOCAL")
[[ -n "$ALPHA_GEOM"                    ]] && ARGS+=(--override "alpha-geom=$ALPHA_GEOM")
[[ -n "$ALPHA_GEOM_WARMSTART"          ]] && ARGS+=(--override "alpha-geom-warmstart=$ALPHA_GEOM_WARMSTART")
[[ -n "$ALPHA_GEOM_FEDERATED"          ]] && ARGS+=(--override "alpha-geom-federated=$ALPHA_GEOM_FEDERATED")
[[ -n "$SYNTHETIC_SAMPLES_TOTAL"        ]] && ARGS+=(--override "synthetic-samples-total=$SYNTHETIC_SAMPLES_TOTAL")
[[ -n "$AUGMENT_FEDERATED"              ]] && ARGS+=(--override "augment-federated=$AUGMENT_FEDERATED")
[[ -n "$NUM_SERVER_ROUNDS"             ]] && ARGS+=(--override "num-server-rounds=$NUM_SERVER_ROUNDS")

echo "Starting repeat runner with:"
printf "  %-30s %s\n" "project:"                        "$PROJECT"
printf "  %-30s %s\n" "objective:"                      "$OBJECTIVE"
printf "  %-30s %s\n" "n-runs:"                         "$N_RUNS"
printf "  %-30s %s\n" "num-supernodes:"                 "$NUM_SUPERNODES"
printf "  %-30s %s\n" "superlink:"                      "$SUPERLINK"
printf "  %-30s %s\n" "base-seed:"                      "$BASE_SEED"
printf "  %-30s %s\n" "wandb-project:"                  "${WANDB_PROJECT:-<same as project>}"
printf "  %-30s %s\n" "entity:"                         "${ENTITY:-<logged-in user>}"
printf "  %-30s %s\n" "dirichlet-alpha:"                "${DIRICHLET_ALPHA:-<from best run>}"
printf "  %-30s %s\n" "alpha-geom-local:"               "${ALPHA_GEOM_LOCAL:-<from best run>}"
printf "  %-30s %s\n" "alpha-geom:"                     "${ALPHA_GEOM:-<from best run>}"
printf "  %-30s %s\n" "alpha-geom-warmstart:"           "${ALPHA_GEOM_WARMSTART:-<from best run>}"
printf "  %-30s %s\n" "alpha-geom-federated:"           "${ALPHA_GEOM_FEDERATED:-<from best run>}"
printf "  %-30s %s\n" "synthetic-samples-total:"         "${SYNTHETIC_SAMPLES_TOTAL:-<from best run>}"
printf "  %-30s %s\n" "augment-federated:"               "${AUGMENT_FEDERATED:-<from best run>}"
printf "  %-30s %s\n" "num-server-rounds:"              "${NUM_SERVER_ROUNDS:-<from best run>}"
echo ""

cd "$SCRIPT_DIR"
uv run python repeat_runner.py "${ARGS[@]}"
