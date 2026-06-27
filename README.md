# Fed-DCN: Federated Deep Clustering with Synthetic Data

Federated clustering using Deep Clustering Networks (DCN) augmented with latent-space synthetic data generation and UMAP geometry pretraining. The active implementation lives in `fed-dcn-synthetic/`.

## Overview

Training proceeds in three phases coordinated by a Flower ServerApp:

**Phase 1 — Local DCN training & statistics sharing**
Each client trains a local autoencoder + k-means clustering model on its private data. Clients share cluster centres, per-cluster standard deviations, within-radius point counts, and decoder weights with the server. The server generates a synthetic dataset by sampling from each client's learned latent-space cluster distributions.

**Phase 2 — Server-side geometry pretraining**
The server pretrains a global model on the synthetic dataset using a UMAP geometry loss (Phase 2a) and a DCN warm-start with cluster loss (Phase 2b). This gives the federation a well-initialised starting point.

**Phase 3 — Federated rounds**
Standard federated learning with FedAvg/FedProx. Clients receive the global model and optionally the synthetic dataset for local data augmentation. The geometry loss can continue during federation using pre-computed UMAP targets.

## Repository structure

```
fed-dcn-synthetic/
├── fed_dcn_synthetic_app/
│   ├── server_app.py        # ServerApp: orchestrates all three phases
│   ├── client_app.py        # ClientApp: local training and stat sharing
│   ├── dcn.py               # DCN trainer (AE + cluster loss + geometry loss)
│   ├── autoencoder.py       # Encoder/decoder architectures
│   ├── clustering.py        # K-means cluster model
│   └── task.py              # Shared factory functions
├── experiment_runner.py     # Optuna HPO runner
├── repeat_runner.py         # Repeat best W&B run with different seeds
├── seed_study_from_wandb.py # Seed an Optuna study from prior W&B results
├── run_experiments.sh       # Configure and launch HPO
├── repeat_best_run.sh       # Configure and launch repeat/ablation runs
├── smoke_test.py            # Quick sanity check for core modules
└── pyproject.toml           # Flower app config and default hyperparameters
```

## How to run

### Prerequisites

[uv](https://github.com/astral-sh/uv) is used for dependency management.

```bash
cd fed-dcn-synthetic
uv sync
```

### Smoke test

Verify the installation:

```bash
uv run python smoke_test.py
```

### Single run (default config)

```bash
cd fed-dcn-synthetic
uv run flwr run . local-simulation
```

Hyperparameters are set in `pyproject.toml` under `[tool.flwr.app.config]`.

### Hyperparameter optimisation

Edit `run_experiments.sh` to set the dataset, number of trials, W&B project name, and Optuna storage path, then:

```bash
./run_experiments.sh
```

Each trial samples hyperparameters with Optuna, runs `flwr run` as a subprocess, and logs results to Weights & Biases.

To seed the Optuna study from a previous W&B project before starting:

```bash
uv run python seed_study_from_wandb.py \
    --wandb-project <source-project> \
    --study-name    <study-name> \
    --study-storage sqlite:///optuna.db \
    --objective     accuracy
```

### Repeat best run (ablation study)

Edit `repeat_best_run.sh` to set the source W&B project and any ablation overrides (geometry loss weights, synthetic sample count, federation augmentation flag, etc.), then:

```bash
./repeat_best_run.sh
```

This queries W&B for the best run by the chosen objective, extracts its hyperparameters, and re-runs it `N_RUNS` times with different random seeds.

### Key ablation parameters (`repeat_best_run.sh`)

| Variable | Effect |
|---|---|
| `AUGMENT_FEDERATED="false"` | Disable synthetic data augmentation during Phase 3 |
| `SYNTHETIC_SAMPLES_TOTAL` | Override total synthetic sample budget |
| `ALPHA_GEOM_FEDERATED` | Override geometry loss weight for Phase 3 only |
| `DIRICHLET_ALPHA` | Change data heterogeneity (lower = more non-IID) |
