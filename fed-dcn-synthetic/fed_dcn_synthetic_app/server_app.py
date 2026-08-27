"""Fed-DCN-Synthetic: Flower ServerApp.

Three-phase pipeline:
  Phase 1  — Clients train local DCNs; server collects cluster statistics and
             generates a centralized synthetic dataset by sampling in the latent
             space and decoding with each client's decoder.
  Phase 2  — Server-side pretraining on the synthetic dataset:
               2a. Reconstruction + UMAP geometry loss.
               2b. DCN warm-start (reconstruction + cluster loss).
  Phase 3  — Federated training loop:
               Each round has two sub-rounds:
               (a) AE weights update via FedAvg, each local batch augmented with
                   a same-size batch from the synthetic dataset.
               (b) Cluster center update: clients run k-means on augmented data,
                   server aggregates via k-means on stacked local centers.
             Early stopping when latent Davies-Bouldin index stagnates.
"""

import json
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torch.utils.data import DataLoader
import torch.nn.functional as F

from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.serverapp import Grid, ServerApp

from fed_dcn_synthetic_app.dcn import SyntheticDCNTrainer
from fed_dcn_synthetic_app.task import (
    build_dcn,
    build_decoder,
    load_test_dataset,
    load_train_dataset,
    synthetic_dataloader,
)

app = ServerApp()


# ---------------------------------------------------------------------------
# Communication helpers (matching fed-dcn pattern)
# ---------------------------------------------------------------------------

def _broadcast_and_collect(
    grid: Grid,
    node_ids: list[int],
    content: RecordDict,
    message_type: str,
    round_id: int,
) -> list:
    messages = [
        Message(content, int(nid), message_type, group_id=str(round_id))
        for nid in node_ids
    ]
    msg_ids = list(grid.push_messages(messages))
    pending = set(msg_ids)
    replies = []
    while pending:
        for reply in grid.pull_messages(list(pending)):
            pending.discard(reply.metadata.reply_to_message_id)
            if reply.has_content():
                replies.append(reply)
            else:
                print("  Warning: received error reply from node, skipping.")
    return replies


def _fedavg_ae(replies: list) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Weighted FedAvg over AE state dicts; returns weighted-avg loss metrics."""
    total_examples = sum(float(r.content["metrics"]["num_examples"]) for r in replies)
    avg: dict[str, torch.Tensor] = {}
    loss_agg = {"loss_total": 0.0, "loss_recon": 0.0, "loss_clust": 0.0, "loss_prox": 0.0}
    for reply in replies:
        m = reply.content["metrics"]
        weight = float(m["num_examples"]) / total_examples
        state = reply.content["ae_weights"].to_torch_state_dict()
        for k, v in state.items():
            if k in avg:
                avg[k] += weight * v.float()
            else:
                avg[k] = weight * v.float()
        for loss_key in loss_agg:
            if loss_key in m:
                loss_agg[loss_key] += weight * float(m[loss_key])
    return avg, loss_agg


def _aggregate_cluster_centers(
    replies: list, n_clusters: int, clustering_model
) -> torch.Tensor:
    all_centers = [
        torch.tensor(
            reply.content["local_centers"].to_numpy_ndarrays()[0],
            dtype=torch.float32,
        )
        for reply in replies
    ]
    stacked = torch.cat(all_centers, dim=0)
    clustering_model.update_clusters_from_stacked(stacked)
    return clustering_model.get_cluster_centers()


def _flatten_ae(dcn) -> dict[str, torch.Tensor]:
    flat: dict[str, torch.Tensor] = {}
    for k, v in dcn.encoder.state_dict().items():
        flat[f"encoder.{k}"] = v
    for k, v in dcn.decoder.state_dict().items():
        flat[f"decoder.{k}"] = v
    return flat


def _apply_flat_ae(dcn, flat_ae: dict[str, torch.Tensor]) -> None:
    encoder_state = {k.removeprefix("encoder."): v for k, v in flat_ae.items() if k.startswith("encoder.")}
    decoder_state = {k.removeprefix("decoder."): v for k, v in flat_ae.items() if k.startswith("decoder.")}
    dcn.encoder.load_state_dict(encoder_state)
    dcn.decoder.load_state_dict(decoder_state)


def _make_run_dir(dataset: str) -> Path:
    run_dir = Path("fitted_models") / dataset / datetime.now().strftime("%Y%m%d%H%M")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _pretrain_cache_path(cfg: dict) -> Path:
    """Stable cache path that encodes the network architecture dimensions.

    Format: {cache_dir}/{dataset}_i{input_dim}_h{hidden_dims}_b{bottleneck}_k{n_clusters}.pt
    Example: pretrained_cache/mnist_i784_h256x128_b5_k10.pt
    """
    import json as _json
    hidden_dims = _json.loads(str(cfg["hidden-dims"]))
    hidden_str = "x".join(str(d) for d in hidden_dims)
    name = (
        f"{cfg['dataset']}"
        f"_i{cfg['input-dim']}"
        f"_h{hidden_str}"
        f"_b{cfg['bottleneck-dim']}"
        f"_k{cfg['n-clusters']}"
        ".pt"
    )
    cache_dir = Path(str(cfg.get("pretraining-cache-dir", "pretrained_cache")))
    return cache_dir / name


# ---------------------------------------------------------------------------
# Phase 1: Synthetic dataset generation
# ---------------------------------------------------------------------------

def _proportional_alloc(counts_per_client: list[np.ndarray], total: int) -> np.ndarray:
    """Allocate `total` samples across (client, cluster) pairs proportional to counts.

    Uses the largest-remainder method so the integer allocations sum to exactly `total`.
    Clusters with count=0 receive 0 samples.

    Returns an int array of shape [n_clients, K].
    """
    flat = np.concatenate(counts_per_client).astype(float)
    global_sum = flat.sum()
    if global_sum > 0:
        exact = flat / global_sum * total
    else:
        # degenerate: no data; distribute evenly among non-zero-count positions
        exact = np.ones_like(flat) * total / max(len(flat), 1)
    floors = np.floor(exact).astype(int)
    deficit = total - int(floors.sum())
    if deficit > 0:
        remainders = exact - floors
        top_idx = np.argsort(remainders)[::-1][:deficit]
        floors[top_idx] += 1
    K = len(counts_per_client[0])
    return floors.reshape(len(counts_per_client), K)


def _generate_synthetic_dataset(
    phase1_replies: list,
    cfg: dict,
    device: torch.device,
    save_path: Path,
    samples_per_cluster: int | None = None,
    synthetic_samples_total: int | None = None,
    stats_dir: Path | None = None,
) -> torch.Tensor:
    """Generate synthetic dataset from client cluster statistics and decoder weights.

    Sampling mode (checked in order):
      1. synthetic_samples_total set → proportional allocation: each client/cluster
         receives a share of the global budget proportional to its within-1-std count.
         Total synthetic samples equals synthetic_samples_total exactly.
      2. samples_per_cluster set → fixed count per non-empty cluster (original mode).
      3. Neither → use raw within-1-std counts as sample budgets.

    If `stats_dir` is provided, writes cluster_stats_summary.json there.

    Returns the synthetic dataset as a float32 tensor of shape [N, input_dim].
    """
    n_clusters = int(cfg["n-clusters"])
    bottleneck_dim = int(cfg["bottleneck-dim"])
    all_images: list[torch.Tensor] = []
    all_client_stats: list[dict] = []

    # Pre-compute per-(client, cluster) allocation for proportional mode.
    if synthetic_samples_total is not None:
        counts_per_client = [
            reply.content["cluster_stats"].to_numpy_ndarrays()[1]
            for reply in phase1_replies
        ]
        alloc = _proportional_alloc(counts_per_client, synthetic_samples_total)
    else:
        alloc = None

    for i, reply in enumerate(phase1_replies):
        centers_np   = reply.content["cluster_centers"].to_numpy_ndarrays()[0]  # [K, D]
        stats_arrays = reply.content["cluster_stats"].to_numpy_ndarrays()
        std_devs_np  = stats_arrays[0]   # [K]
        counts_np    = stats_arrays[1]   # [K] within-1-std counts

        centers  = torch.tensor(centers_np,  dtype=torch.float32)
        std_devs = torch.tensor(std_devs_np, dtype=torch.float32)

        decoder = build_decoder(cfg)
        decoder_state = reply.content["decoder_weights"].to_torch_state_dict()
        decoder.load_state_dict(decoder_state)
        decoder = decoder.to(device).eval()

        n_nonempty = int((counts_np > 0).sum())

        cluster_rows: list[dict] = []
        total_samples = 0
        with torch.no_grad():
            for k in range(n_clusters):
                if alloc is not None:
                    n_samples = int(alloc[i, k])
                elif samples_per_cluster is not None:
                    n_samples = samples_per_cluster if counts_np[k] > 0 else 0
                else:
                    n_samples = int(round(float(counts_np[k]))) if counts_np[k] > 0 else 0

                total_samples += n_samples
                cluster_rows.append({
                    "k": k,
                    "count": float(counts_np[k]),
                    "std_dev": float(std_devs_np[k]),
                    "n_synth": n_samples,
                })
                if n_samples == 0:
                    continue

                center_k = centers[k].to(device)
                std_k    = max(float(std_devs[k].item()), 1e-6)
                noise     = torch.randn(n_samples, bottleneck_dim, device=device)
                z_samples = center_k.unsqueeze(0) + std_k * noise
                x_decoded = decoder(z_samples).cpu()
                all_images.append(x_decoded)

        nonempty_counts = counts_np[counts_np > 0]
        cv = float(nonempty_counts.std() / nonempty_counts.mean()) if len(nonempty_counts) > 1 else 0.0

        print(
            f"  Client {i + 1}/{len(phase1_replies)}: "
            f"{n_nonempty}/{n_clusters} non-empty clusters, "
            f"{total_samples} synthetic samples  (count CV={cv:.2f})"
        )
        col_w = (5, 8, 9, 8)
        print(f"  {'k':>{col_w[0]}}  {'count':>{col_w[1]}}  {'std_dev':>{col_w[2]}}  {'n_synth':>{col_w[3]}}")
        print("  " + "  ".join("-" * w for w in col_w))
        for row in cluster_rows:
            empty_flag = "  <- empty" if row["n_synth"] == 0 else ""
            print(
                f"  {row['k']:>{col_w[0]}}  {row['count']:>{col_w[1]}.1f}"
                f"  {row['std_dev']:>{col_w[2]}.4f}  {row['n_synth']:>{col_w[3]}}{empty_flag}"
            )

        all_client_stats.append({
            "client_index": i,
            "n_clusters": n_clusters,
            "n_nonempty": n_nonempty,
            "total_samples_generated": total_samples,
            "count_cv": round(cv, 4),
            "clusters": cluster_rows,
        })

    synthetic = torch.cat(all_images, dim=0)
    synthetic = synthetic.clamp(0.0, 1.0)
    torch.save(synthetic, save_path)

    per_client_totals = [c["total_samples_generated"] for c in all_client_stats]
    per_client_cvs    = [c["count_cv"] for c in all_client_stats]
    print(
        f"\n  Synthetic dataset: {synthetic.shape[0]} samples total"
        f" | per-client: {per_client_totals}"
        f" | count CV per client: {[round(v, 2) for v in per_client_cvs]}"
    )
    print(f"  Saved → {save_path}")

    if stats_dir is not None:
        summary = {
            "n_clients": len(phase1_replies),
            "sampling_mode": "proportional" if alloc is not None else ("per_cluster" if samples_per_cluster else "raw_counts"),
            "synthetic_samples_total_config": synthetic_samples_total,
            "samples_per_cluster_config": samples_per_cluster,
            "total_synthetic_samples": int(synthetic.shape[0]),
            "clients": all_client_stats,
        }
        with open(stats_dir / "cluster_stats_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Cluster stats → {stats_dir / 'cluster_stats_summary.json'}")

    return synthetic


# ---------------------------------------------------------------------------
# Phase 2: UMAP precomputation
# ---------------------------------------------------------------------------

def _compute_umap_embeddings(
    synthetic: torch.Tensor,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    seed: int,
    batch_size: int,
) -> DataLoader:
    """Compute UMAP embeddings of the synthetic dataset.

    Returns a DataLoader that yields (umap_embedding, x) pairs, matching the
    geometric dataloader format used by SyntheticDCNTrainer._compute_geom_loss.
    """
    import umap as umap_lib
    data_np = synthetic.cpu().numpy()
    print(f"  Computing UMAP on synthetic dataset (n={data_np.shape[0]}, n_components={n_components})...")
    reducer = umap_lib.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=seed,
    )
    embeddings_np = reducer.fit_transform(data_np)
    print(f"  UMAP complete: shape {embeddings_np.shape}")

    embeddings_t = torch.tensor(embeddings_np, dtype=torch.float32)
    from torch.utils.data import TensorDataset
    geom_dataset = TensorDataset(embeddings_t, synthetic)
    return DataLoader(geom_dataset, batch_size=batch_size, shuffle=True)


# ---------------------------------------------------------------------------
# Evaluation on central test set
# ---------------------------------------------------------------------------

def _evaluate_centralized(dcn, dataloader: DataLoader, server_round: int) -> dict:
    device = next(dcn.parameters()).device
    dcn.eval()

    all_latent: list[torch.Tensor] = []
    all_observed: list[torch.Tensor] = []
    all_true_labels: list[torch.Tensor] = []
    all_reconstructions: list[torch.Tensor] = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            z = dcn.encoder(x)
            x_recon = dcn.decoder(z)
            all_latent.append(z.cpu())
            all_observed.append(x.cpu())
            all_true_labels.append(y)
            all_reconstructions.append(x_recon.cpu())

    latent = torch.cat(all_latent, dim=0)
    observed = torch.cat(all_observed, dim=0)
    true_labels = torch.cat(all_true_labels, dim=0).numpy()
    reconstructions = torch.cat(all_reconstructions, dim=0)
    recon_loss = float(F.mse_loss(reconstructions, observed).item())

    centers = dcn.cluster_model.get_cluster_centers()
    if centers is None:
        raise ValueError("Cluster centres not initialized for evaluation.")
    dcn.cluster_model.set_clusters(centers.to(latent.device))

    pred_assignments = dcn.cluster_model.get_cluster_assignments(latent).numpy()
    n_clusters = int(dcn.n_clusters)
    n_classes = int(true_labels.max()) + 1

    confusion = np.zeros((n_clusters, n_classes), dtype=np.int64)
    for pred, true in zip(pred_assignments, true_labels):
        confusion[pred, true] += 1
    row_ind, col_ind = linear_sum_assignment(-confusion)
    acc = float(confusion[row_ind, col_ind].sum()) / len(true_labels)

    nmi = float(normalized_mutual_info_score(true_labels, pred_assignments))
    ari = float(adjusted_rand_score(true_labels, pred_assignments))

    db_latent = dcn.cluster_model.compute_db_index(latent)
    if isinstance(db_latent, torch.Tensor):
        db_latent = db_latent.item()
    db_observed = dcn.cluster_model.compute_db_index_with_labels(observed, pred_assignments)
    db_gt_observed = dcn.cluster_model.compute_db_index_with_labels(observed, true_labels)

    metrics = {
        "round":         server_round,
        "acc":           acc,
        "nmi":           nmi,
        "ari":           ari,
        "db_latent":     float(db_latent),
        "db_observed":   float(db_observed),
        "db_gt_observed": float(db_gt_observed),
        "recon_loss":    recon_loss,
    }
    print(
        f"[Round {server_round}] "
        f"ACC={acc:.4f}  NMI={nmi:.4f}  ARI={ari:.4f}  "
        f"DB_lat={db_latent:.4f}  DB_obs={db_observed:.4f}  "
        f"DB_gt={db_gt_observed:.4f}  Recon={recon_loss:.4f}"
    )
    return metrics


# ---------------------------------------------------------------------------
# Main server entry point
# ---------------------------------------------------------------------------

@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = context.run_config
    device = _device()

    dataset            = str(cfg["dataset"])
    batch_size         = int(cfg["batch-size"])
    seed               = int(cfg["random-seed"])
    n_clusters         = int(cfg["n-clusters"])
    lr                 = float(cfg["learning-rate"])
    lr_pretrain        = float(cfg.get("learning-rate-pretrain", lr))
    alpha_recon        = float(cfg.get("alpha-reconstruction-loss", 1.0))

    local_pretrain_epochs  = int(cfg["local-pretrain-epochs"])
    local_training_epochs  = int(cfg["local-training-epochs"])
    _spc = cfg.get("synthetic-samples-per-cluster")
    samples_per_cluster: int | None = int(_spc) if _spc is not None else None
    _sst = cfg.get("synthetic-samples-total")
    synthetic_samples_total: int | None = int(_sst) if _sst is not None else None

    phase2a_epochs       = int(cfg["phase2a-epochs"])
    phase2b_epochs       = int(cfg["phase2b-epochs"])
    alpha_geom           = float(cfg["alpha-geom"])
    alpha_geom_warmstart = float(cfg.get("alpha-geom-warmstart", 0.0))
    alpha_geom_federated = float(cfg.get("alpha-geom-federated", 0.0))
    _af = str(cfg.get("augment-federated", "true")).lower()
    augment_federated    = _af not in ("false", "0")
    beta_clust_ws        = float(cfg["beta-clust-warmstart"])
    umap_components = int(cfg.get("umap-n-components", int(cfg["bottleneck-dim"])))
    umap_neighbors  = int(cfg.get("umap-n-neighbors", 15))
    umap_min_dist   = float(cfg.get("umap-min-dist", 0.1))

    num_rounds        = int(cfg["num-server-rounds"])
    early_stop_rounds = int(cfg.get("early-stopping-rounds", 10))
    early_stop_metric = str(cfg.get("early-stopping-metric", "db_latent"))
    train_eval_fraction = float(cfg.get("train-eval-fraction", 1.0))

    node_ids = list(grid.get_node_ids())
    print(f"\n=== Fed-DCN-Synthetic ===")
    print(f"Clients: {len(node_ids)}")
    print(f"Dataset: {dataset}, n_clusters: {n_clusters}, batch_size: {batch_size}")

    run_dir = _make_run_dir(dataset)
    print(f"Run output: {run_dir}")

    central_test_loader = load_test_dataset(dataset=dataset, batch_size=batch_size)
    central_train_eval_loader = load_train_dataset(
        dataset=dataset, batch_size=batch_size, fraction=train_eval_fraction, seed=seed
    )

    pretrain_cache = _pretrain_cache_path(cfg)
    _reuse_raw = cfg.get("reuse-pretraining", False)
    reuse_pretraining = _reuse_raw if isinstance(_reuse_raw, bool) else str(_reuse_raw).lower() == "true"

    if reuse_pretraining and pretrain_cache.exists():
        # -------------------------------------------------------------------
        # Load pretraining cache (skip Phase 1+2)
        # -------------------------------------------------------------------
        print(f"\n=== Loading pretraining cache: {pretrain_cache} ===")
        checkpoint = torch.load(pretrain_cache, weights_only=False)
        flat_ae = checkpoint["flat_ae"]
        cluster_centers = checkpoint["cluster_centers"].cpu()
        synthetic_np = checkpoint["synthetic_np"]
        synthetic = torch.from_numpy(synthetic_np)
        umap_embeddings_np = checkpoint.get("umap_embeddings_np")
        if umap_embeddings_np is None and alpha_geom_federated > 0.0:
            print("  Warning: cache has no UMAP embeddings; alpha-geom-federated will be ignored.")

        dcn = build_dcn(cfg)
        dcn = dcn.to(device)
        _apply_flat_ae(dcn, flat_ae)
        dcn.cluster_model.set_clusters(cluster_centers.to(device))

        synthetic_loader = synthetic_dataloader(synthetic, batch_size=batch_size, shuffle=True)
        print(
            f"  Loaded {synthetic.shape[0]} synthetic samples, "
            f"{cluster_centers.shape[0]} cluster centres."
        )
    else:
        if reuse_pretraining:
            print(f"\n  No matching cache at {pretrain_cache} — running Phase 1+2 from scratch.")

        # -------------------------------------------------------------------
        # Phase 1: Local DCN training → synthetic dataset generation
        # -------------------------------------------------------------------
        print("\n=== Phase 1: Local DCN training ===")

        # Always use exactly 10 randomly sampled clients for synthetic data
        # generation so that samples_per_cluster fixes the dataset size
        # independently of the total number of clients.
        _rng = np.random.default_rng(seed)
        _n_synth_clients = len(node_ids) #min(10, len(node_ids))
        synth_node_ids = [node_ids[i] for i in _rng.choice(len(node_ids), size=_n_synth_clients, replace=False)]
        print(f"  Sampling {_n_synth_clients} clients for synthetic data: {synth_node_ids}")

        dcn = build_dcn(cfg)
        dcn = dcn.to(device)
        flat_ae = _flatten_ae(dcn)

        phase1_content = RecordDict({
            "ae_weights": ArrayRecord(torch_state_dict=flat_ae),
            "config": ConfigRecord({
                "phase":                  "local_dcn_training",
                "local-pretrain-epochs":  local_pretrain_epochs,
                "local-training-epochs":  local_training_epochs,
            }),
        })
        phase1_replies = _broadcast_and_collect(
            grid, synth_node_ids, phase1_content, "train", round_id=-1
        )
        print(f"Received cluster statistics from {len(phase1_replies)} clients.")

        print("\n  Generating synthetic dataset...")
        p1_dir = run_dir / "output_phase1"
        p1_dir.mkdir(parents=True, exist_ok=True)
        synthetic = _generate_synthetic_dataset(
            phase1_replies=phase1_replies,
            cfg=cfg,
            device=device,
            save_path=run_dir / "synthetic_dataset.pt",
            samples_per_cluster=samples_per_cluster,
            synthetic_samples_total=synthetic_samples_total,
            stats_dir=p1_dir,
        )
        synthetic_np = synthetic.numpy().astype(np.float32)

        synthetic_loader = synthetic_dataloader(synthetic, batch_size=batch_size, shuffle=True)

        # Save Phase 1 outputs
        torch.save(synthetic, p1_dir / "synthetic_dataset.pt")
        for i, reply in enumerate(phase1_replies):
            centers_np    = reply.content["cluster_centers"].to_numpy_ndarrays()[0]
            stats         = reply.content["cluster_stats"].to_numpy_ndarrays()
            std_devs_np   = stats[0]
            counts_np     = stats[1]
            decoder_state = reply.content["decoder_weights"].to_torch_state_dict()
            np.save(p1_dir / f"client_{i:02d}_centers.npy",  centers_np)
            np.save(p1_dir / f"client_{i:02d}_std_devs.npy", std_devs_np)
            np.save(p1_dir / f"client_{i:02d}_counts.npy",   counts_np)
            torch.save(decoder_state, p1_dir / f"client_{i:02d}_decoder.pt")
        np.save(p1_dir / "synth_node_ids.npy", np.array(synth_node_ids))
        print(f"  Saved Phase 1 outputs ({len(phase1_replies)} clients) → {p1_dir}")

        # -------------------------------------------------------------------
        # Phase 2a: Geometry-aware pretraining on synthetic dataset
        # -------------------------------------------------------------------
        print("\n=== Phase 2a: UMAP pretraining on synthetic dataset ===")
        umap_loader = _compute_umap_embeddings(
            synthetic=synthetic,
            n_components=umap_components,
            n_neighbors=umap_neighbors,
            min_dist=umap_min_dist,
            seed=seed,
            batch_size=batch_size,
        )
        umap_embeddings_np = umap_loader.dataset.tensors[0].cpu().numpy().astype(np.float32)

        dcn = build_dcn(cfg)
        dcn = dcn.to(device)

        pretrain_trainer = SyntheticDCNTrainer(
            dcn=dcn,
            dataloader=synthetic_loader,
            learning_rate=lr_pretrain,
            alpha_recon=alpha_recon,
            geometric_dataloader=umap_loader,
        )
        pretrain_trainer.run_geometry_pretrain(
            n_epochs=phase2a_epochs,
            alpha_geom=alpha_geom,
            verbose=True,
        )
        print(f"  Phase 2a complete. Cluster centres initialized from synthetic data.")

        # -------------------------------------------------------------------
        # Phase 2b: DCN warm-start on synthetic dataset
        # -------------------------------------------------------------------
        print("\n=== Phase 2b: DCN warm-start on synthetic dataset ===")
        pretrain_trainer.run_warmstart(
            n_epochs=phase2b_epochs,
            beta_clust=beta_clust_ws,
            alpha_geom=alpha_geom_warmstart,
            verbose=True,
        )
        cluster_centers = dcn.cluster_model.get_cluster_centers().cpu()
        print(f"  Phase 2b complete. Cluster centres shape: {cluster_centers.shape}")
        flat_ae = _flatten_ae(dcn)

        # Save Phase 2 outputs
        p2_dir = run_dir / "output_phase2"
        p2_dir.mkdir(parents=True, exist_ok=True)
        torch.save(synthetic, p2_dir / "synthetic_dataset.pt")
        torch.save({k: v.detach().cpu().clone() for k, v in flat_ae.items()}, p2_dir / "ae_weights.pt")
        torch.save(cluster_centers.clone(), p2_dir / "cluster_centers.pt")
        print(f"  Saved Phase 2 outputs → {p2_dir}")

        if reuse_pretraining:
            # Save per-run checkpoint inside the timestamped run directory
            pretrain_path = run_dir / "pretrained_weights.pt"
            torch.save(
                {
                    "flat_ae": {k: v.detach().cpu().clone() for k, v in flat_ae.items()},
                    "cluster_centers": cluster_centers.clone(),
                },
                pretrain_path,
            )
            print(f"  Saved per-run pretrained checkpoint → {pretrain_path}")

            # Save to stable cache (overrides any existing cache for this architecture)
            pretrain_cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "flat_ae": {k: v.detach().cpu().clone() for k, v in flat_ae.items()},
                    "cluster_centers": cluster_centers.clone(),
                    "synthetic_np": synthetic_np,
                    "umap_embeddings_np": umap_embeddings_np,
                },
                pretrain_cache,
            )
            print(f"  Saved pretraining cache → {pretrain_cache}")

    # Round-0 evaluation (post-pretraining baseline)
    print("\n=== Evaluation Round 0 (post-pretraining) ===")
    print("[Round 0][Train]")
    train0 = _evaluate_centralized(dcn=dcn, dataloader=central_train_eval_loader, server_round=0)
    print("[Round 0][Test]")
    test0  = _evaluate_centralized(dcn=dcn, dataloader=central_test_loader, server_round=0)
    round0_metrics: dict = {"round": 0}
    round0_metrics.update({f"train_{k}": v for k, v in train0.items() if k != "round"})
    round0_metrics.update({f"test_{k}": v for k, v in test0.items() if k != "round"})
    round0_metrics.update({"loss_total": 0.0, "loss_recon": 0.0, "loss_clust": 0.0, "loss_prox": 0.0})
    metrics_history: list[dict] = [round0_metrics]

    if early_stop_metric == "accuracy":
        _init_val = float(round0_metrics["train_acc"])
        best_metric_val = _init_val if np.isfinite(_init_val) else float("-inf")
        _metric_is_better = lambda cur, best: np.isfinite(cur) and cur > best  # noqa: E731
    else:
        _init_val = float(round0_metrics["train_db_latent"])
        best_metric_val = _init_val if np.isfinite(_init_val) else float("inf")
        _metric_is_better = lambda cur, best: np.isfinite(cur) and cur < best  # noqa: E731

    best_db_latent = float(round0_metrics["train_db_latent"]) if np.isfinite(round0_metrics["train_db_latent"]) else float("inf")
    best_acc = float(round0_metrics["train_acc"])
    best_round = 0
    best_flat_ae = {k: v.detach().cpu().clone() for k, v in flat_ae.items()}
    best_cluster_centers = cluster_centers.clone()
    rounds_without_improvement = 0
    stopped_early = False

    print(f"  Early stopping metric: {early_stop_metric}, initial value: {best_metric_val:.4f}")

    # -----------------------------------------------------------------------
    # Phase 3: Federated training
    # -----------------------------------------------------------------------
    print("\n=== Phase 3: Federated training ===")

    for fed_round in range(1, num_rounds + 1):
        print(f"\n--- Federation Round {fed_round}/{num_rounds} ---")

        base_content = {
            "ae_weights":      ArrayRecord(torch_state_dict=flat_ae),
            "cluster_centers": ArrayRecord(numpy_ndarrays=[cluster_centers.cpu().numpy()]),
        }
        if augment_federated:
            base_content["synthetic_data"] = ArrayRecord(numpy_ndarrays=[synthetic_np])
        if umap_embeddings_np is not None:
            base_content["umap_embeddings"] = ArrayRecord(numpy_ndarrays=[umap_embeddings_np])

        # --- 3a: AE weights update ---
        ae_content = RecordDict({
            **base_content,
            "config": ConfigRecord({
                "phase":        "ae_weights",
                "server-round": fed_round,
            }),
        })
        ae_replies = _broadcast_and_collect(
            grid, node_ids, ae_content, "train", round_id=fed_round * 10 + 1
        )
        flat_ae, loss_metrics = _fedavg_ae(ae_replies)
        _apply_flat_ae(dcn, flat_ae)
        print(
            f"  AE phase ({len(ae_replies)} clients)  "
            f"loss_total={loss_metrics['loss_total']:.4f}  "
            f"loss_recon={loss_metrics['loss_recon']:.4f}  "
            f"loss_clust={loss_metrics['loss_clust']:.4f}  "
            f"loss_prox={loss_metrics['loss_prox']:.4f}"
        )

        # --- 3b: Cluster center update ---
        cluster_content = RecordDict({
            **base_content,
            "ae_weights": ArrayRecord(torch_state_dict=flat_ae),  # use freshly averaged weights
            "config": ConfigRecord({"phase": "cluster_centers"}),
        })
        cluster_replies = _broadcast_and_collect(
            grid, node_ids, cluster_content, "train", round_id=fed_round * 10 + 2
        )
        prev_cluster_centers = cluster_centers.clone()
        cluster_centers = _aggregate_cluster_centers(
            cluster_replies, n_clusters, dcn.cluster_model
        ).cpu()
        center_shift = float(torch.norm(cluster_centers - prev_cluster_centers).item())
        print(
            f"  Cluster phase ({len(cluster_replies)} clients): k-means aggregation complete. "
            f"||Δcenters||_F={center_shift:.4f}"
        )

        # --- Evaluate on train (HPO / early stopping) and test (logging) ---
        dcn.cluster_model.set_clusters(cluster_centers.to(device))
        print(f"[Round {fed_round}][Train]")
        train_metrics = _evaluate_centralized(dcn=dcn, dataloader=central_train_eval_loader, server_round=fed_round)
        print(f"[Round {fed_round}][Test]")
        test_metrics  = _evaluate_centralized(dcn=dcn, dataloader=central_test_loader, server_round=fed_round)
        round_metrics: dict = {"round": fed_round}
        round_metrics.update({f"train_{k}": v for k, v in train_metrics.items() if k != "round"})
        round_metrics.update({f"test_{k}": v for k, v in test_metrics.items() if k != "round"})
        round_metrics.update(loss_metrics)
        round_metrics["center_shift"] = center_shift
        metrics_history.append(round_metrics)

        # --- Early stopping (based on train metrics) ---
        current_metric = float(round_metrics["train_acc" if early_stop_metric == "accuracy" else "train_db_latent"])

        if _metric_is_better(current_metric, best_metric_val):
            best_metric_val = current_metric
            best_acc = float(round_metrics["train_acc"])
            best_db_latent = float(round_metrics["train_db_latent"])
            best_round = fed_round
            rounds_without_improvement = 0
            best_flat_ae = {k: v.detach().cpu().clone() for k, v in flat_ae.items()}
            best_cluster_centers = cluster_centers.clone()
            print(f"  New best {early_stop_metric}: {best_metric_val:.4f} (round {best_round})")
        else:
            rounds_without_improvement += 1
            print(f"  No improvement ({rounds_without_improvement}/{early_stop_rounds}).")

        if early_stop_rounds > 0 and rounds_without_improvement >= early_stop_rounds:
            stopped_early = True
            print(f"Early stopping: {early_stop_metric} did not improve for {early_stop_rounds} rounds.")
            break

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    print(f"\nSaving outputs to {run_dir} ...")
    torch.save(best_flat_ae, run_dir / "final_ae_weights.pt")
    torch.save(best_cluster_centers, run_dir / "final_cluster_centers.pt")

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics_history, f, indent=2)

    run_config_serialisable = {k: str(v) for k, v in cfg.items()}
    run_config_serialisable["num_clients"] = str(len(node_ids))
    run_config_serialisable["best_round"] = str(best_round)
    run_config_serialisable["best_db_latent"] = str(best_db_latent)
    run_config_serialisable["best_acc"] = str(best_acc)
    run_config_serialisable["stopped_early"] = str(stopped_early)
    with open(run_dir / "run_config.json", "w") as f:
        json.dump(run_config_serialisable, f, indent=2)

    print(
        f"Best {early_stop_metric}: {best_metric_val:.4f} at round {best_round} "
        f"(DB_latent={best_db_latent:.4f}, ACC={best_acc:.4f}). "
        f"Saved final_ae_weights.pt, final_cluster_centers.pt, metrics.json, run_config.json."
    )
