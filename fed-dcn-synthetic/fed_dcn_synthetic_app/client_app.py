"""Fed-DCN-Synthetic: Flower ClientApp."""

import traceback

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fed_dcn_synthetic_app.dcn import SyntheticDCNTrainer
from fed_dcn_synthetic_app.task import build_dcn, load_partition, synthetic_dataloader

app = ClientApp()


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_partition(context: Context):
    cfg = context.run_config
    return load_partition(
        partition_id=int(context.node_config["partition-id"]),
        num_partitions=int(context.node_config["num-partitions"]),
        alpha=float(cfg["dirichlet-alpha"]),
        dataset=str(cfg["dataset"]),
        batch_size=int(cfg["batch-size"]),
        seed=int(cfg["random-seed"]),
    )


def _load_ae_weights(msg: Message, dcn) -> None:
    """Load AE weights from a message ArrayRecord into a DCN in-place."""
    ae_state = msg.content["ae_weights"].to_torch_state_dict()
    encoder_state = {k.removeprefix("encoder."): v for k, v in ae_state.items() if k.startswith("encoder.")}
    decoder_state = {k.removeprefix("decoder."): v for k, v in ae_state.items() if k.startswith("decoder.")}
    dcn.encoder.load_state_dict(encoder_state)
    dcn.decoder.load_state_dict(decoder_state)


def _cluster_centers_from_msg(msg: Message) -> torch.Tensor | None:
    record = msg.content.get("cluster_centers")
    if record is None:
        return None
    arrays = record.to_numpy_ndarrays()
    if not arrays:
        return None
    return torch.tensor(arrays[0], dtype=torch.float32)


def _compute_local_geom_loader(
    trainloader: torch.utils.data.DataLoader,
    cfg: dict,
) -> torch.utils.data.DataLoader:
    """Compute UMAP on local training data and return a geometric DataLoader of (embedding, x) pairs."""
    import umap as umap_lib
    from torch.utils.data import TensorDataset

    n_components = int(cfg.get("umap-n-components", int(cfg["bottleneck-dim"])))
    n_neighbors  = int(cfg.get("umap-n-neighbors", 15))
    min_dist     = float(cfg.get("umap-min-dist", 0.1))
    seed         = int(cfg["random-seed"])
    batch_size   = int(cfg["batch-size"])

    all_x: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in trainloader:
            all_x.append(batch[0])
    data = torch.cat(all_x, dim=0)

    reducer = umap_lib.UMAP(
        n_components=n_components,
        n_neighbors=min(n_neighbors, data.shape[0] - 1),
        min_dist=min_dist,
        metric="euclidean",
        random_state=seed,
    )
    embeddings_t = torch.tensor(reducer.fit_transform(data.numpy()), dtype=torch.float32)
    return torch.utils.data.DataLoader(
        TensorDataset(embeddings_t, data),
        batch_size=batch_size,
        shuffle=True,
    )


def _umap_geom_loader_from_msg(msg: Message, batch_size: int) -> torch.utils.data.DataLoader | None:
    """Build a geometric DataLoader from server-sent UMAP embeddings and synthetic data."""
    from torch.utils.data import TensorDataset
    umap_record  = msg.content.get("umap_embeddings")
    synth_record = msg.content.get("synthetic_data")
    if umap_record is None or synth_record is None:
        return None
    umap_arrays  = umap_record.to_numpy_ndarrays()
    synth_arrays = synth_record.to_numpy_ndarrays()
    if not umap_arrays or not synth_arrays:
        return None
    umap_t  = torch.tensor(umap_arrays[0],  dtype=torch.float32)
    synth_t = torch.tensor(synth_arrays[0], dtype=torch.float32)
    return torch.utils.data.DataLoader(
        TensorDataset(umap_t, synth_t),
        batch_size=batch_size,
        shuffle=True,
    )


def _synthetic_loader_from_msg(msg: Message, batch_size: int) -> torch.utils.data.DataLoader | None:
    record = msg.content.get("synthetic_data")
    if record is None:
        return None
    arrays = record.to_numpy_ndarrays()
    if not arrays:
        return None
    images = torch.tensor(arrays[0], dtype=torch.float32)
    return synthetic_dataloader(images, batch_size=batch_size, shuffle=True)


def _flatten_ae(dcn) -> dict[str, torch.Tensor]:
    flat: dict[str, torch.Tensor] = {}
    for k, v in dcn.encoder.state_dict().items():
        flat[f"encoder.{k}"] = v
    for k, v in dcn.decoder.state_dict().items():
        flat[f"decoder.{k}"] = v
    return flat


def _compute_cluster_stats(
    dcn, trainloader, device: torch.device | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-cluster centres, std deviations, and point counts within 1-std radius."""
    if device is None:
        device = next(dcn.parameters()).device
    all_z: list[torch.Tensor] = []
    all_a: list[torch.Tensor] = []
    dcn.eval()
    with torch.no_grad():
        for batch in trainloader:
            x = batch[0].to(device)
            z = dcn.encoder(x)
            a = dcn.cluster_model.get_cluster_assignments(z)
            all_z.append(z.cpu())
            all_a.append(a.cpu())
    dcn.train()
    all_z = torch.cat(all_z, dim=0)
    all_a = torch.cat(all_a, dim=0)
    centers = dcn.cluster_model.get_cluster_centers().cpu()
    K = centers.shape[0]
    std_devs = np.zeros(K, dtype=np.float32)
    counts = np.zeros(K, dtype=np.float32)
    for k in range(K):
        mask = all_a == k
        if mask.sum() == 0:
            std_devs[k] = 1.0
            counts[k] = 0.0
            continue
        z_k = all_z[mask]
        dists = torch.norm(z_k - centers[k], dim=1)
        std_k = float(dists.std().item()) if dists.shape[0] > 1 else 1.0
        std_devs[k] = max(std_k, 1e-6)
        # counts[k] = float((dists <= std_k).sum().item()) # used in all cases except 20 clients case on the USPS dataset
        counts[k] = mask.sum().item() # used in the 20 clients case on the USPS dataset due to the small number of points in each local cluster
    return centers.numpy(), std_devs, counts


# ---------------------------------------------------------------------------
# Train handler
# ---------------------------------------------------------------------------

@app.train()
def train(msg: Message, context: Context) -> Message:
    try:
        return _train_impl(msg, context)
    except Exception:
        print(f"[CLIENT ERROR node={context.node_id}]\n{traceback.format_exc()}")
        raise


def _train_impl(msg: Message, context: Context) -> Message:
    phase = str(msg.content["config"]["phase"])
    device = _device()
    cfg = context.run_config
    trainloader, _ = _load_partition(context)
    batch_size = int(cfg["batch-size"])

    # ------------------------------------------------------------------
    # Phase 1: Local DCN training
    # ------------------------------------------------------------------
    if phase == "local_dcn_training":
        local_pretrain_epochs = int(msg.content["config"].get("local-pretrain-epochs", cfg["local-pretrain-epochs"]))
        local_training_epochs = int(msg.content["config"].get("local-training-epochs", cfg["local-training-epochs"]))
        lambda_clust = float(cfg["lambda-clust-loss"])
        lr_local = float(cfg.get("learning-rate-local", cfg["learning-rate"]))
        dp_noise_sigma = float(cfg.get("dp-noise-sigma", 0.0))

        alpha_geom_local = float(cfg.get("alpha-geom-local", 0.0))

        dcn = build_dcn(cfg, cluster_centers=None)
        _load_ae_weights(msg, dcn)
        dcn = dcn.to(device)

        local_geom_loader = None
        if alpha_geom_local > 0.0:
            local_geom_loader = _compute_local_geom_loader(trainloader, cfg)

        trainer = SyntheticDCNTrainer(
            dcn=dcn,
            dataloader=trainloader,
            learning_rate=lr_local,
            alpha_recon=float(cfg.get("alpha-reconstruction-loss", 1.0)),
            geometric_dataloader=local_geom_loader,
        )

        # Step 1: AE pretraining (reconstruction only) → k-means cluster init
        trainer.run_ae_pretrain(n_epochs=local_pretrain_epochs, alpha_geom=alpha_geom_local, verbose=False)

        # Step 2: DCN training — alternate AE update and k-means cluster update
        trainer.run_federated_ae(
            n_epochs=local_training_epochs,
            beta_clust=lambda_clust,
            alpha_geom=alpha_geom_local,
            early_stopping=None,
            update_clusters_each_epoch=True,
            verbose=False,
        )

        # Compute cluster statistics for synthetic dataset generation
        centers_np, std_devs_np, counts_np = _compute_cluster_stats(dcn, trainloader, device)

        # Apply DP noise to shared statistics (hook for differential privacy)
        if dp_noise_sigma > 0.0:
            rng = np.random.default_rng(int(cfg["random-seed"]) + int(context.node_id))
            centers_np = centers_np + dp_noise_sigma * rng.standard_normal(centers_np.shape).astype(np.float32)
            std_devs_np = np.maximum(1e-6, std_devs_np + dp_noise_sigma * np.abs(rng.standard_normal(std_devs_np.shape).astype(np.float32)))
            counts_np = np.maximum(0.0, counts_np + dp_noise_sigma * rng.standard_normal(counts_np.shape).astype(np.float32))

        decoder_state = dcn.decoder.state_dict()

        content = RecordDict({
            "cluster_centers": ArrayRecord(numpy_ndarrays=[centers_np]),
            "decoder_weights": ArrayRecord(torch_state_dict=decoder_state),
            "cluster_stats":   ArrayRecord(numpy_ndarrays=[std_devs_np, counts_np]),
            "metrics":         MetricRecord({"num_examples": float(len(trainloader.dataset))}),
        })
        return Message(content=content, reply_to=msg)

    # ------------------------------------------------------------------
    # Phase 3a: AE weights update
    # ------------------------------------------------------------------
    elif phase == "ae_weights":
        cluster_centers = _cluster_centers_from_msg(msg)
        if cluster_centers is not None:
            cluster_centers = cluster_centers.to(device)

        dcn = build_dcn(cfg, cluster_centers=cluster_centers)
        _load_ae_weights(msg, dcn)
        dcn = dcn.to(device)

        synthetic_loader = _synthetic_loader_from_msg(msg, batch_size)
        fedprox_mu       = float(cfg.get("fedprox-mu", 0.0))
        lr_fed           = float(cfg.get("learning-rate-federated", cfg["learning-rate"]))
        lambda_clust     = float(cfg["lambda-clust-loss"])
        alpha_recon      = float(cfg.get("alpha-reconstruction-loss", 1.0))
        alpha_geom_fed   = float(cfg.get("alpha-geom-federated", 0.0))
        local_epochs     = int(cfg["local-epochs"])

        fed_geom_loader = _umap_geom_loader_from_msg(msg, batch_size) if alpha_geom_fed > 0.0 else None

        trainer = SyntheticDCNTrainer(
            dcn=dcn,
            dataloader=trainloader,
            learning_rate=lr_fed,
            alpha_recon=alpha_recon,
            fedprox_mu=fedprox_mu,
            augmentation_dataloader=synthetic_loader,
            geometric_dataloader=fed_geom_loader,
        )

        if fedprox_mu > 0:
            ae_state = msg.content["ae_weights"].to_torch_state_dict()
            trainer.set_global_ae_params(ae_state)

        history = trainer.run_federated_ae(
            n_epochs=local_epochs,
            beta_clust=lambda_clust,
            alpha_geom=alpha_geom_fed,
            verbose=False,
        )

        n_epochs = len(history) if history else 1
        mean_total = float(sum(e["total"] for e in history) / n_epochs)
        mean_recon = float(sum(e["reconstruction"] for e in history) / n_epochs)
        mean_clust = float(sum(e["cluster_weighted"] for e in history) / n_epochs)
        mean_prox  = float(sum(e["prox"] for e in history) / n_epochs)

        flat_ae = _flatten_ae(dcn)
        content = RecordDict({
            "ae_weights": ArrayRecord(torch_state_dict=flat_ae),
            "metrics": MetricRecord({
                "num_examples":  float(len(trainloader.dataset)),
                "loss_total":    mean_total,
                "loss_recon":    mean_recon,
                "loss_clust":    mean_clust,
                "loss_prox":     mean_prox,
            }),
        })
        return Message(content=content, reply_to=msg)

    # ------------------------------------------------------------------
    # Phase 3b: Cluster center update
    # ------------------------------------------------------------------
    elif phase == "cluster_centers":
        cluster_centers = _cluster_centers_from_msg(msg)
        if cluster_centers is not None:
            cluster_centers = cluster_centers.to(device)

        dcn = build_dcn(cfg, cluster_centers=cluster_centers)
        _load_ae_weights(msg, dcn)
        dcn = dcn.to(device)

        synthetic_loader = _synthetic_loader_from_msg(msg, batch_size)

        trainer = SyntheticDCNTrainer(
            dcn=dcn,
            dataloader=trainloader,
            learning_rate=float(cfg.get("learning-rate-federated", cfg["learning-rate"])),
            augmentation_dataloader=synthetic_loader,
        )

        local_centers = trainer.run_kmeans_on_augmented_data()
        centers_np = local_centers.cpu().detach().numpy()

        content = RecordDict({
            "local_centers": ArrayRecord(numpy_ndarrays=[centers_np]),
            "metrics": MetricRecord({"num_examples": float(len(trainloader.dataset))}),
        })
        return Message(content=content, reply_to=msg)

    else:
        raise ValueError(f"Unknown training phase: {phase!r}")


# ---------------------------------------------------------------------------
# Evaluate handler
# ---------------------------------------------------------------------------

@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    try:
        return _evaluate_impl(msg, context)
    except Exception:
        print(f"[CLIENT ERROR node={context.node_id}]\n{traceback.format_exc()}")
        raise


def _evaluate_impl(msg: Message, context: Context) -> Message:
    device = _device()
    cfg = context.run_config

    cluster_centers = _cluster_centers_from_msg(msg)
    if cluster_centers is not None:
        cluster_centers = cluster_centers.to(device)

    dcn = build_dcn(cfg, cluster_centers=cluster_centers)
    _load_ae_weights(msg, dcn)
    dcn = dcn.to(device)
    dcn.eval()

    _, testloader = _load_partition(context)

    all_latent: list[torch.Tensor] = []
    all_observed: list[torch.Tensor] = []
    all_true_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for x, y in testloader:
            x = x.to(device)
            z = dcn.encoder(x)
            all_latent.append(z.cpu())
            all_observed.append(x.cpu())
            all_true_labels.append(y)

    latent = torch.cat(all_latent, dim=0)
    observed = torch.cat(all_observed, dim=0)
    true_labels = torch.cat(all_true_labels, dim=0).numpy()

    centers_cpu = dcn.cluster_model.get_cluster_centers().cpu()
    dcn.cluster_model.set_clusters(centers_cpu)
    pred_assignments = dcn.cluster_model.get_cluster_assignments(latent).numpy()
    n_clusters = int(cfg["n-clusters"])

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

    content = RecordDict({
        "metrics": MetricRecord({
            "acc":             acc,
            "nmi":             nmi,
            "ari":             ari,
            "db_latent":       float(db_latent),
            "db_observed":     float(db_observed),
            "db_gt_observed":  float(db_gt_observed),
            "num_examples":    float(len(testloader.dataset)),
        })
    })
    return Message(content=content, reply_to=msg)
