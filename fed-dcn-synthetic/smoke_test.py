"""Quick smoke test for fed-dcn-synthetic modules.

Run with:  uv run python smoke_test.py
"""
import numpy as np
import torch

# ── clustering ──────────────────────────────────────────────────────────────
from fed_dcn_synthetic_app.clustering import KMeansClusteringModel

km = KMeansClusteringModel(n_clusters=3)
stacked = torch.randn(9, 10)
km.update_clusters_from_stacked(stacked)
assert km.get_cluster_centers().shape == (3, 10)
print("clustering.update_clusters_from_stacked: OK")

feats = torch.randn(30, 10)
labels = np.array([i % 3 for i in range(30)])
db = km.compute_db_index_with_labels(feats, labels)
print(f"clustering.compute_db_index_with_labels: {db:.4f}")

# ── task / DCN factory ───────────────────────────────────────────────────────
from fed_dcn_synthetic_app.task import build_dcn, build_decoder, synthetic_dataloader

cfg = {
    "input-dim": 784,
    "hidden-dims": "[256, 128]",
    "bottleneck-dim": 5,
    "n-clusters": 10,
}
dcn = build_dcn(cfg)
assert dcn.n_clusters == 10
print("task.build_dcn: OK")

decoder = build_decoder(cfg)
z = torch.randn(4, 5)
x_hat = decoder(z)
assert x_hat.shape == (4, 784)
print("task.build_decoder forward: OK")

images = torch.rand(50, 784)
loader = synthetic_dataloader(images, batch_size=16)
batch_x, batch_y = next(iter(loader))
assert batch_x.shape[1] == 784
print("task.synthetic_dataloader: OK")

# ── SyntheticDCNTrainer ──────────────────────────────────────────────────────
from fed_dcn_synthetic_app.dcn import SyntheticDCNTrainer
from torch.utils.data import DataLoader, TensorDataset

dummy_data = TensorDataset(torch.rand(64, 784), torch.zeros(64, dtype=torch.long))
dummy_loader = DataLoader(dummy_data, batch_size=16, shuffle=True)

dcn2 = build_dcn(cfg)
trainer = SyntheticDCNTrainer(dcn=dcn2, dataloader=dummy_loader, learning_rate=1e-3)

# AE pretrain (1 epoch) + k-means init
hist = trainer.run_ae_pretrain(n_epochs=1, verbose=True)
assert len(hist) == 1
assert dcn2.cluster_model.get_cluster_centers() is not None
print("SyntheticDCNTrainer.run_ae_pretrain: OK")

# federated AE (1 epoch with cluster loss)
hist2 = trainer.run_federated_ae(n_epochs=1, beta_clust=0.1, verbose=True)
assert len(hist2) == 1
assert "cluster_weighted" in hist2[0]
print("SyntheticDCNTrainer.run_federated_ae: OK")

# k-means on augmented data (no augmentation dataloader — falls back to main)
centers = trainer.run_kmeans_on_augmented_data()
assert centers.shape == (10, 5)
print("SyntheticDCNTrainer.run_kmeans_on_augmented_data: OK")

# warm-start (1 epoch)
hist3 = trainer.run_warmstart(n_epochs=1, beta_clust=0.1, verbose=True)
assert len(hist3) == 1
print("SyntheticDCNTrainer.run_warmstart: OK")

# geometry pretrain (1 epoch) — needs a geometric dataloader
geom_data = TensorDataset(torch.rand(64, 5), torch.rand(64, 784))
geom_loader = DataLoader(geom_data, batch_size=16, shuffle=True)
dcn3 = build_dcn(cfg)
trainer3 = SyntheticDCNTrainer(
    dcn=dcn3, dataloader=dummy_loader, learning_rate=1e-3, geometric_dataloader=geom_loader
)
hist4 = trainer3.run_geometry_pretrain(n_epochs=1, alpha_geom=0.1, verbose=True)
assert dcn3.cluster_model.get_cluster_centers() is not None
print("SyntheticDCNTrainer.run_geometry_pretrain: OK")

# ── FedProx set_global_ae_params ─────────────────────────────────────────────
flat_ae = {}
for k, v in dcn2.encoder.state_dict().items():
    flat_ae[f"encoder.{k}"] = v
for k, v in dcn2.decoder.state_dict().items():
    flat_ae[f"decoder.{k}"] = v
trainer.set_global_ae_params(flat_ae)
assert trainer.global_ae_params is not None
print("SyntheticDCNTrainer.set_global_ae_params: OK")

# ── _compute_cluster_stats ────────────────────────────────────────────────────
from fed_dcn_synthetic_app.client_app import _compute_cluster_stats
# device is derived from the model (may be MPS/CUDA/CPU)
centers_np, std_devs_np, counts_np = _compute_cluster_stats(dcn2, dummy_loader)
assert centers_np.shape == (10, 5)
assert std_devs_np.shape == (10,)
assert counts_np.shape == (10,)
print("client_app._compute_cluster_stats: OK")

# ── module imports ────────────────────────────────────────────────────────────
from fed_dcn_synthetic_app.server_app import app as server_app  # noqa: F401
from fed_dcn_synthetic_app.client_app import app as client_app  # noqa: F401
print("server_app and client_app import: OK")

print("\nAll smoke tests passed.")
