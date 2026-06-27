"""DCN model and trainer for the synthetic-data-augmented federated learning app."""

from fed_dcn_synthetic_app.clustering import ClusteringModel
from fed_dcn_synthetic_app.autoencoder import AutoencoderFactory

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class DCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        bottleneck_dim: int,
        ae_factory: AutoencoderFactory,
        cluster_model: ClusteringModel,
    ) -> None:
        super().__init__()
        self.encoder = ae_factory.create_encoder(input_dim, hidden_dims, bottleneck_dim)
        self.decoder = ae_factory.create_decoder(input_dim, hidden_dims, bottleneck_dim)
        self.cluster_model = cluster_model
        self.n_clusters = cluster_model.get_number_of_clusters()

    def forward(self, x):
        z = self.encoder(x)
        assignments = (
            self.cluster_model.get_cluster_assignments(z)
            if self.cluster_model.get_cluster_centers() is not None
            else None
        )
        x_recon = self.decoder(z)
        return z, assignments, x_recon


class SyntheticDCNTrainer:
    """DCN trainer that handles all phases of the synthetic-augmented federated pipeline.

    Phases supported:
    - Local DCN pretraining (Phase 1 on clients): reconstruction-only, then k-means init,
      then full DCN loss (reconstruction + cluster).
    - Geometry pretraining (Phase 2a on server): reconstruction + UMAP geometry loss on
      the synthetic dataset.
    - DCN warm-start (Phase 2b on server): DCN loss on the synthetic dataset.
    - Federated AE update (Phase 3 on clients): DCN loss on real batches augmented with
      same-size synthetic batches from the server-provided synthetic dataset.
    - Federated cluster update (Phase 3 on clients): full k-means on the augmented data.
    """

    def __init__(
        self,
        dcn: DCN,
        dataloader: DataLoader,
        learning_rate: float = 0.001,
        alpha_recon: float = 1.0,
        fedprox_mu: float = 0.0,
        augmentation_dataloader: DataLoader | None = None,
        geometric_dataloader: DataLoader | None = None,
    ) -> None:
        self.dcn = dcn
        self.dataloader = dataloader
        self.augmentation_dataloader = augmentation_dataloader
        self.geometric_dataloader = geometric_dataloader
        self.learning_rate = float(learning_rate)
        self.alpha_recon = float(alpha_recon)
        self.fedprox_mu = float(fedprox_mu)

        self.global_ae_params: dict[str, torch.Tensor] | None = None

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.dcn = self.dcn.to(self.device)
        self.optimizer = optim.Adam(self.dcn.parameters(), lr=self.learning_rate)
        print(f"Using device: {self.device}.")

        self._aug_iter = (
            iter(self.augmentation_dataloader)
            if self.augmentation_dataloader is not None
            else None
        )
        self._geom_iter = (
            iter(self.geometric_dataloader)
            if self.geometric_dataloader is not None
            else None
        )

    # ------------------------------------------------------------------
    # Federated interface (matching fed-dcn surface)
    # ------------------------------------------------------------------

    def set_global_ae_params(self, ae_state_dict: dict[str, torch.Tensor]) -> None:
        """Store frozen global AE snapshot for FedProx regularisation."""
        self.global_ae_params = {
            k: v.to(self.device).detach().clone() for k, v in ae_state_dict.items()
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _next_aug_batch(self) -> torch.Tensor | None:
        """Return the next batch from the augmentation dataloader (cycling)."""
        if self._aug_iter is None:
            return None
        try:
            batch = next(self._aug_iter)
        except StopIteration:
            self._aug_iter = iter(self.augmentation_dataloader)
            batch = next(self._aug_iter)
        return batch[0].to(self.device)

    def _compute_geom_loss(self, alpha_geom: float) -> torch.Tensor:
        """MSE between encoder output and UMAP targets over the geometric dataloader."""
        if self.geometric_dataloader is None or alpha_geom <= 0.0:
            return torch.tensor(0.0, device=self.device)
        if self._geom_iter is None:
            self._geom_iter = iter(self.geometric_dataloader)
        try:
            geom_targets, x_geom = next(self._geom_iter)
        except StopIteration:
            self._geom_iter = iter(self.geometric_dataloader)
            geom_targets, x_geom = next(self._geom_iter)

        geom_targets = geom_targets.to(self.device)
        x_geom = x_geom.to(self.device)
        z = self.dcn.encoder(x_geom)
        return nn.MSELoss()(z, geom_targets)

    def _compute_prox_loss(self) -> torch.Tensor:
        """FedProx proximal term: fedprox_mu · ||w - w_global||²."""
        if self.fedprox_mu <= 0.0 or self.global_ae_params is None:
            return torch.tensor(0.0, device=self.device)
        prox = torch.tensor(0.0, device=self.device)
        for name, param in self.dcn.encoder.named_parameters():
            key = f"encoder.{name}"
            if key in self.global_ae_params:
                prox = prox + torch.sum((param - self.global_ae_params[key]) ** 2)
        for name, param in self.dcn.decoder.named_parameters():
            key = f"decoder.{name}"
            if key in self.global_ae_params:
                prox = prox + torch.sum((param - self.global_ae_params[key]) ** 2)
        return self.fedprox_mu * prox

    def _initialize_cluster_centers(self) -> None:
        """K-means initialisation from the full main dataloader."""
        all_z: list[torch.Tensor] = []
        self.dcn.eval()
        with torch.no_grad():
            for batch in self.dataloader:
                z = self.dcn.encoder(batch[0].to(self.device))
                all_z.append(z.cpu())
        self.dcn.train()
        features = torch.cat(all_z, dim=0).to(self.device)
        self.dcn.cluster_model.update_clusters(features)

    def _train_one_epoch(
        self,
        alpha_geom: float,
        beta_clust: float,
        update_ae: bool = True,
    ) -> dict[str, float]:
        """Run one epoch of training with the given loss configuration.

        For each real batch, an equal-size batch from the augmentation dataloader (if set)
        is concatenated so that reconstruction and cluster losses see the augmented data.
        The geometry loss is computed over the geometric_dataloader (if set).
        """
        losses_total: list[float] = []
        losses_recon: list[float] = []
        losses_clust: list[float] = []
        losses_geom: list[float] = []
        losses_prox: list[float] = []

        for batch in self.dataloader:
            x_real = batch[0].to(self.device)

            # Augment batch with same-size synthetic samples
            x_aug = self._next_aug_batch()
            if x_aug is not None:
                n = x_real.shape[0]
                if x_aug.shape[0] >= n:
                    x_aug = x_aug[:n]
                else:
                    reps = (n + x_aug.shape[0] - 1) // x_aug.shape[0]
                    x_aug = x_aug.repeat(reps, 1)[:n]
                x = torch.cat([x_real, x_aug], dim=0)
            else:
                x = x_real

            self.optimizer.zero_grad()

            if update_ae:
                z, _, x_recon = self.dcn.forward(x)

                recon_loss = nn.MSELoss()(x_recon, x)
                losses_recon.append(float(recon_loss.detach().item()))

                if beta_clust > 0.0 and self.dcn.cluster_model.get_cluster_centers() is not None:
                    clust_loss = self.dcn.cluster_model.compute_clustering_loss(z)
                else:
                    clust_loss = torch.tensor(0.0, device=self.device)
                losses_clust.append(float(clust_loss.detach().item()))

                geom_loss = self._compute_geom_loss(alpha_geom)
                losses_geom.append(float(geom_loss.detach().item()))

                prox_loss = self._compute_prox_loss()
                losses_prox.append(float(prox_loss.detach().item()))

                total_loss = (
                    self.alpha_recon * recon_loss
                    + alpha_geom * geom_loss
                    + beta_clust * clust_loss
                    + prox_loss
                )
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.dcn.parameters(), max_norm=1.0)
                self.optimizer.step()
                losses_total.append(float(total_loss.detach().item()))

        def _avg(lst: list[float]) -> float:
            return float(sum(lst) / len(lst)) if lst else 0.0

        avg_clust = _avg(losses_clust)
        avg_geom = _avg(losses_geom)
        return {
            "total":               _avg(losses_total),
            "reconstruction":      _avg(losses_recon),
            "cluster_unweighted":  avg_clust,
            "cluster_weighted":    beta_clust * avg_clust,
            "geometry_unweighted": avg_geom,
            "geometry":            alpha_geom * avg_geom,
            "prox":                _avg(losses_prox),
        }

    # ------------------------------------------------------------------
    # High-level training routines
    # ------------------------------------------------------------------

    def run_ae_pretrain(self, n_epochs: int, alpha_geom: float = 0.0, verbose: bool = True) -> list[dict]:
        """Reconstruction-only pretraining (no cluster loss), then k-means initialisation.

        Used in Phase 1 (local client pretraining) and at the start of Phase 2a.
        After training finishes, cluster centres are initialised via k-means on the
        main dataloader.
        """
        self.optimizer = optim.Adam(self.dcn.parameters(), lr=self.learning_rate)
        history: list[dict] = []
        for epoch in range(n_epochs):
            m = self._train_one_epoch(alpha_geom=alpha_geom, beta_clust=0.0, update_ae=True)
            history.append(m)
            if verbose:
                geom_str = f"  Geom={m['geometry_unweighted']:.4f}" if alpha_geom > 0.0 else ""
                print(
                    f"AE Pretrain {epoch + 1}/{n_epochs}  "
                    f"Total={m['total']:.4f}  Recon={m['reconstruction']:.4f}{geom_str}"
                )
        self._initialize_cluster_centers()
        return history

    def run_geometry_pretrain(
        self, n_epochs: int, alpha_geom: float, verbose: bool = True
    ) -> list[dict]:
        """Phase 2a: reconstruction + UMAP geometry loss; then k-means initialisation.

        Trains on the main dataloader (expected to be the synthetic dataset) with
        geometry targets from the geometric_dataloader.  After training, cluster
        centres are initialised via k-means on the main dataloader.
        """
        self.optimizer = optim.Adam(self.dcn.parameters(), lr=self.learning_rate)
        history: list[dict] = []
        for epoch in range(n_epochs):
            m = self._train_one_epoch(alpha_geom=alpha_geom, beta_clust=0.0, update_ae=True)
            history.append(m)
            if verbose:
                print(
                    f"GeomPretrain {epoch + 1}/{n_epochs}  "
                    f"Total={m['total']:.4f}  "
                    f"Recon={m['reconstruction']:.4f}  "
                    f"Geom={m['geometry_unweighted']:.4f}"
                )
        self._initialize_cluster_centers()
        return history

    def run_warmstart(
        self, n_epochs: int, beta_clust: float, alpha_geom: float = 0.0, verbose: bool = True
    ) -> list[dict]:
        """Phase 2b: DCN loss (reconstruction + beta * cluster) on the synthetic dataset."""
        self.optimizer = optim.Adam(self.dcn.parameters(), lr=self.learning_rate)
        history: list[dict] = []
        for epoch in range(n_epochs):
            m = self._train_one_epoch(alpha_geom=alpha_geom, beta_clust=beta_clust, update_ae=True)
            history.append(m)
            if verbose:
                geom_str = f"  Geom={m['geometry_unweighted']:.4f}" if alpha_geom > 0.0 else ""
                print(
                    f"Warmstart {epoch + 1}/{n_epochs}  "
                    f"Total={m['total']:.4f}  "
                    f"Recon={m['reconstruction']:.4f}  "
                    f"Clust(w)={m['cluster_weighted']:.4f}{geom_str}"
                )
        return history

    def run_federated_ae(
        self,
        n_epochs: int,
        beta_clust: float,
        alpha_geom: float = 0.0,
        early_stopping: int | None = None,
        update_clusters_each_epoch: bool = False,
        verbose: bool = True,
    ) -> list[dict]:
        """DCN loss on local data augmented with synthetic batches.

        When update_clusters_each_epoch=True, runs k-means after each AE update epoch
        (classic DCN alternating optimisation). Used for Phase 1 local training.
        Returns list of per-epoch loss dicts with keys matching fed-dcn's format so that
        server_app aggregation helpers work unchanged.
        """
        self.optimizer = optim.Adam(self.dcn.parameters(), lr=self.learning_rate)
        history: list[dict] = []
        best_loss = np.inf
        no_improve = 0

        for epoch in range(n_epochs):
            m = self._train_one_epoch(alpha_geom=alpha_geom, beta_clust=beta_clust, update_ae=True)
            if update_clusters_each_epoch:
                self.run_kmeans_on_augmented_data()
            history.append(m)

            if m["total"] < best_loss:
                best_loss = m["total"]
                no_improve = 0
            else:
                no_improve += 1

            if verbose:
                geom_str = f"  Geom={m['geometry_unweighted']:.4f}" if alpha_geom > 0.0 else ""
                print(
                    f"FedAE {epoch + 1}/{n_epochs}  "
                    f"Total={m['total']:.4f}  "
                    f"Recon={m['reconstruction']:.4f}  "
                    f"Clust(w)={m['cluster_weighted']:.4f}  "
                    f"Prox={m['prox']:.4f}{geom_str}"
                )

            if early_stopping is not None and no_improve >= early_stopping:
                print(f"Early stopping after {no_improve} epochs without improvement.")
                break

        return history

    def run_kmeans_on_augmented_data(self) -> torch.Tensor:
        """Phase 3 cluster update: full k-means on local + synthetic data in the latent space.

        Returns the new local cluster centres [K, D].
        """
        all_z: list[torch.Tensor] = []
        aug_iter = (
            iter(self.augmentation_dataloader)
            if self.augmentation_dataloader is not None
            else None
        )
        self.dcn.eval()
        with torch.no_grad():
            for batch in self.dataloader:
                x_real = batch[0].to(self.device)
                if aug_iter is not None:
                    try:
                        x_aug = next(aug_iter)[0].to(self.device)
                    except StopIteration:
                        aug_iter = iter(self.augmentation_dataloader)
                        x_aug = next(aug_iter)[0].to(self.device)
                    n = x_real.shape[0]
                    if x_aug.shape[0] >= n:
                        x_aug = x_aug[:n]
                    else:
                        reps = (n + x_aug.shape[0] - 1) // x_aug.shape[0]
                        x_aug = x_aug.repeat(reps, 1)[:n]
                    x = torch.cat([x_real, x_aug], dim=0)
                else:
                    x = x_real
                z = self.dcn.encoder(x)
                all_z.append(z.cpu())
        self.dcn.train()

        all_z_t = torch.cat(all_z, dim=0).to(self.device)
        self.dcn.cluster_model.update_clusters(all_z_t)
        return self.dcn.cluster_model.get_cluster_centers()
