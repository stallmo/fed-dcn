from typing import Protocol

import torch
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import numpy as np

class ClusteringModel(Protocol):

    def get_number_of_clusters(self) -> int:
        ...
    def update_clusters(self, features: torch.Tensor) -> None:
        ...
    def update_clusters_from_stacked(self, stacked_centers: torch.Tensor) -> None:
        ...
    def get_cluster_assignments(self, features: torch.Tensor) -> torch.Tensor:
        ...
    def get_cluster_centers(self) -> torch.Tensor:
        ...
    def compute_clustering_loss(self, features: torch.Tensor) -> torch.Tensor:
        ...
    def compute_db_index(self, features: torch.Tensor) -> torch.Tensor:
        ...
    def compute_db_index_with_labels(
        self, features: torch.Tensor, labels: np.ndarray
    ) -> float:
        ...
    def set_clusters(self, cluster_centers: torch.Tensor) -> None:
        ...

class KMeansClusteringModel:

    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters
        self.cluster_centers = None

    def _pairwise_sq_distances(self, features: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        if centers.device != features.device:
            centers = centers.to(features.device)
        feat_sq  = (features ** 2).sum(dim=1, keepdim=True)
        cent_sq  = (centers  ** 2).sum(dim=1, keepdim=True).T
        cross    = features @ centers.T
        return feat_sq + cent_sq - 2 * cross

    def get_number_of_clusters(self) -> int:
        return self.n_clusters

    def update_clusters(self, features: torch.Tensor) -> None:
        k_means_init = "k-means++"
        if self.cluster_centers is not None:
            k_means_init = self.cluster_centers.cpu().detach().numpy()
        kmeans = KMeans(n_clusters=self.n_clusters, init=k_means_init, n_init="auto")
        kmeans.fit(features.cpu().detach().numpy())
        self.cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(features.device)

    def get_cluster_assignments(self, features: torch.Tensor) -> torch.Tensor:
        if self.cluster_centers is None:
            raise ValueError("Cluster centers have not been initialized.")
        distances = self._pairwise_sq_distances(features, centers=self.cluster_centers.detach())
        return torch.argmin(distances, dim=1)

    def get_cluster_centers(self) -> torch.Tensor:
        return self.cluster_centers

    def compute_clustering_loss(self, features: torch.Tensor) -> torch.Tensor:
        return self.compute_kmeans_loss(features)

    def compute_kmeans_loss(self, features: torch.Tensor) -> torch.Tensor:
        if self.cluster_centers is None:
            raise ValueError("Cluster centers have not been initialized.")
        centers = self.cluster_centers.detach()
        distances = self._pairwise_sq_distances(features, centers=centers)
        min_distances = torch.min(distances, dim=1)[0]
        return min_distances.mean()

    def compute_db_index(self, features: torch.Tensor) -> torch.Tensor:
        if self.cluster_centers is not None:
            features = features.to(self.cluster_centers.device)
        X = features.detach().cpu().numpy()
        labels = self.get_cluster_assignments(features).cpu().numpy()
        if len(np.unique(labels)) == 1:
            return np.inf
        return torch.tensor(davies_bouldin_score(X, labels))

    def set_clusters(self, cluster_centers: torch.Tensor) -> None:
        self.cluster_centers = cluster_centers

    def update_clusters_from_stacked(self, stacked_centers: torch.Tensor) -> None:
        if self.cluster_centers is not None:
            init = self.cluster_centers.cpu().detach().numpy()
            n_init = 1
        else:
            init = "k-means++"
            n_init = "auto"
        kmeans = KMeans(n_clusters=self.n_clusters, init=init, n_init=n_init)
        kmeans.fit(stacked_centers.cpu().detach().numpy())
        self.cluster_centers = torch.tensor(
            kmeans.cluster_centers_, dtype=torch.float32
        ).to(stacked_centers.device)

    def compute_db_index_with_labels(
        self, features: torch.Tensor, labels: np.ndarray
    ) -> float:
        X = features.detach().cpu().numpy()
        if len(np.unique(labels)) <= 1:
            return float("inf")
        return float(davies_bouldin_score(X, labels))
