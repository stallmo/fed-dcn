"""Data loading and model construction utilities."""

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import Compose, ToTensor, Lambda
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner

from fed_dcn_synthetic_app.autoencoder import StackedAutoencoderFactory
from fed_dcn_synthetic_app.clustering import KMeansClusteringModel
from fed_dcn_synthetic_app.dcn import DCN

_fds_cache: dict[tuple, FederatedDataset] = {}


def _get_transforms(dataset: str) -> Compose:
    if dataset in ("mnist", "fashion-mnist"):
        return Compose([
            ToTensor(),
            Lambda(lambda x: torch.flatten(x)),
        ])
    raise ValueError(f"Unsupported dataset: {dataset!r}.")


def _hf_dataset_name(dataset: str) -> str:
    if dataset == "mnist":
        return "ylecun/mnist"
    if dataset == "fashion-mnist":
        return "zalando-datasets/fashion_mnist"
    return dataset


def _image_key(dataset: str) -> str:
    return "image"


def _label_key(dataset: str) -> str:
    return "label"


def _make_collate(image_key: str, label_key: str):
    def collate(batch):
        images = torch.stack([s[image_key] for s in batch])
        labels = torch.tensor([s[label_key] for s in batch])
        return images, labels
    return collate


def _apply_transforms(batch: dict, transforms: Compose, image_key: str) -> dict:
    batch[image_key] = [transforms(img) for img in batch[image_key]]
    return batch


def load_partition(
    partition_id: int,
    num_partitions: int,
    alpha: float,
    dataset: str,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    global _fds_cache
    cache_key = (dataset, alpha, num_partitions, "client", seed)
    if cache_key not in _fds_cache:
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by=_label_key(dataset),
            alpha=alpha,
            seed=seed,
        )
        _fds_cache[cache_key] = FederatedDataset(
            dataset=_hf_dataset_name(dataset),
            partitioners={"train": partitioner},
        )
    fds = _fds_cache[cache_key]
    transforms = _get_transforms(dataset)
    img_key = _image_key(dataset)
    lbl_key = _label_key(dataset)

    partition = fds.load_partition(partition_id)
    partition = partition.with_transform(
        lambda batch: _apply_transforms(batch, transforms, img_key)
    )
    collate = _make_collate(img_key, lbl_key)
    trainloader = DataLoader(partition, batch_size=batch_size, shuffle=True, collate_fn=collate)
    testloader = DataLoader(partition, batch_size=batch_size, shuffle=False, collate_fn=collate)
    return trainloader, testloader


def load_test_dataset(dataset: str, batch_size: int) -> DataLoader:
    from datasets import load_dataset as hf_load
    transforms = _get_transforms(dataset)
    img_key = _image_key(dataset)
    lbl_key = _label_key(dataset)
    test_split = hf_load(_hf_dataset_name(dataset), split="test")
    test_split = test_split.with_transform(
        lambda batch: _apply_transforms(batch, transforms, img_key)
    )
    collate = _make_collate(img_key, lbl_key)
    return DataLoader(test_split, batch_size=batch_size, shuffle=False, collate_fn=collate)


def load_train_dataset(dataset: str, batch_size: int, fraction: float = 1.0, seed: int = 42) -> DataLoader:
    from datasets import load_dataset as hf_load
    transforms = _get_transforms(dataset)
    img_key = _image_key(dataset)
    lbl_key = _label_key(dataset)
    train_split = hf_load(_hf_dataset_name(dataset), split="train")
    if fraction < 1.0:
        n = max(1, int(len(train_split) * fraction))
        train_split = train_split.shuffle(seed=seed).select(range(n))
    train_split = train_split.with_transform(
        lambda batch: _apply_transforms(batch, transforms, img_key)
    )
    collate = _make_collate(img_key, lbl_key)
    return DataLoader(train_split, batch_size=batch_size, shuffle=False, collate_fn=collate)


def synthetic_dataloader(synthetic_images: torch.Tensor, batch_size: int, shuffle: bool = True) -> DataLoader:
    """Wrap a flat synthetic image tensor [N, input_dim] into a DataLoader."""
    labels = torch.zeros(synthetic_images.shape[0], dtype=torch.long)
    dataset = TensorDataset(synthetic_images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def build_dcn(run_config: dict, cluster_centers: Optional[torch.Tensor] = None) -> DCN:
    input_dim: int = int(run_config["input-dim"])
    hidden_dims: list = json.loads(str(run_config["hidden-dims"]))
    bottleneck_dim: int = int(run_config["bottleneck-dim"])
    n_clusters: int = int(run_config["n-clusters"])

    cluster_model = KMeansClusteringModel(n_clusters=n_clusters)
    if cluster_centers is not None:
        cluster_model.set_clusters(cluster_centers)

    return DCN(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        bottleneck_dim=bottleneck_dim,
        ae_factory=StackedAutoencoderFactory(),
        cluster_model=cluster_model,
    )


def build_decoder(run_config: dict) -> torch.nn.Module:
    """Build a standalone decoder from run config (used by server to decode synthetic samples)."""
    hidden_dims: list = json.loads(str(run_config["hidden-dims"]))
    input_dim: int = int(run_config["input-dim"])
    bottleneck_dim: int = int(run_config["bottleneck-dim"])
    return StackedAutoencoderFactory.create_decoder(input_dim, hidden_dims, bottleneck_dim)
