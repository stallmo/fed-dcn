"""Small registry letting fed_dcn_synthetic_app's core dataset-dispatch functions
(task.load_partition/load_test_dataset/load_train_dataset/build_dcn/build_decoder,
server_app._generate_synthetic_dataset) delegate to a dataset-specific implementation without
hardcoding it, while leaving MNIST/Fashion-MNIST/USPS's existing code paths untouched. Sized for a
handful of "embedding-style" datasets (see datasets/__init__.py), not a plugin system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from torch.utils.data import DataLoader

from fed_dcn_synthetic_app.autoencoder import AutoencoderFactory, LinearOutputAutoencoderFactory


@dataclass(frozen=True)
class DatasetSpec:
    """
    Everything task.py/server_app.py need to support one non-image ("embedding-style") dataset.

    :ivar name: The dataset name as it appears in run_config["dataset"].
    :ivar load_partition: Same signature/contract as task.load_partition, plus a trailing
                           ``run_config`` keyword argument.
    :ivar load_test_dataset: Same signature/contract as task.load_test_dataset, plus a trailing
                              ``run_config`` keyword argument.
    :ivar load_train_dataset: Same signature/contract as task.load_train_dataset, plus a trailing
                               ``run_config`` keyword argument.
    :ivar ae_factory_cls: Autoencoder factory used for this dataset's DCN. Most datasets registered
                           here are unbounded, standardized embeddings (not [0,1] pixels), so this
                           defaults to the linear-output factory rather than Stacked's Sigmoid one.
    :ivar clamp_synthetic: Whether server_app._generate_synthetic_dataset should clamp its output
                            to [0,1]. Defaults to True (safe/conservative): a dataset module must
                            *explicitly* opt out of the clamp. This means a registry miss *and* a
                            registered dataset that forgets to set this both clamp -- only an
                            explicit clamp_synthetic=False (as datasets/mimii.py sets) disables it.
                            Preserves exact current behavior everywhere by default.
    """

    name: str
    load_partition: Callable[..., tuple[DataLoader, DataLoader]]
    load_test_dataset: Callable[..., DataLoader]
    load_train_dataset: Callable[..., DataLoader]
    ae_factory_cls: type[AutoencoderFactory] = LinearOutputAutoencoderFactory
    clamp_synthetic: bool = True


DATASET_REGISTRY: dict[str, DatasetSpec] = {}


def register(spec: DatasetSpec) -> None:
    """
    Register a dataset spec. Called as a module-level side effect by each dataset module in this
    package (see datasets/__init__.py).

    :param spec: The dataset's :class:`DatasetSpec`.
    :raises ValueError: If a dataset with this name is already registered.
    """
    if spec.name in DATASET_REGISTRY:
        raise ValueError(f"Dataset {spec.name!r} is already registered.")
    DATASET_REGISTRY[spec.name] = spec
