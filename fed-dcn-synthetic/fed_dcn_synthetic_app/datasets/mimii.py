"""Native MIMII sound-embeddings dataset support for fed_dcn_synthetic_app.

One client per machine_type; train/test split and per-machine-type pooling come from
``audio_processing.data.make_train_test_split``, imported lazily inside :func:`build_mimii_fed_data`
(mirroring ``task.py``'s own existing lazy import of ``datasets.load_dataset``) so importing this
module -- done unconditionally, at process startup, by ``fed_dcn_synthetic_app.datasets`` -- never
requires ``audio-processing`` to already be importable unless a MIMII data function actually runs.

All data locations are supplied via ``run_config`` (Flower's ``[tool.flwr.app.config]``) -- this
module has no built-in notion of "the repo root", since it lives in a different repo than the data
it's pointed at.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from fed_dcn_synthetic_app.autoencoder import LinearOutputAutoencoderFactory
from fed_dcn_synthetic_app.datasets.registry import DatasetSpec, register

MACHINE_TYPES: list[str] = ["fan", "pump", "slider", "valve"]
MACHINE_IDS: list[str] = ["id_00", "id_02", "id_04", "id_06"]

_DEFAULT_CONTENT_TYPE = "env"
_DEFAULT_EMBEDDING_SIZE = 512
_DEFAULT_SPLIT_SEED = 42


def load_mimii_embeddings(
    embeddings_dir: Path,
    content_type: str,
    embedding_size: int,
    machine_types: list[str],
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Load precomputed OpenL3 embeddings for the MIMII dataset.

    Each wav file's per-frame embedding sequence ``(T, D)`` is mean-pooled across time into a
    single ``D``-dim vector, giving one row per file.

    :param embeddings_dir: Root of the embeddings tree (``<embeddings_dir>/content_type=<ct>/
                            embedding_size=<es>/<machine_type>/<machine_id>/<label>/<stem>.npz``).
    :param content_type: OpenL3 content type used when the embeddings were computed ("env" or "music").
    :param embedding_size: OpenL3 embedding size used when the embeddings were computed (512 or 6144).
    :param machine_types: Which top-level machine_type directories to load (e.g. ``["fan", "pump"]``).
    :return: ``(embeddings, metadata)`` -- a ``(N, D)`` float32 array and a DataFrame with columns
             ``machine_type``, ``machine_id``, ``anomaly_label``, ``file`` (one row per embedding,
             same order as ``embeddings``).
    """
    combo_root = embeddings_dir / f"content_type={content_type}" / f"embedding_size={embedding_size}"
    if not combo_root.exists():
        raise FileNotFoundError(
            f"{combo_root} does not exist -- check the 'mimii-embeddings-dir'/'mimii-content-type'/"
            f"'mimii-embedding-size' run_config values, and that the embeddings have been computed."
        )

    embeddings: list[np.ndarray] = []
    rows: list[dict] = []

    for machine_type in machine_types:
        machine_dir = combo_root / machine_type
        if not machine_dir.exists():
            print(f"  Warning: {machine_dir} not found, skipping (not computed yet?)")
            continue

        npz_paths = sorted(machine_dir.rglob("*.npz"))
        for npz_path in npz_paths:
            machine_id = npz_path.parent.parent.name
            anomaly_label = npz_path.parent.name
            data = np.load(npz_path)
            embeddings.append(data["embedding"].mean(axis=0))
            rows.append(
                {
                    "machine_type": machine_type,
                    "machine_id": machine_id,
                    "anomaly_label": anomaly_label,
                    "file": npz_path.stem,
                }
            )

    if not embeddings:
        raise ValueError(f"No .npz files found under {combo_root} for machine_types={machine_types}")

    embeddings_arr = np.stack(embeddings).astype(np.float32)
    metadata = pd.DataFrame(rows)
    return embeddings_arr, metadata


@dataclass
class MimiiFedData:
    """
    Result of :func:`build_mimii_fed_data`.

    :ivar embeddings: All loaded embeddings, shape ``(N, D)``.
    :ivar metadata: Row-aligned metadata (``machine_type``/``machine_id``/``anomaly_label``/``file``).
    :ivar scaler: A ``StandardScaler`` fit once on the pooled normal-train rows across all clients.
    :ivar client_train_idx: ``machine_type`` -> row indices into ``embeddings``/``metadata`` that make
                             up that client's local (normal-only) training partition.
    :ivar test_idx: Row indices into ``embeddings``/``metadata`` making up the centralized,
                     ~50/50 normal/abnormal test set, pooled across all machine types/ids.
    """

    embeddings: np.ndarray
    metadata: pd.DataFrame
    scaler: StandardScaler
    client_train_idx: dict[str, np.ndarray]
    test_idx: np.ndarray


def build_mimii_fed_data(
    embeddings_dir: Path,
    mimii_data_dir: Path,
    content_type: str = _DEFAULT_CONTENT_TYPE,
    embedding_size: int = _DEFAULT_EMBEDDING_SIZE,
    machine_types: list[str] = MACHINE_TYPES,
    machine_ids: list[str] = MACHINE_IDS,
    split_seed: int = _DEFAULT_SPLIT_SEED,
) -> MimiiFedData:
    """
    Build federated MIMII train/test partitions from precomputed embeddings.

    For each ``(machine_type, machine_id)``, ``make_train_test_split`` is called once (seeded by
    ``split_seed``, independent of any model-training seed, so every process -- each Flower client
    and the server -- derives the identical split from disk). Train files are grouped per
    ``machine_type`` into ``client_train_idx``; all test files (normal + abnormal) are pooled into
    ``test_idx``. A single ``StandardScaler`` is then fit on the union of all clients' train rows.

    :param embeddings_dir: Root of the precomputed embeddings tree (see :func:`load_mimii_embeddings`).
    :param mimii_data_dir: Root of the raw MIMII wav tree, used to resolve each machine's directory
                            for ``make_train_test_split``'s own file search.
    :param content_type: OpenL3 content type used when the embeddings were computed.
    :param embedding_size: OpenL3 embedding size used when the embeddings were computed.
    :param machine_types: Machine types to include -- also defines the client set (one client per type).
    :param machine_ids: Machine ids to pool within each machine type.
    :param split_seed: Seed for ``make_train_test_split``'s balanced-sampling, fixed across runs so
                        every process reproduces the identical partitions.
    :return: A populated :class:`MimiiFedData`.
    """
    from audio_processing.data import make_train_test_split  # lazy: see module docstring

    embeddings, metadata = load_mimii_embeddings(embeddings_dir, content_type, embedding_size, machine_types)

    row_keys = list(
        zip(metadata["machine_type"], metadata["machine_id"], metadata["anomaly_label"], metadata["file"])
    )

    train_keys_by_type: dict[str, set[tuple[str, str, str, str]]] = {mt: set() for mt in machine_types}
    test_keys: set[tuple[str, str, str, str]] = set()

    for machine_type in machine_types:
        for machine_id in machine_ids:
            machine_dir = mimii_data_dir / machine_type / machine_id
            if not machine_dir.exists():
                print(f"  Warning: {machine_dir} not found, skipping")
                continue
            split = make_train_test_split(machine_dir, seed=split_seed)
            train_keys_by_type[machine_type].update(
                (machine_type, machine_id, "normal", p.stem) for p in split.train_paths
            )
            test_keys.update((machine_type, machine_id, "normal", p.stem) for p in split.test_normal_paths)
            test_keys.update(
                (machine_type, machine_id, "abnormal", p.stem) for p in split.test_abnormal_paths
            )

    client_train_idx: dict[str, np.ndarray] = {}
    for machine_type in machine_types:
        mask = np.array([key in train_keys_by_type[machine_type] for key in row_keys])
        client_train_idx[machine_type] = np.flatnonzero(mask)

    test_mask = np.array([key in test_keys for key in row_keys])
    test_idx = np.flatnonzero(test_mask)

    all_train_idx = (
        np.concatenate([client_train_idx[mt] for mt in machine_types])
        if machine_types
        else np.array([], dtype=np.int64)
    )
    if len(all_train_idx) == 0:
        raise ValueError(
            "No training rows found -- check embeddings_dir/mimii_data_dir and machine_types/machine_ids."
        )
    scaler = StandardScaler().fit(embeddings[all_train_idx])

    return MimiiFedData(
        embeddings=embeddings,
        metadata=metadata,
        scaler=scaler,
        client_train_idx=client_train_idx,
        test_idx=test_idx,
    )


def _mimii_run_config(run_config: dict | None) -> dict:
    """
    Resolve MIMII-specific ``build_mimii_fed_data`` kwargs from a Flower ``run_config``.

    :param run_config: The run's ``[tool.flwr.app.config]`` dict (or ``None``).
    :raises KeyError: If ``mimii-embeddings-dir``/``mimii-data-dir`` are missing -- this dataset has
                       no built-in default data location.
    """
    run_config = run_config or {}
    try:
        embeddings_dir = Path(run_config["mimii-embeddings-dir"])
        mimii_data_dir = Path(run_config["mimii-data-dir"])
    except KeyError as exc:
        raise KeyError(
            "dataset='mimii' requires 'mimii-embeddings-dir' and 'mimii-data-dir' to be set in "
            "[tool.flwr.app.config] (run_config) -- fed_dcn_synthetic_app.datasets.mimii has no "
            "built-in default data location."
        ) from exc
    return dict(
        embeddings_dir=embeddings_dir,
        mimii_data_dir=mimii_data_dir,
        content_type=str(run_config.get("mimii-content-type", _DEFAULT_CONTENT_TYPE)),
        embedding_size=int(run_config.get("mimii-embedding-size", _DEFAULT_EMBEDDING_SIZE)),
        split_seed=int(run_config.get("mimii-split-seed", _DEFAULT_SPLIT_SEED)),
        machine_types=(
            json.loads(run_config["mimii-machine-types"])
            if run_config.get("mimii-machine-types")
            else MACHINE_TYPES
        ),
        machine_ids=(
            json.loads(run_config["mimii-machine-ids"]) if run_config.get("mimii-machine-ids") else MACHINE_IDS
        ),
    )


_cache: dict[tuple, MimiiFedData] = {}


def _get_mimii_data(run_config: dict | None) -> MimiiFedData:
    cfg = _mimii_run_config(run_config)
    key = (
        str(cfg["embeddings_dir"]),
        str(cfg["mimii_data_dir"]),
        cfg["content_type"],
        cfg["embedding_size"],
        tuple(cfg["machine_types"]),
        tuple(cfg["machine_ids"]),
        cfg["split_seed"],
    )
    if key not in _cache:
        _cache[key] = build_mimii_fed_data(**cfg)
    return _cache[key]


def _binary_labels(data: MimiiFedData, idx: np.ndarray) -> np.ndarray:
    # .copy() ensures a writable, contiguous array -- torch.from_numpy on a pandas-backed view can
    # otherwise produce a tensor over read-only memory (undefined behavior on write).
    return (data.metadata.iloc[idx]["anomaly_label"] == "abnormal").astype(np.int64).to_numpy().copy()


def load_partition(
    partition_id: int,
    num_partitions: int,
    alpha: float,
    dataset: str,
    batch_size: int,
    seed: int,
    run_config: dict | None = None,
) -> tuple[DataLoader, DataLoader]:
    """
    Build one client's (train, test) DataLoaders -- one client per MIMII machine type.

    :param partition_id: Index into the effective machine-type list for this client.
    :param num_partitions: Must equal the effective machine-type count -- partitioning here is
                            fixed by machine type, not by count.
    :param alpha: Unused (Dirichlet-partitioning parameter, not applicable to this fixed partitioning).
    :param dataset: Must be ``"mimii"``.
    :param batch_size: DataLoader batch size.
    :param seed: Unused -- the train/test split is fixed by ``mimii-split-seed``, independent of the
                 run's model-training seed.
    :param run_config: The run's ``[tool.flwr.app.config]`` dict; must contain
                        ``mimii-embeddings-dir``/``mimii-data-dir`` (see :func:`_mimii_run_config`).
    :return: ``(trainloader, testloader)``. ``testloader`` is a copy of the trainloader wrapped only
             for interface compatibility -- this pipeline never invokes a client's evaluate handler.
    """
    assert dataset == "mimii", f"datasets.mimii.load_partition only supports dataset='mimii', got {dataset!r}"
    machine_types = _mimii_run_config(run_config)["machine_types"]
    assert num_partitions == len(machine_types), (
        f"num_partitions must equal len(machine_types)={len(machine_types)} (one client per machine "
        f"type), got {num_partitions}"
    )

    data = _get_mimii_data(run_config)
    machine_type = machine_types[partition_id]
    idx = data.client_train_idx[machine_type]

    X = data.scaler.transform(data.embeddings[idx]).astype(np.float32)
    y = _binary_labels(data, idx)  # constant 0 by construction (train is normal-only); unused by training
    dataset_t = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))

    # drop_last avoids a stray undersized final batch crashing BatchNorm1d (which requires >1
    # sample per batch in train mode).
    trainloader = DataLoader(dataset_t, batch_size=batch_size, shuffle=True, drop_last=True)
    testloader = DataLoader(dataset_t, batch_size=batch_size, shuffle=False)
    return trainloader, testloader


def load_test_dataset(dataset: str, batch_size: int, run_config: dict | None = None) -> DataLoader:
    """
    Build the centralized, pooled ~50/50 normal/abnormal MIMII test set.

    :param dataset: Must be ``"mimii"``.
    :param batch_size: DataLoader batch size.
    :param run_config: The run's ``[tool.flwr.app.config]`` dict; see :func:`_mimii_run_config`.
    :return: A DataLoader yielding ``(x, binary_anomaly_label)`` batches.
    """
    assert (
        dataset == "mimii"
    ), f"datasets.mimii.load_test_dataset only supports dataset='mimii', got {dataset!r}"
    data = _get_mimii_data(run_config)
    X = data.scaler.transform(data.embeddings[data.test_idx]).astype(np.float32)
    y = _binary_labels(data, data.test_idx)
    return DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y)), batch_size=batch_size, shuffle=False
    )


def load_train_dataset(
    dataset: str,
    batch_size: int,
    fraction: float = 1.0,
    seed: int = 42,
    run_config: dict | None = None,
) -> DataLoader:
    """
    Build the pooled (across all clients) normal-only train set, for the server's centralized
    train-side diagnostic evaluation (not used for training itself).

    :param dataset: Must be ``"mimii"``.
    :param batch_size: DataLoader batch size.
    :param fraction: If less than 1.0, randomly subsample this fraction of rows (speeds up
                      per-round evaluation).
    :param seed: Seed for the subsampling RNG.
    :param run_config: The run's ``[tool.flwr.app.config]`` dict; see :func:`_mimii_run_config`.
    :return: A DataLoader yielding ``(x, binary_anomaly_label)`` batches. The label is constant 0
             (train is normal-only by construction) -- this makes train-side Hungarian-matching
             accuracy degenerate/uninformative, hence experiments on this dataset typically set
             ``early-stopping-metric = "db_latent"`` rather than ``"accuracy"``.
    """
    assert (
        dataset == "mimii"
    ), f"datasets.mimii.load_train_dataset only supports dataset='mimii', got {dataset!r}"
    data = _get_mimii_data(run_config)
    machine_types = _mimii_run_config(run_config)["machine_types"]
    idx = np.concatenate([data.client_train_idx[mt] for mt in machine_types])
    if fraction < 1.0:
        n = max(1, int(len(idx) * fraction))
        idx = idx[np.random.default_rng(seed).choice(len(idx), size=n, replace=False)]
    X = data.scaler.transform(data.embeddings[idx]).astype(np.float32)
    y = _binary_labels(data, idx)
    return DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y)), batch_size=batch_size, shuffle=False
    )


register(
    DatasetSpec(
        name="mimii",
        load_partition=load_partition,
        load_test_dataset=load_test_dataset,
        load_train_dataset=load_train_dataset,
        ae_factory_cls=LinearOutputAutoencoderFactory,
        clamp_synthetic=False,
    )
)
