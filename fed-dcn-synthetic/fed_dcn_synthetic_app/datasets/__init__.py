"""Self-registering dataset extensions for fed_dcn_synthetic_app.

Importing this package imports every known dataset module below, each of which calls
``register(...)`` as a module-level side effect. Because ``task.py`` unconditionally
``import``s this package, this happens in *every* process that imports ``fed_dcn_synthetic_app``
at all -- including Ray simulation worker processes, which re-import
``fed_dcn_synthetic_app.client_app``/``server_app`` directly and never go through any external
wrapper module. This is what makes registration reliable without needing anything injected from
outside the package (e.g. a downstream project's own `sitecustomize.py`/monkeypatching).

Adding a new embedding-style dataset later = add one module here + one import line below.
"""

from fed_dcn_synthetic_app.datasets.registry import DATASET_REGISTRY, DatasetSpec, register
from fed_dcn_synthetic_app.datasets import mimii  # noqa: F401  (self-registers "mimii")

__all__ = ["DATASET_REGISTRY", "DatasetSpec", "register"]
