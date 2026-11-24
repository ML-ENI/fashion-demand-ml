"""Disk cache for slow notebook steps.

Each result is stored under ``directory/<name>-<hash>.pkl``. The hash covers the key (data run, parameters, and a
step version), so changing any of them computes the step again instead of reusing a stale result.
"""

import hashlib
import json
import pickle
from pathlib import Path


def cached(directory, name, key, compute, refresh=False):
    """Return ``compute()``, reading it from disk when the same ``name`` and ``key`` were computed before."""
    digest = hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]
    path = Path(directory) / f"{name}-{digest}.pkl"
    if path.is_file() and not refresh:
        with path.open("rb") as stream:
            return pickle.load(stream)
    value = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".tmp")
    with partial.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    partial.replace(path)
    return value
