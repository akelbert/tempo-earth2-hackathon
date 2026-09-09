"""Machine-readable model and data catalogs for workshop discovery.

The catalog distinguishes guaranteed event capabilities from candidates and
from large authoritative archives.  Keeping that distinction in data (rather
than prose copied among notebooks) makes it difficult to accidentally promise
an unvalidated model or imply that a tiny teaching extract is the whole archive.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any

CATALOG_FILES = {
    "data": "data_catalog.json",
    "models": "model_catalog.json",
}


def _read_json(uri: str | Path) -> dict[str, Any]:
    value = str(uri)
    if value.startswith(("gs://", "s3://", "http://", "https://")):
        import fsspec

        with fsspec.open(value, "rt") as handle:
            return json.load(handle)
    return json.loads(Path(value).read_text())


def catalog_path(kind: str) -> str:
    """Return the configured or packaged catalog path for ``kind``."""
    if kind not in CATALOG_FILES:
        raise ValueError(f"Unknown catalog {kind!r}; choose from {sorted(CATALOG_FILES)}")
    root = os.environ.get("WORKSHOP_CATALOG_URI")
    if root:
        return f"{root.rstrip('/')}/{CATALOG_FILES[kind]}"
    return str(Path(__file__).with_name(CATALOG_FILES[kind]))


def load_catalog(kind: str) -> dict[str, Any]:
    """Load one catalog, optionally overridden by ``WORKSHOP_CATALOG_URI``."""
    catalog = _read_json(catalog_path(kind))
    expected = "datasets" if kind == "data" else "models"
    if catalog.get("schema_version") != 1 or not isinstance(catalog.get(expected), list):
        raise ValueError(f"Invalid {kind} catalog at {catalog_path(kind)}")
    return catalog


def model_entries(tiers: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """Return model entries, optionally restricted to deployment tiers."""
    entries = deepcopy(load_catalog("models")["models"])
    if tiers is None:
        return entries
    allowed = set(tiers)
    return [entry for entry in entries if entry["tier"] in allowed]


def data_entries(themes: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """Return data entries with their event URI resolved from the environment."""
    entries = deepcopy(load_catalog("data")["datasets"])
    wanted = {theme.lower() for theme in themes or ()}
    resolved = []
    for entry in entries:
        if wanted and not wanted.intersection(theme.lower() for theme in entry["themes"]):
            continue
        root_env = entry.get("workshop_root_env")
        root = os.environ.get(root_env, "") if root_env else ""
        relative = entry.get("workshop_relative_path")
        if root:
            entry["workshop_uri"] = (
                f"{root.rstrip('/')}/{relative.lstrip('/')}" if relative else root
            )
        else:
            entry["workshop_uri"] = None
        resolved.append(entry)
    return resolved


def as_frame(kind: str = "data"):
    """Return a compact pandas table suitable for display in a notebook."""
    import pandas as pd

    if kind == "data":
        rows = data_entries()
        columns = [
            "id",
            "provider",
            "access_mode",
            "workshop_status",
            "workshop_uri",
            "themes",
        ]
    elif kind == "models":
        rows = model_entries()
        columns = [
            "id",
            "kind",
            "tier",
            "live_inference",
            "extra",
            "science_uses",
        ]
    else:
        raise ValueError("kind must be 'data' or 'models'")
    frame = pd.DataFrame(rows)
    for column in ("themes", "science_uses"):
        if column in frame:
            frame[column] = frame[column].map(
                lambda values: ", ".join(values) if isinstance(values, list) else values
            )
    return frame[columns]
