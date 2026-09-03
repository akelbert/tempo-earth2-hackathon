"""Workshop configuration and environment reporting.

Every path and identifier the notebooks depend on is resolved here so that a
single environment variable change can retarget the whole workshop (for example
pointing at a different staged TEMPO copy) without editing notebooks.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Default Earth2Studio prognostic model for the introductory notebook.
#
# FCN (FourCastNet, AFNO) is chosen as the workshop default because:
#   * its only extra dependency is nvidia-physicsnemo (no makani, no JAX, no
#     flash-attn build), which keeps the container reproducible;
#   * its checkpoint is served from Hugging Face (nvidia/fourcastnet1) rather
#     than NGC, so no personal API key has to enter the image or attendee homes;
#   * its 26 output variables include 10 m, 100 m and 850 hPa winds, which is
#     exactly what a boundary-layer NO2 transport story needs.
#
# Swap it by setting WORKSHOP_MODEL; see tempo_earth2.forecast.SUPPORTED_MODELS.
DEFAULT_MODEL = "FCN"

# TEMPO gridded NO2 tropospheric and stratospheric columns.
# V03 stopped on 2025-09-16; V04 is the ongoing collection as of August 2026.
DEFAULT_TEMPO_SHORT_NAME = "TEMPO_NO2_L3"
DEFAULT_TEMPO_VERSION = "V04"


@dataclass
class WorkshopConfig:
    """Resolved locations and identifiers for one workshop environment."""

    tempo_uri: str
    model_cache: Path
    data_cache: Path
    outputs: Path
    reference_notebooks: Path
    work_dir: Path
    model_name: str = DEFAULT_MODEL
    tempo_short_name: str = DEFAULT_TEMPO_SHORT_NAME
    tempo_version: str = DEFAULT_TEMPO_VERSION
    manifest: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> WorkshopConfig:
        home = Path(os.environ.get("HOME", Path.home()))
        work_dir = Path(os.environ.get("WORKSHOP_WORK_DIR", home / "work"))
        outputs = Path(os.environ.get("WORKSHOP_OUTPUTS", work_dir / "outputs"))
        cfg = cls(
            tempo_uri=os.environ.get("TEMPO_DATA_URI", str(work_dir / "data" / "tempo")),
            model_cache=Path(
                os.environ.get(
                    "EARTH2STUDIO_MODEL_CACHE", home / ".cache" / "earth2studio"
                )
            ),
            data_cache=Path(
                os.environ.get(
                    "EARTH2STUDIO_DATA_CACHE", home / ".cache" / "earth2studio"
                )
            ),
            outputs=outputs,
            reference_notebooks=Path(
                os.environ.get("WORKSHOP_REFERENCE_DIR", "/opt/earth2/notebooks")
            ),
            work_dir=work_dir,
            model_name=os.environ.get("WORKSHOP_MODEL", DEFAULT_MODEL),
            tempo_short_name=os.environ.get(
                "TEMPO_SHORT_NAME", DEFAULT_TEMPO_SHORT_NAME
            ),
            tempo_version=os.environ.get("TEMPO_VERSION", DEFAULT_TEMPO_VERSION),
        )
        cfg.manifest = cfg.read_manifest()
        return cfg

    def read_manifest(self) -> dict[str, Any]:
        """Read ``manifest.json`` written next to the staged TEMPO copy.

        The manifest is how ``scripts/stage_tempo.py`` tells the notebooks which
        dates and scans actually exist locally, so attendees never have to guess
        a date that happens to be staged.
        """
        candidates = [
            f"{self.tempo_uri.rstrip('/')}/manifest.json",
            str(Path(self.tempo_uri).parent / "manifest.json"),
        ]
        for candidate in candidates:
            try:
                if candidate.startswith(("gs://", "s3://", "http://", "https://")):
                    import fsspec

                    with fsspec.open(candidate, "rt") as handle:
                        return json.load(handle)
                path = Path(candidate)
                if path.is_file():
                    return json.loads(path.read_text())
            except Exception:  # noqa: BLE001, S112  # pragma: no cover - advisory
                continue
        return {}

    def ensure_dirs(self) -> None:
        for path in (self.outputs, self.data_cache, self.work_dir):
            path.mkdir(parents=True, exist_ok=True)


def describe_environment() -> dict[str, Any]:
    """Collect the facts the design document requires the image to report.

    Returns a plain dict so it can be printed, rendered as a table, or written
    to the benchmark record without further processing.
    """
    info: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "hostname": platform.node(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }

    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_build"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = props.name
            info["gpu_memory_gb"] = round(props.total_memory / 1024**3, 1)
            info["compute_capability"] = f"{props.major}.{props.minor}"
            info["gpu_count"] = torch.cuda.device_count()
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - optional probe
        info["torch"] = f"unavailable: {exc}"

    for package in (
        "earth2studio",
        "physicsnemo",
        "xarray",
        "zarr",
        "numpy",
        "netCDF4",
        "earthaccess",
        "glue",
        "glue_jupyter",
        "matplotlib",
        "cartopy",
    ):
        try:
            module = __import__(package)
            info[package] = getattr(module, "__version__", "unknown")
        except Exception:  # noqa: BLE001  # optional dependency probe
            info[package] = "not installed"

    cfg = WorkshopConfig.from_env()
    info["tempo_data_uri"] = cfg.tempo_uri
    info["model_cache"] = str(cfg.model_cache)
    info["data_cache"] = str(cfg.data_cache)
    info["model_cache_writable"] = os.access(cfg.model_cache, os.W_OK)
    info["home_writable"] = os.access(str(cfg.work_dir.parent), os.W_OK)

    for label, path in (("model_cache", cfg.model_cache), ("home", cfg.work_dir.parent)):
        try:
            usage = shutil.disk_usage(path)
            info[f"{label}_free_gb"] = round(usage.free / 1024**3, 1)
        except Exception:  # noqa: BLE001  # filesystem capability probe
            info[f"{label}_free_gb"] = None

    return info
