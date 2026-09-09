"""Helpers for the TEMPO x Earth-2 hackathon workshop notebooks.

This package exists so the workshop notebooks stay readable. Everything here is
plain xarray/numpy/torch and can be imported and tested without a GPU; only
:mod:`tempo_earth2.forecast` needs Earth2Studio, and it imports it lazily.
"""

__version__ = "0.1.0"

from tempo_earth2.catalog import data_entries, model_entries
from tempo_earth2.config import WorkshopConfig, describe_environment
from tempo_earth2.context import data_root, data_uri, read_aqs_month
from tempo_earth2.tempo import (
    NO2_TROPOSPHERIC,
    REGIONS,
    Region,
    load_tempo_scan,
    load_tempo_series,
    open_tempo_source,
)

__all__ = [
    "NO2_TROPOSPHERIC",
    "REGIONS",
    "Region",
    "WorkshopConfig",
    "data_entries",
    "data_root",
    "data_uri",
    "describe_environment",
    "load_tempo_scan",
    "load_tempo_series",
    "model_entries",
    "open_tempo_source",
    "read_aqs_month",
]
