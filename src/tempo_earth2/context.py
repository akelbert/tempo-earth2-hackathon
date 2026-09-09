"""Small, uniform loaders for the workshop's contextual teaching data.

The event bucket contains curated extracts, not source archives. Tabular data
are CSV so attendees can inspect them without special software. Large gridded
data use consolidated Zarr; tiny teaching fixtures use NetCDF.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def data_root(root: str | Path | None = None) -> str:
    """Return the configured context-data root."""
    if root is not None:
        return str(root).rstrip("/")
    configured = os.environ.get("WORKSHOP_CONTEXT_DATA_URI")
    if configured:
        return configured.rstrip("/")
    work = Path(os.environ.get("WORKSHOP_WORK_DIR", Path.home() / "work"))
    return str(work / "data" / "context")


def data_uri(relative: str | Path, root: str | Path | None = None) -> str:
    """Join a relative dataset name to a local or object-store root."""
    base = data_root(root)
    name = str(relative).lstrip("/")
    if base.startswith(("gs://", "s3://", "http://", "https://")):
        return f"{base}/{name}"
    return str(Path(base) / name)


def read_csv(
    relative: str | Path,
    root: str | Path | None = None,
    time_columns: tuple[str, ...] = ("time_utc", "date"),
    **kwargs,
) -> pd.DataFrame:
    """Read a context CSV and parse its conventional time columns.

    Extra keyword arguments are passed to :func:`pandas.read_csv`, so callers
    can use ``usecols``, ``nrows``, or chunking for unfamiliar collections.
    """
    frame = pd.read_csv(data_uri(relative, root), **kwargs)
    for column in time_columns:
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame


def read_aqs_month(
    month: str,
    root: str | Path | None = None,
    parameters: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Read one staged ``YYYY-MM`` AQS partition, optionally by pollutant."""
    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", month):
        raise ValueError("month must use YYYY-MM")
    frame = read_csv(f"aqs/by-month/{month}.csv.gz", root=root)
    if parameters:
        frame = frame.loc[frame["parameter"].isin(parameters)].copy()
    return frame


def open_zarr(relative: str | Path, root: str | Path | None = None) -> xr.Dataset:
    """Open a consolidated context Zarr store."""
    return xr.open_zarr(data_uri(relative, root), consolidated=True)


def open_netcdf(relative: str | Path, root: str | Path | None = None) -> xr.Dataset:
    """Open a small NetCDF fixture locally or load it from object storage."""
    uri = data_uri(relative, root)
    if uri.startswith(("gs://", "s3://", "http://", "https://")):
        import fsspec

        with fsspec.open(uri, "rb") as handle:
            return xr.load_dataset(handle, engine="h5netcdf")
    return xr.open_dataset(uri, engine="h5netcdf")


def nearest_grid_values(
    grid: xr.Dataset,
    observations: pd.DataFrame,
    variables: tuple[str, ...],
    time_column: str = "time_utc",
) -> pd.DataFrame:
    """Attach nearest-time, nearest-pixel grid values to point observations."""
    rows: list[dict[str, float]] = []
    for row in observations.itertuples(index=False):
        selected = grid.sel(
            lat=float(row.latitude),
            lon=float(row.longitude),
            method="nearest",
        )
        if "time" in selected.dims or "time" in selected.coords:
            selected = selected.sel(
                time=np.datetime64(getattr(row, time_column).to_datetime64()),
                method="nearest",
            )
        rows.append({name: float(selected[name].values) for name in variables})
    values = pd.DataFrame(rows, index=observations.index)
    return pd.concat([observations.copy(), values], axis=1)


def linear_baseline(
    frame: pd.DataFrame,
    features: list[str],
    target: str,
    train: np.ndarray | pd.Series | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Fit ordinary least squares and return predictions plus simple metrics."""
    columns = features + [target]
    valid = np.isfinite(frame[columns].to_numpy(dtype=float)).all(axis=1)
    train_mask = valid if train is None else valid & np.asarray(train, dtype=bool)
    if train_mask.sum() <= len(features):
        raise ValueError("Not enough finite training rows for this baseline")

    x = frame[features].to_numpy(dtype=float)
    y = frame[target].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(frame)), x])
    coefficients, *_ = np.linalg.lstsq(design[train_mask], y[train_mask], rcond=None)
    predictions = design @ coefficients

    scored = valid & np.isfinite(predictions)
    residual = y[scored] - predictions[scored]
    rmse = float(np.sqrt(np.mean(residual**2)))
    denominator = float(np.sum((y[scored] - np.mean(y[scored])) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / denominator if denominator else np.nan
    metrics = {"rmse": rmse, "r2": r2, "rows": float(scored.sum())}
    return predictions, metrics
