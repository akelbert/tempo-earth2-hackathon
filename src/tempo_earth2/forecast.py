"""Thin Earth2Studio wrapper for the workshop notebooks.

Earth2Studio is imported lazily so that ``import tempo_earth2`` still works in a
partially installed environment - the environment-check notebook
needs to report *why* a piece is missing rather than fail at import.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import xarray as xr

#: Earth2Studio prognostic models the workshop knows about, mapped to the pip
#: extra that provides them.
#:
#: Only the extras listed in the Dockerfile's ``EARTH2STUDIO_EXTRAS`` are
#: actually installed - by default just ``fcn``. The others are here so that
#: swapping the workshop model is a one-line change in two files rather than an
#: archaeology exercise. Selecting an uninstalled model raises Earth2Studio's
#: own optional-dependency error, which names the missing extra; the image
#: installs nothing at runtime, so the fix is a rebuild, not a pip install.
SUPPORTED_MODELS: dict[str, str] = {
    "FCN": "fcn",
    "SFNO": "sfno",
    "DLWP": "dlwp",
    "FCN3": "fcn3",
    "GraphCastOperational": "graphcast",
    "Aurora": "aurora",
}

#: Near-surface and boundary-layer fields relevant to NO2 transport. Requesting
#: a subset keeps the forecast Zarr around 10x smaller than the full model state.
TRANSPORT_VARIABLES = ("u10m", "v10m", "u100m", "v100m", "t2m", "msl", "u850", "v850")

#: Earth2Studio prognostic models advance the state on a fixed timestep.
MODEL_TIMESTEP_HOURS: dict[str, int] = {
    "FCN": 6,
    "SFNO": 6,
    "DLWP": 6,
    "FCN3": 6,
    "GraphCastOperational": 6,
    "Aurora": 6,
}


@dataclass
class ForecastRun:
    """A completed forecast plus the metadata the benchmark record needs."""

    dataset: xr.Dataset
    model_name: str
    init_time: datetime
    nsteps: int
    device: str
    wall_seconds: float
    peak_gpu_gb: float | None
    store_path: str | None

    def summary(self) -> dict[str, object]:
        return {
            "model": self.model_name,
            "init_time": self.init_time.isoformat(),
            "nsteps": self.nsteps,
            "device": self.device,
            "wall_seconds": round(self.wall_seconds, 1),
            "peak_gpu_gb": self.peak_gpu_gb,
            "variables": list(self.dataset.data_vars),
            "store": self.store_path,
        }


def cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001  # optional CUDA runtime probe
        return False


def nearest_init_time(target: datetime, timestep_hours: int = 6) -> datetime:
    """Round *down* to the most recent model initialization time.

    Rounding down rather than to-nearest matters: an initialization after the
    TEMPO scan would let the model see the future, which quietly turns the
    forecast-versus-observation comparison into a reanalysis comparison.
    """
    if target.tzinfo is None:
        target = target.replace(tzinfo=UTC)
    target = target.astimezone(UTC)
    hour = (target.hour // timestep_hours) * timestep_hours
    return target.replace(hour=hour, minute=0, second=0, microsecond=0)


def steps_to_cover(init_time: datetime, valid_time: datetime, timestep_hours: int = 6) -> int:
    """Number of forecast steps needed to reach or pass ``valid_time``."""
    if valid_time.tzinfo is None:
        valid_time = valid_time.replace(tzinfo=UTC)
    delta = valid_time.astimezone(UTC) - init_time
    return max(1, int(np.ceil(delta.total_seconds() / (timestep_hours * 3600))))


def load_model(model_name: str = "FCN"):
    """Load an Earth2Studio prognostic model from its default package."""
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"{model_name!r} is not in the workshop image. "
            f"Available: {', '.join(sorted(SUPPORTED_MODELS))}"
        )
    try:
        from earth2studio.models import px
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "earth2studio is not importable. Inside the workshop container this "
            "means the image is wrong, not that you need to pip install anything."
        ) from exc

    try:
        cls = getattr(px, model_name)
    except AttributeError as exc:
        raise ImportError(f"earth2studio.models.px has no {model_name}") from exc

    package = cls.load_default_package()
    return cls.load_model(package)


def run_forecast(
    init_time: datetime | str,
    nsteps: int = 4,
    model_name: str = "FCN",
    variables: Sequence[str] = TRANSPORT_VARIABLES,
    store_path: str | Path | None = None,
    model=None,
    verbose: bool = True,
) -> ForecastRun:
    """Run a deterministic Earth2Studio forecast from GFS initial conditions.

    Parameters
    ----------
    init_time
        Model initialization time. Must be on the model timestep (00/06/12/18Z
        for every model in :data:`SUPPORTED_MODELS`); use
        :func:`nearest_init_time` to snap an arbitrary time.
    nsteps
        Number of forecast steps. FCN steps are 6 hours.
    variables
        Output variables to keep. Restricting this is the single most effective
        way to keep the forecast store small.
    store_path
        Where to write the forecast Zarr. ``None`` keeps it in memory, which is
        fine for short runs and avoids filling an attendee home directory.
    model
        A pre-loaded model, to avoid paying checkpoint load time twice.
    """
    import time as _time

    import torch
    from earth2studio import run
    from earth2studio.data import GFS
    from earth2studio.io import ZarrBackend

    if isinstance(init_time, str):
        init_time = datetime.fromisoformat(init_time)
    if init_time.tzinfo is not None:
        init_time = init_time.astimezone(UTC).replace(tzinfo=None)

    step_hours = MODEL_TIMESTEP_HOURS.get(model_name, 6)
    if init_time.hour % step_hours or init_time.minute or init_time.second:
        raise ValueError(
            f"{init_time} is not a valid {model_name} initialization time "
            f"(must be on a {step_hours}-hour boundary). "
            "Use tempo_earth2.forecast.nearest_init_time()."
        )

    if model is None:
        model = load_model(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # overwrite=True: this workflow always runs a fresh forecast, never an
    # incremental resume. Without it, ZarrBackend reopens whatever store
    # already exists at store_path (keyed only by model + init_time, not
    # nsteps) and keeps its stale coordinate arrays - e.g. a lead_time axis
    # sized for an earlier, shorter nsteps - which corrupts later writes at
    # lead times the old store never had room for.
    io = ZarrBackend(
        file_name=str(store_path) if store_path else None,
        backend_kwargs={"overwrite": True},
    )

    available = set(model.output_coords(model.input_coords())["variable"].tolist())
    requested = [v for v in variables if v in available]
    missing = [v for v in variables if v not in available]
    if not requested:
        raise ValueError(
            f"None of {list(variables)} are produced by {model_name}. "
            f"It produces: {sorted(available)}"
        )
    if missing and verbose:
        print(f"[forecast] {model_name} does not produce {missing}; skipping those.")

    started = _time.perf_counter()
    io = run.deterministic(
        [init_time],
        nsteps,
        model,
        GFS(),
        io,
        output_coords={"variable": np.array(requested)},
        device=device,
        verbose=verbose,
    )
    wall = _time.perf_counter() - started

    peak_gb = None
    if device.type == "cuda":
        peak_gb = round(torch.cuda.max_memory_allocated() / 1024**3, 2)

    return ForecastRun(
        dataset=to_xarray(io, requested),
        model_name=model_name,
        init_time=init_time,
        nsteps=nsteps,
        device=str(device),
        wall_seconds=wall,
        peak_gpu_gb=peak_gb,
        store_path=str(store_path) if store_path else None,
    )


def _as_timedelta(values: np.ndarray) -> np.ndarray:
    """Coerce a lead-time coordinate back to ``timedelta64``.

    Earth2Studio stores ``lead_time`` as a native ``timedelta64`` Zarr array, so
    this is normally a no-op. It exists because a Zarr version that round-trips
    the values as plain integers would otherwise produce a forecast whose valid
    times are silently wrong by a factor of 3.6e12 - the kind of failure that
    only shows up as a bad skill score.
    """
    if np.issubdtype(values.dtype, np.timedelta64):
        return values
    if np.issubdtype(values.dtype, np.integer) and values.size > 1:
        step = int(np.min(np.diff(values.astype("int64"))))
        # Every model in SUPPORTED_MODELS steps in whole hours, so a step of a
        # few units is hours and a step of billions is nanoseconds.
        unit = "h" if abs(step) < 1000 else "ns"
        return values.astype(f"timedelta64[{unit}]")
    raise TypeError(
        f"lead_time came back as {values.dtype}, which cannot be interpreted as "
        "a duration. The Earth2Studio IO backend has changed; fix to_xarray()."
    )


def _as_datetime(values: np.ndarray) -> np.ndarray:
    """Coerce an initialization-time coordinate back to ``datetime64[ns]``."""
    if np.issubdtype(values.dtype, np.datetime64):
        return values.astype("datetime64[ns]")
    if np.issubdtype(values.dtype, np.integer):
        return values.astype("datetime64[ns]")
    raise TypeError(f"time came back as {values.dtype}, which is not a datetime.")


def to_xarray(io, variables: Sequence[str]) -> xr.Dataset:
    """Convert an Earth2Studio ``ZarrBackend`` result to an xarray Dataset.

    Longitude is rewrapped to -180..180 and latitude sorted ascending so the
    result shares a convention with the TEMPO grid.
    """
    lat = np.asarray(io["lat"][:])
    lon = np.asarray(io["lon"][:])
    lead = _as_timedelta(np.asarray(io["lead_time"][:]))
    time = _as_datetime(np.asarray(io["time"][:]))

    data = {}
    for name in variables:
        arr = np.asarray(io[name][:])
        data[name] = (("time", "lead_time", "lat", "lon"), arr)

    ds = xr.Dataset(data, coords={"time": time, "lead_time": lead, "lat": lat, "lon": lon})

    if np.nanmax(ds["lon"].values) > 180.0:
        ds = ds.assign_coords(
            lon=(((ds["lon"].values + 180.0) % 360.0) - 180.0)
        ).sortby("lon")
    if ds["lat"].size > 1 and ds["lat"].values[0] > ds["lat"].values[-1]:
        ds = ds.sortby("lat")

    ds = ds.assign_coords(valid_time=ds["time"] + ds["lead_time"])
    for name in data:
        ds[name].attrs.setdefault("source", "Earth2Studio forecast")
    return ds


def select_valid_time(ds: xr.Dataset, target: datetime | np.datetime64) -> xr.Dataset:
    """Pick the forecast lead time whose valid time is closest to ``target``."""
    if isinstance(target, datetime):
        if target.tzinfo is not None:
            target = target.astimezone(UTC).replace(tzinfo=None)
        target = np.datetime64(target, "ns")

    valid = ds["valid_time"].values  # shape (time, lead_time)
    flat = np.abs(valid.astype("datetime64[ns]") - target.astype("datetime64[ns]"))
    i_time, i_lead = np.unravel_index(np.argmin(flat), flat.shape)
    picked = ds.isel(time=i_time, lead_time=i_lead)
    picked.attrs["lead_hours"] = float(
        ds["lead_time"].values[i_lead] / np.timedelta64(1, "h")
    )
    picked.attrs["valid_time"] = str(valid[i_time, i_lead])
    return picked


def regrid_to(source: xr.Dataset, target_lat, target_lon, method: str = "linear") -> xr.Dataset:
    """Interpolate a model field onto the (much finer) TEMPO grid.

    Both grids are regular lat/lon, so a separable bilinear interpolation is
    exact enough here and avoids pulling in a full regridding stack. TEMPO L3 is
    ~0.02 degrees against the model's 0.25 degrees, so this is pure upsampling:
    it adds no information, it just puts the wind on the same pixels as the NO2.
    """
    return source.interp(
        lat=np.asarray(target_lat), lon=np.asarray(target_lon), method=method
    )


def precomputed_forecast_path(config, model_name: str, init_time: datetime) -> Path:
    """Location of an optional reference forecast for teaching and tests."""
    stamp = init_time.strftime("%Y%m%dT%H%MZ")
    return Path(
        os.environ.get("WORKSHOP_PRECOMPUTED", "/opt/earth2/precomputed")
    ) / f"{model_name.lower()}_{stamp}.zarr"


def load_precomputed(path: str | Path) -> xr.Dataset:
    """Open a pre-baked forecast produced by ``scripts/prefetch_models.py``."""
    ds = xr.open_zarr(str(path), consolidated=True)
    if "valid_time" not in ds.coords and {"time", "lead_time"} <= set(ds.coords):
        ds = ds.assign_coords(valid_time=ds["time"] + ds["lead_time"])
    return ds
