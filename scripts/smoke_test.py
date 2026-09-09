#!/usr/bin/env python
"""Synthetic end-to-end check of the workshop analysis code.

Runs without a GPU, without Earth2Studio, and without an Earthdata login, so it
can gate every commit. It builds a synthetic TEMPO-shaped dataset containing a
Gaussian plume, translates that plume by a known distance, and asserts that
advecting the first field with the corresponding constant wind reproduces the
second.

That is the check worth having: it catches sign errors, latitude ordering, and
longitude convention mistakes, which are exactly the bugs that would otherwise
be found by an attendee staring at a plume pointing the wrong way.

    python scripts/smoke_test.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

# Some managed shells expose a read-only ~/.config. Keep Matplotlib's font
# cache in a task-specific writable directory so a smoke test never stalls on
# repeated cache generation.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tempo-earth2-mpl"))

import matplotlib

matplotlib.use("Agg")

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tempo_earth2 import advect, catalog, forecast, plots, sources
from tempo_earth2.tempo import (
    REGIONS,
    apply_quality_mask,
    granule_time,
    open_tempo_source,
    subset,
)

EARTH_RADIUS_M = 6_371_000.0
PASSED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  ok    {name}" + (f"  ({detail})" if detail else ""))
    else:
        print(f"  FAIL  {name}  {detail}")
        raise AssertionError(name)


def gaussian_plume(lat, lon, lat0, lon0, width_deg=0.35, amplitude=8e15):
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    return amplitude * np.exp(
        -(((la - lat0) ** 2 + (lo - lon0) ** 2) / (2 * width_deg**2))
    )


def build_synthetic_tempo(path: Path, u_ms: float, v_ms: float, dt_hours: float):
    """Two scans, the second being the first translated by a constant wind."""
    lat = np.arange(40.0, 43.0, 0.02)
    lon = np.arange(-75.0, -71.0, 0.02)

    lat0, lon0 = 41.2, -73.4
    # The exact displacement the wind implies, in degrees.
    dt = dt_hours * 3600.0
    dlat = np.degrees(v_ms * dt / EARTH_RADIUS_M)
    dlon = np.degrees(u_ms * dt / (EARTH_RADIUS_M * np.cos(np.radians(lat0))))

    scan0 = gaussian_plume(lat, lon, lat0, lon0)
    scan1 = gaussian_plume(lat, lon, lat0 + dlat, lon0 + dlon)

    # A smooth background so the field is not all plume, plus a stratospheric
    # column that the notebook must ignore.
    background = 1.5e15 * (1.0 + 0.1 * np.sin(np.radians(lon))[None, :])
    no2 = np.stack([scan0 + background, scan1 + background])

    times = np.array(
        [
            np.datetime64("2026-06-15T17:00:00", "ns"),
            np.datetime64("2026-06-15T17:00:00", "ns")
            + np.timedelta64(int(dt_hours * 3600), "s"),
        ]
    )

    ds = xr.Dataset(
        {
            "no2_trop": (("time", "lat", "lon"), no2),
            "no2_strat": (("time", "lat", "lon"), np.full_like(no2, 3e15)),
            "qa_flag": (("time", "lat", "lon"), np.zeros_like(no2, dtype=np.int8)),
            "cloud_fraction": (("time", "lat", "lon"), np.full_like(no2, 0.05)),
        },
        coords={"time": times, "lat": lat, "lon": lon},
    )
    ds.to_zarr(path, mode="w", consolidated=True, zarr_format=2)
    return ds, (dlat, dlon)


def build_synthetic_forecast(u_ms: float, v_ms: float) -> xr.Dataset:
    """Mimic the shape of forecast.to_xarray output on a coarse global grid."""
    lat = np.linspace(-90, 89.75, 720)
    lon = np.linspace(-180, 179.75, 1440)
    lead = np.array([np.timedelta64(h, "h") for h in (0, 6, 12)])
    time = np.array([np.datetime64("2026-06-15T12:00:00", "ns")])

    shape = (1, 3, 720, 1440)
    ds = xr.Dataset(
        {
            "u10m": (("time", "lead_time", "lat", "lon"), np.full(shape, u_ms)),
            "v10m": (("time", "lead_time", "lat", "lon"), np.full(shape, v_ms)),
            "u850": (("time", "lead_time", "lat", "lon"), np.full(shape, u_ms * 1.4)),
            "v850": (("time", "lead_time", "lat", "lon"), np.full(shape, v_ms * 1.4)),
        },
        coords={"time": time, "lead_time": lead, "lat": lat, "lon": lon},
    )
    return ds.assign_coords(valid_time=ds["time"] + ds["lead_time"])


def main() -> int:
    u_ms, v_ms, dt_hours = 7.5, -3.0, 1.0

    print("Building synthetic TEMPO store ...")
    with tempfile.TemporaryDirectory() as tmp:
        store = Path(tmp) / "tempo.zarr"
        _truth, (dlat, dlon) = build_synthetic_tempo(store, u_ms, v_ms, dt_hours)
        print(f"  plume displaced by {dlon:+.3f} lon, {dlat:+.3f} lat degrees")

        print("\nTEMPO loading")
        tempo = subset(open_tempo_source(str(store)), region=REGIONS["northeast"])
        check("open_tempo_source returns lat/lon/time", set(tempo.dims) >= {"time", "lat", "lon"})
        check("latitude ascending", bool(tempo["lat"][0] < tempo["lat"][-1]))
        check("longitude in -180..180", bool(tempo["lon"].max() <= 180))
        check("region subset non-empty", tempo["lat"].size > 0 and tempo["lon"].size > 0,
              f"{tempo['lat'].size} x {tempo['lon'].size}")

        no2_t0 = apply_quality_mask(tempo.isel(time=0))
        no2_t1 = apply_quality_mask(tempo.isel(time=1))
        check("quality mask keeps clean pixels", bool(np.isfinite(no2_t0.values).all()))

        print("\nForecast handling")
        fx = build_synthetic_forecast(u_ms, v_ms)
        target = datetime(2026, 6, 15, 17, 30, tzinfo=UTC)
        picked = forecast.select_valid_time(fx, target)
        check("select_valid_time picks nearest lead", picked.attrs["lead_hours"] == 6.0,
              f"lead {picked.attrs['lead_hours']} h")

        init = forecast.nearest_init_time(datetime(2026, 6, 15, 17, 38, tzinfo=UTC))
        check("nearest_init_time rounds down", init.hour == 12, f"{init:%H:%M}Z")
        # 12:00Z init, 17:30Z target: one 6-hour step reaches 18:00Z, which
        # covers it. A 19:00Z target needs two.
        check("steps_to_cover (within one step)",
              forecast.steps_to_cover(init, target) == 1)
        check("steps_to_cover (needs two)",
              forecast.steps_to_cover(init, datetime(2026, 6, 15, 19, 0, tzinfo=UTC)) == 2)

        wind = forecast.regrid_to(
            picked[["u10m", "v10m"]], no2_t0["lat"].values, no2_t0["lon"].values
        )
        check("regrid preserves grid shape",
              wind["u10m"].shape == no2_t0.shape,
              f"{wind['u10m'].shape}")
        check("regrid preserves wind value",
              np.allclose(wind["u10m"].values, u_ms, atol=1e-6))

        print("\nAdvection (the real test)")
        advected, skill = advect.experiment(
            no2_t0, no2_t1, wind["u10m"], wind["v10m"], dt_hours=dt_hours, substeps=8
        )

        # The advected field should reproduce scan 1 far better than persistence.
        check("advection beats persistence", skill.skill_score > 0.8,
              f"skill {skill.skill_score:+.3f}")
        check("advected RMSE is small relative to plume amplitude",
              skill.rmse_advected < 0.05 * 8e15,
              f"{skill.rmse_advected:.2e}")

        # Where is the advected maximum? It must land on the true t1 maximum.
        def argmax_latlon(field):
            arr = np.nan_to_num(field.values, nan=-np.inf)
            j, i = np.unravel_index(np.argmax(arr), arr.shape)
            return float(field["lat"][j]), float(field["lon"][i])

        adv_peak = argmax_latlon(advected)
        true_peak = argmax_latlon(no2_t1)
        t0_peak = argmax_latlon(no2_t0)
        offset = np.hypot(adv_peak[0] - true_peak[0], adv_peak[1] - true_peak[1])
        moved = np.hypot(t0_peak[0] - true_peak[0], t0_peak[1] - true_peak[1])
        check("advected peak lands on the observed peak", offset < 0.05,
              f"offset {offset:.3f} deg vs {moved:.3f} deg unmoved")
        check("the plume actually moved (test is not trivial)", moved > 0.1)

        print("\nNaN handling")
        holed = no2_t0.copy()
        holed.values[60:75, 60:75] = np.nan  # a 15x15 hole in a 150x200 grid
        holed_fraction = float(np.isfinite(holed.values).mean())
        out = advect.advect(holed, wind["u10m"], wind["v10m"], dt_hours=dt_hours)
        out_fraction = float(np.isfinite(out.values).mean())

        # Loss at the upwind boundary is expected and correct: the wind carries
        # air in from outside the region and we do not know what was in it. The
        # width of that strip is the displacement, so compute it and exclude it
        # rather than picking a tolerance by hand.
        cols_lost = int(np.ceil(abs(dlon) / 0.02)) + 1
        rows_lost = int(np.ceil(abs(dlat) / 0.02)) + 1
        interior = out.values[rows_lost:-rows_lost or None, cols_lost:-cols_lost or None]
        interior_finite = float(np.isfinite(interior).mean())

        check("upwind boundary is marked unknown rather than fabricated",
              not np.isfinite(out.values[:, 0]).any(),
              f"west edge blanked ({cols_lost} columns) for eastward wind")
        check("the hole moves rather than growing without bound",
              interior_finite > holed_fraction - 0.02,
              f"interior {interior_finite:.1%} finite vs {holed_fraction:.1%} input "
              f"(overall {out_fraction:.1%}, boundary strip excluded)")

        print("\nDiagnostics")
        speed = advect.wind_speed(wind["u10m"], wind["v10m"])
        check("wind_speed magnitude", np.allclose(speed.values, np.hypot(u_ms, v_ms)))
        # A constant wind field gives a degenerate ventilation curve; use a
        # varying one so the binning path is genuinely exercised.
        varying = speed + 4.0 * np.sin(np.radians(speed["lon"] * 40))
        rel = advect.ventilation_relationship(no2_t0, varying, bins=8)
        check("ventilation_relationship bins", rel.sizes["wind_speed"] == 8)
        check("ventilation bins are populated", int(rel["count"].sum()) == int(np.isfinite(no2_t0.values).sum()))

        print("\nPlotting")
        plots.plot_no2(no2_t0, title="synthetic")
        plots.plot_no2_with_wind(no2_t0, wind["u10m"], wind["v10m"])
        plots.plot_advection_triptych(no2_t0, advected, no2_t1, skill=skill)
        plots.plot_ventilation(rel)
        check("all plot helpers ran", True)

        print("\nEarth2Studio IO conversion")
        # to_xarray reads a ZarrBackend by __getitem__, so a dict of numpy
        # arrays exercises exactly the same path without needing Earth2Studio.
        fake_io = {
            "lat": np.linspace(90, -90, 720, endpoint=False),
            "lon": np.linspace(0, 360, 1440, endpoint=False),
            "lead_time": np.array([0, 6, 12], dtype="timedelta64[h]"),
            "time": np.array(["2026-06-15T12:00:00"], dtype="datetime64[ns]"),
            "u10m": np.full((1, 3, 720, 1440), 5.0, dtype="float32"),
            "v10m": np.full((1, 3, 720, 1440), -2.0, dtype="float32"),
        }
        converted = forecast.to_xarray(fake_io, ["u10m", "v10m"])
        check("to_xarray rewraps longitude to -180..180",
              float(converted["lon"].max()) <= 180.0 and float(converted["lon"].min()) >= -180.0)
        check("to_xarray sorts latitude ascending",
              bool(converted["lat"][0] < converted["lat"][-1]))
        check("to_xarray builds valid_time",
              str(converted["valid_time"].values[0, 1])[:16] == "2026-06-15T18:00",
              str(converted["valid_time"].values[0, 1])[:16])

        # The integer-lead-time fallback: a store that lost the timedelta dtype
        # must still produce the right valid times, not values off by 3.6e12.
        integer_io = dict(fake_io, lead_time=np.array([0, 6, 12], dtype="int64"))
        recovered = forecast.to_xarray(integer_io, ["u10m"])
        check("integer lead_time is recovered as hours",
              np.array_equal(recovered["lead_time"].values, converted["lead_time"].values),
              str(recovered["lead_time"].values))

        print("\nResult assembly (the notebook's save step)")
        picked_wind = forecast.regrid_to(
            forecast.select_valid_time(converted, datetime(2026, 6, 15, 18, 0, tzinfo=UTC))[
                ["u10m", "v10m"]
            ],
            no2_t0["lat"].values,
            no2_t0["lon"].values,
        )
        # no2_t0, no2_t1 and the wind each carry a different scalar time
        # coordinate; merging them without dropping those is a conflict.
        bare = lambda da: da.reset_coords(drop=True)
        merged = xr.Dataset(
            {
                "no2_t0": bare(no2_t0),
                "no2_t1": bare(no2_t1),
                "no2_advected": bare(advected),
                "u10m": bare(picked_wind["u10m"]),
            }
        )
        check("differing scalar time coords merge after reset_coords",
              set(merged.coords) == {"lat", "lon"},
              f"coords: {sorted(merged.coords)}")
        check("merged result has all fields", len(merged.data_vars) == 4)

        print("\nMisc")
        stamp = granule_time("TEMPO_NO2_L3_V04_20260615T170903Z_S009.nc")
        check("granule_time parses", stamp == datetime(2026, 6, 15, 17, 9, 3, tzinfo=UTC),
              str(stamp))

        print("\nCatalog and bounded-source helpers")
        guaranteed = catalog.model_entries(("guaranteed",))
        check("catalog has a guaranteed FCN", any(item["id"] == "fcn" for item in guaranteed))
        check("catalog does not promise DLWP live",
              not next(item for item in catalog.model_entries() if item["id"] == "dlwp")["live_inference"])
        data = catalog.data_entries(("wildfire",))
        check("wildfire catalog spans multiple sources", len(data) >= 2, str([item["id"] for item in data]))
        usgs_url = sources.usgs_instantaneous_values_url(
            "01100000", "2026-06-01", "2026-06-03"
        )
        check("USGS helper bounds sites and dates",
              "sites=01100000" in usgs_url and "startDT=2026-06-01" in usgs_url)
        coops_url = sources.noaa_coops_url(
            "8443970", "2026-06-01", "2026-06-03"
        )
        check("CO-OPS helper fixes UTC and units",
              "time_zone=gmt" in coops_url and "units=metric" in coops_url)

    print(f"\n{len(PASSED)} checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
