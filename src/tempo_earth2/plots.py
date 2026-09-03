"""Plot helpers for the workshop notebooks.

Cartopy is used when available for coastlines and borders, and the plots
degrade to plain matplotlib axes when it is not. Nothing here is required by the
science - it exists so the notebook cells stay three lines long.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

#: TEMPO tropospheric NO2 columns are ~1e15-1e16 molecules/cm2 over cities.
NO2_SCALE = 1e15
NO2_LABEL = r"tropospheric NO$_2$ column ($10^{15}$ molecules cm$^{-2}$)"


def _cartopy():
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        return ccrs, cfeature
    except Exception:  # noqa: BLE001  # cartopy has optional native dependencies
        return None, None


def _make_axes(fig, position, extent=None):
    ccrs, cfeature = _cartopy()
    if ccrs is None:
        ax = fig.add_subplot(*position)
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        return ax, None
    ax = fig.add_subplot(*position, projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4)
    ax.add_feature(cfeature.STATES, linewidth=0.3, alpha=0.6)
    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    # Do not suppress individual gridliner sides (e.g. gl.top_labels = False).
    # cartopy 0.25's GeoAxes.get_tightbbox() returns a degenerate
    # Bbox(nan, y0, nan, inf) whenever any side's labels are turned off after
    # creation - independent of which features are on the axes, reproduced
    # even with none at all. Jupyter's inline backend saves displayed figures
    # with bbox_inches='tight' by default, and a NaN bbox makes matplotlib's
    # tight-bbox union silently drop the whole GeoAxes, leaving only the
    # colorbar visible. Leaving all four sides labeled duplicates the lon/lat
    # ticks on the top and right edges - a minor cosmetic cost against a plot
    # that otherwise doesn't render its data at all.
    ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4)
    return ax, ccrs.PlateCarree()


def _robust_limits(field: xr.DataArray, low: float = 2.0, high: float = 98.0):
    values = np.asarray(field.values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    return float(np.percentile(values, low)), float(np.percentile(values, high))


def plot_no2(
    field: xr.DataArray,
    title: str = "TEMPO tropospheric NO$_2$",
    vmin: float | None = None,
    vmax: float | None = None,
    ax=None,
    cmap: str = "magma_r",
    add_colorbar: bool = True,
    **kwargs: Any,
):
    """Map one TEMPO NO2 field, scaled to 1e15 molecules/cm2."""
    import matplotlib.pyplot as plt

    scaled = field / NO2_SCALE
    if vmin is None or vmax is None:
        lo, hi = _robust_limits(scaled)
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax

    created = ax is None
    if created:
        fig = plt.figure(figsize=(8, 6))
        extent = [
            float(field["lon"].min()),
            float(field["lon"].max()),
            float(field["lat"].min()),
            float(field["lat"].max()),
        ]
        ax, transform = _make_axes(fig, (1, 1, 1), extent)
    else:
        fig = ax.get_figure()
        transform = getattr(ax, "projection", None)
        transform = None if transform is None else transform.__class__()

    mesh_kwargs = {"cmap": cmap, "vmin": vmin, "vmax": vmax, "shading": "auto"}
    if transform is not None:
        mesh_kwargs["transform"] = transform
    mesh_kwargs.update(kwargs)

    mesh = ax.pcolormesh(
        field["lon"].values, field["lat"].values, scaled.values, **mesh_kwargs
    )
    ax.set_title(title, fontsize=11)
    if add_colorbar:
        fig.colorbar(mesh, ax=ax, orientation="vertical", shrink=0.8, label=NO2_LABEL)
    return ax, mesh


def plot_no2_with_wind(
    no2: xr.DataArray,
    u: xr.DataArray,
    v: xr.DataArray,
    title: str = "TEMPO NO$_2$ with Earth-2 forecast wind",
    every: int | None = None,
    **kwargs: Any,
):
    """Map NO2 with forecast wind barbs overlaid.

    ``every`` subsamples the wind so the arrows stay readable; by default it is
    chosen to give roughly 25 arrows across the domain.
    """
    ax, _ = plot_no2(no2, title=title, **kwargs)

    if every is None:
        every = max(1, int(no2["lon"].size // 25))

    us = u.isel(lat=slice(None, None, every), lon=slice(None, None, every))
    vs = v.isel(lat=slice(None, None, every), lon=slice(None, None, every))

    quiver_kwargs: dict[str, Any] = {
        "color": "tab:cyan",
        "scale": 350,
        "width": 0.0025,
        "alpha": 0.9,
    }
    projection = getattr(ax, "projection", None)
    if projection is not None:
        quiver_kwargs["transform"] = projection.__class__()

    ax.quiver(
        us["lon"].values, us["lat"].values, us.values, vs.values, **quiver_kwargs
    )
    return ax


def plot_advection_triptych(
    observed_t0: xr.DataArray,
    advected: xr.DataArray,
    observed_t1: xr.DataArray,
    skill=None,
    figsize: tuple[float, float] = (16, 5),
):
    """Three panels: what TEMPO saw, what advection predicted, what TEMPO saw next.

    All three panels share a color scale, because the whole point is to compare
    them by eye before believing the score.
    """
    import matplotlib.pyplot as plt

    scaled = observed_t1 / NO2_SCALE
    vmin, vmax = _robust_limits(scaled)

    fig = plt.figure(figsize=figsize)
    extent = [
        float(observed_t0["lon"].min()),
        float(observed_t0["lon"].max()),
        float(observed_t0["lat"].min()),
        float(observed_t0["lat"].max()),
    ]

    panels = [
        (observed_t0, "TEMPO at $t_0$ (observed)"),
        (advected, "$t_0$ advected by Earth-2 wind (predicted)"),
        (observed_t1, "TEMPO at $t_1$ (observed)"),
    ]

    mesh = None
    for index, (field, title) in enumerate(panels, start=1):
        ax, _ = _make_axes(fig, (1, 3, index), extent)
        _, mesh = plot_no2(
            field, title=title, vmin=vmin, vmax=vmax, ax=ax, add_colorbar=False
        )

    fig.colorbar(
        mesh, ax=fig.axes, orientation="vertical", shrink=0.75, label=NO2_LABEL
    )
    if skill is not None:
        fig.suptitle(str(skill), fontsize=11)
    return fig


def plot_ventilation(relationship: xr.Dataset, ax=None):
    """Plot median NO2 against wind speed, with sample counts."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))

    speed = relationship["wind_speed"].values
    median = relationship["median_no2"].values / NO2_SCALE
    ax.plot(speed, median, marker="o", color="tab:blue")
    ax.set_xlabel("Earth-2 forecast 10 m wind speed (m s$^{-1}$)")
    ax.set_ylabel(NO2_LABEL)
    ax.set_title("Ventilation: NO$_2$ column against forecast wind speed")
    ax.grid(alpha=0.3)

    counts = ax.twinx()
    counts.bar(speed, relationship["count"].values, alpha=0.15, color="grey",
               width=(speed[1] - speed[0]) * 0.8 if speed.size > 1 else 1.0)
    counts.set_ylabel("pixels per bin")
    return ax
