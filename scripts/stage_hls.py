#!/usr/bin/env python
"""Stage a small, real NASA HLS Landsat cutout as analysis-ready NetCDF."""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import earthaccess
import numpy as np
import rasterio
import xarray as xr
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

BANDS = {
    "B02": "blue",
    "B03": "green",
    "B04": "red",
    "B05": "nir",
    "B06": "swir1",
    "B07": "swir2",
    "Fmask": "fmask",
}


def _asset_links(granule) -> dict[str, str]:
    selected = {}
    for url in granule.data_links():
        clean = url.split("?", 1)[0]
        for code in BANDS:
            if clean.endswith(f".{code}.tif"):
                selected[code] = url
    missing = sorted(set(BANDS) - set(selected))
    if missing:
        raise RuntimeError(f"HLS granule is missing assets: {missing}")
    return selected


def _read_cutout(path: Path, bbox) -> tuple[np.ma.MaskedArray, dict]:
    with rasterio.open(path) as source:
        projected = transform_bounds("EPSG:4326", source.crs, *bbox)
        window = from_bounds(*projected, transform=source.transform).round_offsets().round_lengths()
        data = source.read(1, window=window, masked=True)
        transform = source.window_transform(window)
        if transform.b or transform.d:
            raise RuntimeError("Rotated HLS grids are not supported by this cutout writer")
        x = transform.c + (np.arange(data.shape[1]) + 0.5) * transform.a
        y = transform.f + (np.arange(data.shape[0]) + 0.5) * transform.e
        metadata = {
            "x": x,
            "y": y,
            "crs": source.crs.to_string(),
            "transform": tuple(transform),
            "nodata": source.nodata,
        }
    return data, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--granule-id", required=True)
    parser.add_argument("--bbox", type=float, nargs=4, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    earthaccess.login(strategy="netrc")
    results = earthaccess.search_data(
        short_name="HLSL30",
        version="2.0",
        bounding_box=tuple(args.bbox),
        temporal=("2026-01-01", "2026-12-31"),
        count=200,
    )
    matches = [
        result
        for result in results
        if result.get("meta", {}).get("native-id") == args.granule_id
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {args.granule_id} result; found {len(matches)}")
    links = _asset_links(matches[0])

    with tempfile.TemporaryDirectory(prefix="hls-stage-") as temporary:
        downloaded = earthaccess.download(list(links.values()), temporary)
        by_name = {Path(path).name: Path(path) for path in downloaded}
        arrays, spatial = {}, None
        for code, name in BANDS.items():
            filename = Path(links[code].split("?", 1)[0]).name
            values, metadata = _read_cutout(by_name[filename], args.bbox)
            if spatial is None:
                spatial = metadata
            elif values.shape != next(iter(arrays.values())).shape:
                raise RuntimeError("HLS bands did not produce identical cutout grids")
            if code == "Fmask":
                arrays[name] = values.filled(255).astype("uint8")
            else:
                arrays[name] = values.astype("float32").filled(np.nan) * 0.0001

    assert spatial is not None
    dataset = xr.Dataset(
        {name: (("y", "x"), values) for name, values in arrays.items()},
        coords={"x": spatial["x"], "y": spatial["y"]},
        attrs={
            "source": "NASA Harmonized Landsat Sentinel-2 (HLS) L30 v2.0",
            "granule_id": args.granule_id,
            "acquired_utc": args.granule_id.split(".")[3],
            "crs": spatial["crs"],
            "bbox_lonlat": ",".join(map(str, args.bbox)),
            "processing": "Source COGs cropped to bbox; reflectance scaled by 0.0001; no spatial resampling.",
        },
    )
    for name in set(BANDS.values()) - {"fmask"}:
        dataset[name].attrs.update(units="1", long_name=f"HLS surface reflectance: {name}")
    dataset.fmask.attrs.update(
        long_name="HLS quality bitmask",
        note="bits 0-4: cirrus, cloud, adjacent cloud, shadow, snow/ice",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(args.output, engine="h5netcdf")
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "source": "NASA HLS L30 v2.0",
        "granule_id": args.granule_id,
        "bbox": args.bbox,
        "assets": {code: url.split("?", 1)[0] for code, url in links.items()},
        "output": str(args.output),
        "shape": dict(dataset.sizes),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps({"output": str(args.output), "shape": manifest["shape"]}, indent=2))


if __name__ == "__main__":
    main()
