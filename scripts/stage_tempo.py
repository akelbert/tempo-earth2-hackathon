#!/usr/bin/env python
"""Stage a regional TEMPO NO2 subset for the workshop.

A full TEMPO L3 granule is a ~850 MB full-disk file, and the notebook needs
several consecutive scans. Fifty attendees each pulling that from NASA's
us-west-2 buckets on event day is not a plan. This script runs once, ahead of
the event, and produces a small regional Zarr store plus a manifest that the
notebooks read to discover which scans exist.

    # one-off, from a machine with an Earthdata login
    python scripts/stage_tempo.py \\
        --date 2026-06-15 --region northeast --scans 6 \\
        --output ./data/tempo/northeast.zarr

    # then publish it
    gsutil -m rsync -r ./data/tempo gs://WORKSHOP_BUCKET/tempo

Authentication: run ``earthaccess.login(persist=True)`` once, or set
EARTHDATA_USERNAME and EARTHDATA_PASSWORD. Personal credentials must not be
baked into the attendee image.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tempo_earth2.tempo import (
    NO2_TROPOSPHERIC,
    REGIONS,
    Region,
    granule_time,
    open_tempo_granule,
)

# Support-group fields worth keeping: cloud fraction drives the quality screen,
# and the solar zenith angle explains most of the remaining scan-to-scan
# variability an attendee will notice.
SUPPORT_VARIABLES = {
    "eff_cloud_fraction": "cloud_fraction",
    "solar_zenith_angle": "solar_zenith_angle",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--date", required=True, help="UTC date, YYYY-MM-DD")
    parser.add_argument(
        "--region",
        default="northeast",
        help=f"named region ({', '.join(sorted(REGIONS))}) or 'W,S,E,N'",
    )
    parser.add_argument(
        "--scans",
        type=int,
        default=6,
        help="number of consecutive scans to stage (default 6)",
    )
    parser.add_argument(
        "--start-hour",
        type=int,
        default=15,
        help="earliest UTC hour to consider; default 15 (mid-morning in the east)",
    )
    parser.add_argument("--output", required=True, help="destination .zarr path or URI")
    parser.add_argument("--short-name", default="TEMPO_NO2_L3")
    parser.add_argument(
        "--version",
        default="V04",
        help="collection version; V03 ended 2025-09-16, V04 is ongoing",
    )
    parser.add_argument(
        "--keep-granules",
        metavar="DIR",
        help="keep the downloaded granules in DIR instead of a temporary directory",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="search and report, download nothing"
    )
    return parser.parse_args(argv)


def resolve_region(spec: str) -> Region:
    if spec in REGIONS:
        return REGIONS[spec]
    try:
        west, south, east, north = (float(x) for x in spec.split(","))
    except ValueError as exc:
        raise SystemExit(
            f"Region {spec!r} is neither a known name ({', '.join(sorted(REGIONS))}) "
            "nor a 'W,S,E,N' bounding box."
        ) from exc
    return Region("custom", west, south, east, north)


def search(args: argparse.Namespace, region: Region) -> list:
    import earthaccess

    earthaccess.login(persist=True)

    day = datetime.strptime(args.date, "%Y-%m-%d").replace(tzinfo=UTC)
    start = day.replace(hour=args.start_hour)
    # TEMPO scans hourly during daylight; the following UTC morning bounds a day.
    end = (day.replace(hour=23, minute=59, second=59))

    results = earthaccess.search_data(
        short_name=args.short_name,
        version=args.version,
        temporal=(start.strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S")),
        bounding_box=region.bbox,
    )
    if not results:
        raise SystemExit(
            f"No {args.short_name} {args.version} granules for {args.date} "
            f"after {args.start_hour:02d}Z over {region.name}. "
            "Check the date is within the collection's coverage."
        )
    return results[: args.scans]


def load_support(path: Path) -> xr.Dataset | None:
    """Read the support_data group, which carries the cloud screen."""
    try:
        support = xr.open_dataset(
            str(path), group="support_data", engine="h5netcdf", decode_times=False
        )
    except (OSError, KeyError, ValueError):
        return None

    keep = {src: dst for src, dst in SUPPORT_VARIABLES.items() if src in support}
    if not keep:
        return None
    support = support[list(keep)]
    # support_data's own latitude/longitude are unindexed dims at TEMPO's full
    # native domain resolution, distinct from - and much larger than - the
    # product group's lat/lon (which open_tempo_granule already renamed).
    # stage_granule() below assigns the product's real lat/lon coordinate
    # values onto this dataset by name; without renaming these dims first,
    # that assign_coords silently creates two unrelated same-length
    # dimensions instead of landing on the one dimension both groups share,
    # so the later region .sel() never crops these variables and they end up
    # misaligned with everything else in the merged dataset.
    dim_renames = {
        candidate: target
        for candidate, target in (("latitude", "lat"), ("longitude", "lon"))
        if candidate in support.dims
    }
    return support.rename({**keep, **dim_renames})


def stage_granule(path: Path, region: Region) -> xr.Dataset:
    ds = open_tempo_granule(path)
    support = load_support(path)
    if support is not None:
        # The support group shares the product group's dimensions but not its
        # coordinates, so attach the product coordinates before subsetting.
        support = support.assign_coords(
            {k: ds[k] for k in ("lat", "lon", "time") if k in ds.coords}
        )
        ds = xr.merge([ds, support], combine_attrs="override")

    ds = ds.sel(
        lat=slice(region.lat_min, region.lat_max),
        lon=slice(region.lon_min, region.lon_max),
    )

    if "time" not in ds.dims:
        stamp = granule_time(path)
        ds = ds.expand_dims(time=[np.datetime64(stamp.replace(tzinfo=None), "ns")])

    return ds.load()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    region = resolve_region(args.region)
    granules = search(args, region)

    print(f"Found {len(granules)} granule(s) for {args.date} over {region.name}:")
    total_mb = 0.0
    for granule in granules:
        size = float(granule.size() or 0.0)
        total_mb += size
        print(f"  {granule['meta']['native-id']}  ({size:.0f} MB)")
    print(f"Download volume: {total_mb / 1024:.1f} GB")

    if args.dry_run:
        return 0

    import earthaccess

    workdir = Path(args.keep_granules) if args.keep_granules else Path(
        tempfile.mkdtemp(prefix="tempo-stage-")
    )
    workdir.mkdir(parents=True, exist_ok=True)

    try:
        print(f"\nDownloading to {workdir} ...")
        files = earthaccess.download(granules, str(workdir))

        staged = []
        for path in sorted(files, key=lambda p: str(p)):
            print(f"  subsetting {Path(path).name}")
            staged.append(stage_granule(Path(path), region))

        combined = xr.concat(staged, dim="time", combine_attrs="override").sortby("time")
        combined.attrs.update(
            {
                "title": f"TEMPO {args.short_name} {args.version} regional subset",
                "region": region.name,
                "bbox": list(region.bbox),
                "staged_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                "source_collection": f"{args.short_name} {args.version}",
                "source_granules": [g["meta"]["native-id"] for g in granules],
            }
        )

        # Chunk one scan per chunk: the notebook reads whole scans, and a
        # per-scan chunk keeps a single read from pulling the entire series.
        encoding = {
            name: {"chunks": (1,) + combined[name].shape[1:]}
            for name in combined.data_vars
            if combined[name].ndim >= 3
        }

        output = args.output
        print(f"\nWriting {output} ...")
        combined.to_zarr(
            output,
            mode="w",
            consolidated=True,
            encoding=encoding,
            zarr_format=2,
        )

        scans = [
            np.datetime_as_string(t, unit="s") for t in combined["time"].values
        ]
        manifest = {
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "collection": args.short_name,
            "version": args.version,
            "region": region.name,
            "bbox": list(region.bbox),
            "scans": scans,
            "variables": sorted(combined.data_vars),
            "primary_variable": NO2_TROPOSPHERIC,
            "shape": {k: int(v) for k, v in combined.sizes.items()},
            "store": str(output),
        }
        # pathlib collapses the "//" in a gs:// URI, so remote stores need
        # string concatenation and fsspec rather than Path.
        base = str(output).rstrip("/")
        if base.startswith(("gs://", "s3://", "http://", "https://")):
            import fsspec

            manifest_path = f"{base}/manifest.json"
            with fsspec.open(manifest_path, "wt") as handle:
                json.dump(manifest, handle, indent=2)
        else:
            manifest_path = Path(base) / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2))

        print("\nStaged scans:")
        for scan in scans:
            print(f"  {scan}")
        nbytes = combined.nbytes / 1024**2
        print(f"\nDone. {nbytes:.0f} MB in memory, manifest at {manifest_path}")
        return 0
    finally:
        if not args.keep_granules:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
