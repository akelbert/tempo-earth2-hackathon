#!/usr/bin/env python
"""Stage a bounded NOAA CO-OPS water-level or prediction time series."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tempo_earth2.sources import noaa_coops_url, read_noaa_coops


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", required=True)
    parser.add_argument("--station-name", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument(
        "--product", choices=("water_level", "predictions"), required=True
    )
    parser.add_argument("--datum", default="MSL")
    parser.add_argument("--units", default="metric")
    parser.add_argument("--interval", default="6")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    url = noaa_coops_url(
        args.station,
        args.start_date,
        args.end_date,
        product=args.product,
        datum=args.datum,
        units=args.units,
        interval=args.interval,
    )
    frame = read_noaa_coops(
        args.station,
        args.start_date,
        args.end_date,
        product=args.product,
        datum=args.datum,
        units=args.units,
        interval=args.interval,
    )
    if frame.empty:
        raise SystemExit("NOAA CO-OPS returned no observations")

    frame["station_name"] = args.station_name
    frame["product"] = args.product
    frame["datum"] = args.datum
    frame["units"] = args.units
    frame["source"] = "NOAA CO-OPS Data API"
    frame["source_url"] = url
    ordered = [
        "time_utc",
        "value",
        "station",
        "station_name",
        "product",
        "datum",
        "units",
        "source",
        "source_url",
    ]
    ordered.extend(column for column in frame if column not in ordered)
    frame = frame[ordered]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": "NOAA CO-OPS Data API",
        "source_url": url,
        "station": args.station,
        "station_name": args.station_name,
        "product": args.product,
        "datum": args.datum,
        "units": args.units,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "rows": len(frame),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(frame):,} rows to {args.output}")
    print(f"Wrote provenance to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
