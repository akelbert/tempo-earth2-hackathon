#!/usr/bin/env python
"""Generate synthetic TEMPO scans and a matching forecast, for testing.

This exists so the notebook can be run on a laptop **before** anyone has a NASA
Earthdata login or a staged TEMPO copy. The output is clearly labeled
synthetic and is not science - it is three Gaussian plumes over the northeast
corridor, drifting with a constant wind.

Its value is that the answer is known: the scans are built by advecting the
plumes with the forecast's own 10 m wind, so the notebook's skill score should
come out near +1.0. If it does not, something in the pipeline is wrong, and you
find that out in ten seconds instead of after a 40-minute staging download.

    python scripts/make_demo_data.py --output data

Then point the notebook at it:

    export TEMPO_DATA_URI=$PWD/data/tempo/northeast.zarr
    export WORKSHOP_PRECOMPUTED=$PWD/data/precomputed

For real data, use scripts/stage_tempo.py instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import _synthetic

U_MS, V_MS, DT_HOURS = 7.5, -3.0, 1.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="data", help="destination directory")
    parser.add_argument("--scans", type=int, default=4)
    args = parser.parse_args(argv)

    root = Path(args.output).resolve()
    tempo = root / "tempo" / "northeast.zarr"
    precomputed = root / "precomputed"
    tempo.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing synthetic TEMPO scans to {tempo} ...")
    ds = _synthetic.build_tempo(
        tempo, u_ms=U_MS, v_ms=V_MS, dt_hours=DT_HOURS, nscans=args.scans
    )

    print(f"Writing synthetic forecast to {precomputed} ...")
    store = _synthetic.write_precomputed(precomputed, "FCN", u_ms=U_MS, v_ms=V_MS)

    scans = [np.datetime_as_string(t, unit="s") for t in ds["time"].values]
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "collection": "SYNTHETIC (not TEMPO)",
        "version": "n/a",
        "region": "northeast",
        "bbox": [-76.0, 38.5, -70.0, 43.5],
        "scans": scans,
        "variables": sorted(ds.data_vars),
        "primary_variable": "no2_trop",
        "shape": {k: int(v) for k, v in ds.sizes.items()},
        "store": str(tempo),
        "note": (
            "Synthetic test data. Three Gaussian plumes advected by a constant "
            f"{U_MS} m/s eastward, {V_MS} m/s northward wind. Not science."
        ),
    }
    (tempo / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("\nScans:")
    for scan in scans:
        print(f"  {scan}")

    print(f"\nDone. {ds.nbytes / 1024**2:.0f} MB of TEMPO-shaped data.\n")
    print("Point the notebook at it with:\n")
    print(f'  export TEMPO_DATA_URI="{tempo}"')
    print(f'  export WORKSHOP_PRECOMPUTED="{precomputed}"')
    print(f"\nForecast store: {store}")
    print(
        "\nIn the notebook, set USE_REFERENCE_FORECAST = True. Expect a skill score "
        "near +0.98:\nthe scans were built from this exact wind field."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
