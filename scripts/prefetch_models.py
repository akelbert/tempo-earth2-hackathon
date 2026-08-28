#!/usr/bin/env python
"""Populate the shared Earth2Studio model cache, and optionally pre-bake forecasts.

Two jobs, both of which exist to remove event-day dependencies:

1. ``--model FCN`` downloads the checkpoint into ``EARTH2STUDIO_MODEL_CACHE``.
   Run this at image build time (``--build-arg PREFETCH_MODEL=true``) or as a
   node pre-warm step. Fifty pods each downloading the same multi-gigabyte
   checkpoint from Hugging Face at 09:05 on day one is a failure mode, not a
   plan.

2. ``--precompute`` additionally runs the forecast and writes it to
   ``WORKSHOP_PRECOMPUTED``. That is what makes the CPU fallback path usable:
   the guided exercises read a pre-baked forecast instead of running inference.

    python scripts/prefetch_models.py --model FCN
    python scripts/prefetch_models.py --model FCN --precompute \\
        --init-time 2026-06-15T12:00 --nsteps 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, "/opt/earth2/lib")

from tempo_earth2.forecast import (  # noqa: E402
    SUPPORTED_MODELS,
    TRANSPORT_VARIABLES,
    load_model,
    nearest_init_time,
    run_forecast,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("WORKSHOP_MODEL", "FCN"),
        choices=sorted(SUPPORTED_MODELS),
    )
    parser.add_argument(
        "--precompute",
        action="store_true",
        help="also run and store a forecast for the CPU fallback path",
    )
    parser.add_argument(
        "--init-time",
        help="forecast initialization, ISO 8601; snapped down to the model timestep",
    )
    parser.add_argument("--nsteps", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("WORKSHOP_PRECOMPUTED", "/opt/earth2/precomputed"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    cache = os.environ.get("EARTH2STUDIO_MODEL_CACHE", "<default ~/.cache/earth2studio>")
    print(f"Model cache: {cache}")
    print(f"Loading {args.model} ...")

    model = load_model(args.model)
    print(f"Loaded {args.model}.")

    if cache != "<default ~/.cache/earth2studio>" and Path(cache).exists():
        size = sum(f.stat().st_size for f in Path(cache).rglob("*") if f.is_file())
        print(f"Cache now holds {size / 1024**3:.2f} GB")

    if not args.precompute:
        return 0

    if args.init_time:
        target = datetime.fromisoformat(args.init_time)
    else:
        raise SystemExit("--precompute requires --init-time")
    init = nearest_init_time(target).replace(tzinfo=None)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    store = output_dir / f"{args.model.lower()}_{init:%Y%m%dT%H%M}Z.zarr"

    print(f"\nRunning {args.model} from {init:%Y-%m-%d %H:%MZ} for {args.nsteps} steps ...")
    result = run_forecast(
        init,
        nsteps=args.nsteps,
        model_name=args.model,
        variables=TRANSPORT_VARIABLES,
        store_path=store,
        model=model,
    )

    summary = result.summary()
    print(json.dumps(summary, indent=2))
    (output_dir / f"{store.stem}.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {store}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
