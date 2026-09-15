#!/usr/bin/env python
"""Populate the shared Earth2Studio model cache, and optionally pre-bake forecasts.

Two jobs, both of which exist to remove event-day dependencies:

1. ``--models FCN,DLWP,PrecipitationAFNO`` downloads the reviewed checkpoints
   into ``EARTH2STUDIO_MODEL_CACHE``. Run this at image build time
   (``--build-arg PREFETCH_MODEL=true``) or as a node pre-warm step. Fifty pods
   each downloading the same checkpoints at 09:05 on day one is a failure
   mode, not a plan.

2. ``--precompute`` additionally runs the forecast and writes it to
   ``WORKSHOP_PRECOMPUTED`` for demonstrations, regression tests, or rapid
   analysis iteration. It is not a deployment availability path.

    python scripts/prefetch_models.py --models FCN,DLWP,PrecipitationAFNO
    python scripts/prefetch_models.py --model FCN --precompute \\
        --init-time 2026-06-15T12:00 --nsteps 4
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, "/opt/earth2/lib")

from tempo_earth2.forecast import (
    SUPPORTED_DIAGNOSTICS,
    SUPPORTED_PROGNOSTICS,
    TRANSPORT_VARIABLES,
    load_model,
    nearest_init_time,
    run_forecast,
)

PREFETCHABLE_MODELS = tuple(SUPPORTED_PROGNOSTICS) + tuple(SUPPORTED_DIAGNOSTICS)
TRANSIENT_REGISTRY_ERRORS = (
    "429 Too Many Requests",
    "maximum queue size reached",
    "502 Bad Gateway",
    "503 Service Unavailable",
    "504 Gateway Timeout",
)
RETRY_AFTER_RE = re.compile(r"Retry after (\d+) seconds", re.IGNORECASE)
MAX_CHECKPOINT_ATTEMPTS = 5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--model",
        choices=PREFETCHABLE_MODELS,
        help="one checkpoint to fetch (kept for backwards compatibility)",
    )
    selection.add_argument(
        "--models",
        default=os.environ.get("PREFETCH_MODELS"),
        help="comma-separated checkpoints to fetch",
    )
    parser.add_argument(
        "--precompute",
        action="store_true",
        help="also run and store an optional reference forecast",
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


def selected_models(args: argparse.Namespace) -> list[str]:
    """Return a validated, de-duplicated model selection."""
    requested = args.models or args.model or os.environ.get("WORKSHOP_MODEL", "FCN")
    names = [name.strip() for name in requested.split(",") if name.strip()]
    unknown = sorted(set(names) - set(PREFETCHABLE_MODELS))
    if unknown:
        raise SystemExit(
            f"unsupported model(s): {', '.join(unknown)}; "
            f"choose from {', '.join(PREFETCHABLE_MODELS)}"
        )
    return list(dict.fromkeys(names))


def _load_checkpoint_once(name: str):
    """Load one reviewed package so Earth2Studio fills its configured cache."""
    if name in SUPPORTED_PROGNOSTICS:
        return load_model(name)
    if name == "PrecipitationAFNO":
        from earth2studio.models.dx import PrecipitationAFNO

        return PrecipitationAFNO.load_model(
            PrecipitationAFNO.load_default_package()
        )
    raise AssertionError(f"validated model has no loader: {name}")


def load_checkpoint(name: str):
    """Load a checkpoint, honoring transient registry rate-limit backoff."""
    for attempt in range(1, MAX_CHECKPOINT_ATTEMPTS + 1):
        try:
            return _load_checkpoint_once(name)
        except Exception as exc:  # Remote clients wrap HTTP errors inconsistently.
            message = str(exc)
            transient = any(marker in message for marker in TRANSIENT_REGISTRY_ERRORS)
            if not transient or attempt == MAX_CHECKPOINT_ATTEMPTS:
                raise

            match = RETRY_AFTER_RE.search(message)
            delay = int(match.group(1)) + 5 if match else min(30 * 2 ** (attempt - 1), 300)
            print(
                f"Transient model-registry error while loading {name} "
                f"(attempt {attempt}/{MAX_CHECKPOINT_ATTEMPTS}); retrying in "
                f"{delay} seconds.",
                flush=True,
            )
            time.sleep(delay)

    raise AssertionError("unreachable checkpoint retry state")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    models = selected_models(args)

    cache = os.environ.get("EARTH2STUDIO_MODEL_CACHE", "<default ~/.cache/earth2studio>")
    print(f"Model cache: {cache}")
    loaded = {}
    for name in models:
        print(f"Loading {name} ...")
        loaded[name] = load_checkpoint(name)
        print(f"Loaded {name}.")

    if cache != "<default ~/.cache/earth2studio>" and Path(cache).exists():
        size = sum(f.stat().st_size for f in Path(cache).rglob("*") if f.is_file())
        print(f"Cache now holds {size / 1024**3:.2f} GB")

    if not args.precompute:
        return 0

    if len(models) != 1 or models[0] not in SUPPORTED_PROGNOSTICS:
        raise SystemExit("--precompute requires one supported prognostic model")
    model_name = models[0]

    if args.init_time:
        target = datetime.fromisoformat(args.init_time)
    else:
        raise SystemExit("--precompute requires --init-time")
    init = nearest_init_time(target).replace(tzinfo=None)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    store = output_dir / f"{model_name.lower()}_{init:%Y%m%dT%H%M}Z.zarr"

    print(f"\nRunning {model_name} from {init:%Y-%m-%d %H:%MZ} for {args.nsteps} steps ...")
    result = run_forecast(
        init,
        nsteps=args.nsteps,
        model_name=model_name,
        variables=TRANSPORT_VARIABLES,
        store_path=store,
        model=loaded[model_name],
    )

    summary = result.summary()
    print(json.dumps(summary, indent=2))
    (output_dir / f"{store.stem}.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {store}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
