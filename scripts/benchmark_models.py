#!/usr/bin/env python
"""Benchmark installed Earth2Studio prognostic models on the target GPU.

This is a release-engineering command, not an attendee notebook. Run it in a
release image on the same L4 profile used by JupyterHub. It records failures,
runtime, peak allocated GPU memory, and checkpoint-cache growth as JSON. A
model is not advertised in ``model_catalog.json`` merely because it imports.

Example (inside the release container with its model extras installed):

    python /opt/earth2/scripts/benchmark_models.py \
        --model FCN DLWP --init-time 2026-05-31T12:00 --output /tmp/models.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, "/opt/earth2/lib")

from tempo_earth2.catalog import model_entries
from tempo_earth2.forecast import run_forecast


def _cache_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _load_model(name: str):
    from earth2studio.models import px

    cls = getattr(px, name)
    return cls.load_model(cls.load_default_package())


def _model_names() -> dict[str, str]:
    return {
        entry["earth2studio_class"]: entry["id"]
        for entry in model_entries(("guaranteed",))
        if entry.get("earth2studio_class") and entry["kind"].endswith("weather forecast")
    }


def parse_args() -> argparse.Namespace:
    available = _model_names()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", nargs="+", choices=sorted(available), required=True)
    parser.add_argument("--init-time", required=True, help="six-hourly ISO time")
    parser.add_argument("--nsteps", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; release validation requires the target GPU")

    cache = Path(
        os.environ.get("EARTH2STUDIO_MODEL_CACHE", Path.home() / ".cache/earth2studio")
    )
    init_time = datetime.fromisoformat(args.init_time)
    records = []
    for name in args.model:
        before = _cache_size(cache)
        started = time.perf_counter()
        record = {
            "model": name,
            "catalog_id": _model_names()[name],
            "gpu": torch.cuda.get_device_name(0),
            "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
            "nsteps": args.nsteps,
            "init_time": init_time.isoformat(),
        }
        try:
            model = _load_model(name)
            loaded = time.perf_counter()
            result = run_forecast(
                init_time,
                nsteps=args.nsteps,
                model_name=name,
                variables=("u10m", "v10m", "t2m", "msl"),
                model=model,
                verbose=False,
            )
            record.update(result.summary())
            record["status"] = "passed"
            record["load_seconds"] = round(loaded - started, 2)
        except Exception as exc:  # noqa: BLE001 - the report must preserve failures
            record["status"] = "failed"
            record["error_type"] = type(exc).__name__
            record["error"] = str(exc)
            record["traceback"] = traceback.format_exc().splitlines()[-20:]
        record["total_seconds"] = round(time.perf_counter() - started, 2)
        record["cache_growth_gb"] = round((_cache_size(cache) - before) / 1024**3, 3)
        records.append(record)
        print(json.dumps(record, indent=2))

    report = {
        "earth2studio_version": __import__("earth2studio").__version__,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")
    return 1 if any(record["status"] != "passed" for record in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
