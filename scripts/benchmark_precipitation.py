#!/usr/bin/env python
"""Benchmark the coupled FCN -> PrecipitationAFNO workflow on a target GPU.

PrecipitationAFNO is a diagnostic model, so loading it alone is not a useful
release test. This command runs Earth2Studio's official diagnostic workflow
from GFS initial conditions through FCN and into six-hour accumulated
precipitation, then records timing, peak allocated GPU memory, cache growth,
and output shape.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path


def _cache_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-time", required=True, help="six-hourly ISO time")
    parser.add_argument("--nsteps", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import numpy as np
    import torch
    from earth2studio import run
    from earth2studio.data import GFS
    from earth2studio.io import ZarrBackend
    from earth2studio.models.dx import PrecipitationAFNO
    from earth2studio.models.px import FCN

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; model promotion requires a GPU benchmark")

    init_time = datetime.fromisoformat(args.init_time)
    cache = Path(
        os.environ.get("EARTH2STUDIO_MODEL_CACHE", Path.home() / ".cache/earth2studio")
    )
    before = _cache_size(cache)
    started = time.perf_counter()
    record = {
        "workflow": "FCN -> PrecipitationAFNO",
        "prognostic": "FCN",
        "diagnostic": "PrecipitationAFNO",
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "nsteps": args.nsteps,
        "init_time": init_time.isoformat(),
    }
    try:
        prognostic = FCN.load_model(FCN.load_default_package())
        prognostic_loaded = time.perf_counter()
        diagnostic = PrecipitationAFNO.load_model(
            PrecipitationAFNO.load_default_package()
        )
        models_loaded = time.perf_counter()

        torch.cuda.reset_peak_memory_stats()
        io = run.diagnostic(
            [init_time],
            args.nsteps,
            prognostic,
            diagnostic,
            GFS(),
            ZarrBackend(file_name=None),
            device=torch.device("cuda"),
            verbose=False,
        )
        output = np.asarray(io["tp"])
        record.update(
            {
                "status": "passed",
                "fcn_load_seconds": round(prognostic_loaded - started, 2),
                "diagnostic_load_seconds": round(models_loaded - prognostic_loaded, 2),
                "inference_seconds": round(time.perf_counter() - models_loaded, 2),
                "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
                "output_variable": "tp",
                "output_shape": list(output.shape),
                "finite_fraction": round(float(np.isfinite(output).mean()), 6),
                "minimum_m": float(np.nanmin(output)),
                "maximum_m": float(np.nanmax(output)),
            }
        )
    except Exception as exc:  # noqa: BLE001 - preserve release-test failures
        record.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc().splitlines()[-20:],
            }
        )

    record["total_seconds"] = round(time.perf_counter() - started, 2)
    record["cache_growth_gb"] = round((_cache_size(cache) - before) / 1024**3, 3)
    report = {
        "earth2studio_version": __import__("earth2studio").__version__,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "record": record,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Wrote {args.output}")
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
