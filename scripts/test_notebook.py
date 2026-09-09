#!/usr/bin/env python
"""Execute the shipped introductory notebook against synthetic data.

This runs the actual ``.ipynb`` that goes into the image, cell by cell, using
the optional reference-forecast path. It needs no GPU, no Earth2Studio, and no
Earthdata login, so it can run in CI and on a laptop. This is test coverage,
not a supported attendee deployment mode.

What it cannot check is the Earth2Studio inference path itself. What it does
check is everything else: that the notebook's cells run in order, that the
grids line up, that the coordinate handling survives a real execution, and that
the advection produces the right answer on data whose answer is known.

    python scripts/test_notebook.py
    python scripts/test_notebook.py --keep   # leave the workspace for inspection
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import _synthetic
import make_teaching_data

DEFAULT_NOTEBOOK = ROOT / "notebooks" / "01_tempo_earth2_intro.ipynb"

# Wind and timing chosen so the notebook's own defaults (SCAN_INDEX=1,
# SCAN_GAP=1) land on a scan pair one hour apart, initialized at 12Z.
U_MS, V_MS, DT_HOURS = 7.5, -3.0, 1.0


def prepare(workspace: Path) -> dict[str, str]:
    tempo = workspace / "tempo" / "northeast.zarr"
    tempo.parent.mkdir(parents=True, exist_ok=True)
    _synthetic.build_tempo(tempo, u_ms=U_MS, v_ms=V_MS, dt_hours=DT_HOURS, nscans=4)

    precomputed = workspace / "precomputed"
    store = _synthetic.write_precomputed(precomputed, "FCN", u_ms=U_MS, v_ms=V_MS)
    context = make_teaching_data.build(tempo, workspace / "context")

    print(f"  TEMPO store       {tempo}")
    print(f"  precomputed       {store}")
    print(f"  context data      {context}")

    return {
        "TEMPO_DATA_URI": str(tempo),
        "WORKSHOP_DATA_URI": str(workspace),
        "WORKSHOP_PRECOMPUTED": str(precomputed),
        "WORKSHOP_CONTEXT_DATA_URI": str(context),
        "WORKSHOP_WORK_DIR": str(workspace / "work"),
        "WORKSHOP_OUTPUTS": str(workspace / "work" / "outputs"),
        "EARTH2STUDIO_MODEL_CACHE": str(workspace / "cache" / "models"),
        "EARTH2STUDIO_DATA_CACHE": str(workspace / "cache" / "data"),
        "MPLCONFIGDIR": str(workspace / "cache" / "matplotlib"),
        "MPLBACKEND": "Agg",
        # Append rather than replace: inside the container the package lives at
        # /opt/earth2/lib and PYTHONPATH already points there. Overwriting it
        # would break every import in the notebook.
        "PYTHONPATH": os.pathsep.join(
            p for p in (str(ROOT / "src"), os.environ.get("PYTHONPATH", "")) if p
        ),
    }


def patch_settings(notebook: dict) -> int | None:
    """Switch the notebook to its reference data. Returns the cell index.

    Returns ``None`` for notebooks that have no such setting, such as the
    environment check.
    """
    for index, cell in enumerate(notebook["cells"]):
        # nbformat.read rejoins source into a single string; the on-disk form is
        # a list of lines. Handle both so this works either way.
        source = cell["source"]
        if isinstance(source, list):
            source = "".join(source)
        if cell["cell_type"] == "code" and "USE_REFERENCE_FORECAST = False" in source:
            cell["source"] = source.replace(
                "USE_REFERENCE_FORECAST = False", "USE_REFERENCE_FORECAST = True"
            )
            return index
    return None


def drop_optional(notebook: dict) -> int:
    """Remove cells tagged ``optional`` before execution.

    Those cells build ipywidgets and Glue viewers. Both need a live frontend to
    mean anything, and the Glue cell reliably stalls a headless nbclient kernel
    even though its body raises ImportError in isolation. Nothing downstream of
    them is used, so a headless run is more informative without them. Use
    ``--with-optional`` to reproduce the stall.
    """
    before = len(notebook["cells"])
    notebook["cells"] = [
        cell
        for cell in notebook["cells"]
        if "optional" not in cell.get("metadata", {}).get("tags", [])
    ]
    return before - len(notebook["cells"])


def report(notebook: dict) -> int:
    """Print per-cell outcomes; return the number of cells that errored."""
    errors = 0
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                errors += 1
                print(f"\n  cell {index} raised {output['ename']}: {output['evalue']}")
                for line in output.get("traceback", [])[-12:]:
                    print(f"    {line}")
    return errors


def extract_result(notebook: dict) -> dict | None:
    """Read the intro_result.json path out of the executed notebook."""
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            text = "".join(output.get("text", []))
            for line in text.splitlines():
                if line.startswith("Wrote ") and line.endswith("intro_result.json"):
                    path = Path(line[len("Wrote "):])
                    if path.is_file():
                        return json.loads(path.read_text())
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true", help="keep the temporary workspace")
    parser.add_argument(
        "--notebook",
        type=Path,
        default=DEFAULT_NOTEBOOK,
        help="notebook to execute (default: the introductory notebook)",
    )
    parser.add_argument(
        "--timeout", type=int, default=300, help="per-cell timeout in seconds"
    )
    parser.add_argument(
        "--with-optional",
        action="store_true",
        help="also execute cells tagged 'optional' (widgets, Glue)",
    )
    args = parser.parse_args(argv)

    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError:
        print(
            "nbclient and nbformat are required:\n"
            "  uv pip install --python .venv/bin/python nbclient nbformat ipykernel",
            file=sys.stderr,
        )
        return 2

    if not args.notebook.exists():
        print(f"{args.notebook} does not exist. Run: make notebooks", file=sys.stderr)
        return 1

    workspace = Path(tempfile.mkdtemp(prefix="notebook-test-"))
    print(f"Notebook:  {args.notebook.name}")
    print(f"Workspace: {workspace}\n")

    try:
        env = prepare(workspace)
        os.environ.update(env)

        notebook = nbformat.read(args.notebook, as_version=4)
        settings_cell = patch_settings(notebook)
        if settings_cell is not None:
            print(
                f"  patched settings cell {settings_cell} "
                "to USE_REFERENCE_FORECAST = True"
            )

        if not args.with_optional:
            skipped = drop_optional(notebook)
            print(
                f"  skipped {skipped} cell(s) tagged 'optional' "
                "(widgets and Glue; --with-optional to include them)"
            )
        print()

        print("Executing notebook ...", flush=True)

        def on_cell_start(cell, cell_index, **kwargs):
            source = cell["source"]
            if isinstance(source, list):
                source = "".join(source)
            first = next(
                (line for line in source.splitlines() if line.strip()), ""
            )
            print(f"  [{cell_index:>2}] {first[:70]}", end="", flush=True)

        def on_cell_executed(cell, cell_index, **kwargs):
            print("  ok", flush=True)

        client = NotebookClient(
            notebook,
            timeout=args.timeout,
            kernel_name="python3",
            allow_errors=True,
            resources={"metadata": {"path": str(workspace)}},
            on_cell_start=on_cell_start,
            on_cell_executed=on_cell_executed,
        )
        client.execute()

        errors = report(notebook)
        if errors:
            print(f"\n{errors} cell(s) raised. See tracebacks above.")
            return 1
        print("  all cells executed without error")

        result = extract_result(notebook)
        if result is None:
            # Notebooks other than the introductory one do not produce a result
            # record; executing without error is the whole assertion for them.
            if settings_cell is None:
                print("\nPASS: notebook executed cleanly.")
                return 0
            print("\nCould not locate intro_result.json; the save cell may not have run.")
            return 1

        print("\nNotebook result:")
        print(f"  t0                 {result['t0']}")
        print(f"  t1                 {result['t1']}")
        print(f"  valid pixels       {result['valid_fraction_t0']:.1%}")
        print(f"  skill score        {result['skill']['score']:+.4f}")
        print(f"  RMSE advected      {result['skill']['rmse_advected']:.3e}")
        print(f"  RMSE persistence   {result['skill']['rmse_persistence']:.3e}")
        print(f"  scored pixels      {result['skill']['n_valid']:,}")
        for row in result["level_comparison"]:
            print(f"  level {row['level']:<5}        skill {row['skill']:+.4f}")

        # The synthetic observation was built by advecting with exactly the 10 m
        # wind in the synthetic forecast, so the notebook must recover it almost
        # perfectly. Anything less means the notebook lost the correspondence
        # somewhere between loading, regridding and scoring.
        if result["skill"]["score"] < 0.9:
            print(
                f"\nFAIL: skill {result['skill']['score']:+.4f} on data built from "
                "this exact wind field. Expected > 0.9."
            )
            return 1

        # The 100 m and 850 hPa winds in the synthetic forecast are scaled
        # versions, so they must score worse. If they do not, the notebook is
        # not really using the level it says it is.
        by_level = {row["level"]: row["skill"] for row in result["level_comparison"]}
        if "100m" in by_level and by_level["100m"] >= by_level["10m"]:
            print(
                "\nFAIL: the 100 m wind scored at least as well as the 10 m wind, "
                "but the synthetic observation was built from the 10 m wind. "
                "The wind-level selection is probably not taking effect."
            )
            return 1

        print("\nPASS: the notebook recovered the wind it was built from.")
        return 0
    finally:
        if args.keep:
            print(f"\nWorkspace kept at {workspace}")
        else:
            shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
