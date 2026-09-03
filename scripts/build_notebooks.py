#!/usr/bin/env python
"""Build .ipynb files from percent-format sources.

The notebooks live in version control as ``notebooks/src/*.py`` in jupytext
percent format, because reviewing a diff of notebook JSON is not a reasonable
thing to ask anyone to do three days before an event. This script renders them
to ``notebooks/*.ipynb``, which is what ships in the image.

Stdlib only, so it runs anywhere including inside a minimal CI job.

    python scripts/build_notebooks.py
    python scripts/build_notebooks.py --check   # fail if outputs are stale
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT / "notebooks" / "src"
OUTPUT_DIR = ROOT / "notebooks"

KERNELSPEC = {
    "display_name": "Python 3 (Earth-2 Lab)",
    "language": "python",
    "name": "python3",
}


def parse_percent(text: str, stem: str = "cell") -> list[dict]:
    """Split percent-format source into notebook cells.

    Cell markers accept jupytext-style tags:

        # %% tags=["optional"]

    The workshop uses one tag, ``optional``: cells that enrich the notebook but
    that nothing downstream depends on. ``test_notebook.py --skip-optional``
    excludes them from automated runs.
    """
    cells: list[dict] = []
    kind = "code"
    tags: list[str] = []
    buffer: list[str] = []

    def cell_id() -> str:
        # nbformat 4.5 requires a stable cell id. Deriving it from position
        # keeps the rendered .ipynb byte-identical across rebuilds, which is
        # what makes `build_notebooks.py --check` a meaningful CI gate.
        return f"{stem}-{len(cells):03d}"

    def flush() -> None:
        if not buffer:
            return
        body = "\n".join(buffer).strip("\n")
        if not body.strip():
            buffer.clear()
            return
        if kind == "markdown":
            # Strip the leading "# " that keeps the source file valid Python.
            lines = [
                line[2:] if line.startswith("# ") else ("" if line == "#" else line)
                for line in body.split("\n")
            ]
            cells.append(
                {
                    "cell_type": "markdown",
                    "id": cell_id(),
                    "metadata": {"tags": list(tags)} if tags else {},
                    "source": _as_source(lines),
                }
            )
        else:
            cells.append(
                {
                    "cell_type": "code",
                    "id": cell_id(),
                    "execution_count": None,
                    "metadata": {"tags": list(tags)} if tags else {},
                    "outputs": [],
                    "source": _as_source(body.split("\n")),
                }
            )
        buffer.clear()

    for line in text.split("\n"):
        stripped = line.rstrip()
        if stripped.startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in stripped else "code"
            tags = _parse_tags(stripped)
            continue
        buffer.append(line)
    flush()
    return cells


def _parse_tags(marker: str) -> list[str]:
    """Extract ``tags=["a", "b"]`` from a percent-format cell marker."""
    match = re.search(r'tags\s*=\s*\[([^\]]*)\]', marker)
    if not match:
        return []
    return [tag.strip().strip("\"'") for tag in match.group(1).split(",") if tag.strip()]


def _as_source(lines: list[str]) -> list[str]:
    """nbformat stores source as a list of lines, each keeping its newline."""
    while lines and not lines[-1].strip():
        lines.pop()
    return [line + "\n" for line in lines[:-1]] + lines[-1:] if lines else []


def build(path: Path) -> dict:
    return {
        "cells": parse_percent(path.read_text(), stem=path.stem),
        "metadata": {
            "kernelspec": KERNELSPEC,
            "language_info": {"name": "python", "file_extension": ".py"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if any .ipynb is missing or out of date",
    )
    args = parser.parse_args(argv)

    sources = sorted(SOURCE_DIR.glob("*.py"))
    if not sources:
        print(f"No percent sources found in {SOURCE_DIR}", file=sys.stderr)
        return 1

    stale = []
    for source in sources:
        target = OUTPUT_DIR / f"{source.stem}.ipynb"
        rendered = json.dumps(build(source), indent=1) + "\n"
        current = target.read_text() if target.exists() else None
        if current == rendered:
            print(f"  up to date  {target.relative_to(ROOT)}")
            continue
        if args.check:
            stale.append(target.relative_to(ROOT))
            continue
        target.write_text(rendered)
        print(f"  wrote       {target.relative_to(ROOT)}")

    if stale:
        print(
            "\nStale notebooks: " + ", ".join(str(p) for p in stale)
            + "\nRun: python scripts/build_notebooks.py",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
