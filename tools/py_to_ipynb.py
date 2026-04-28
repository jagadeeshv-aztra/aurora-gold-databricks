import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _split_databricks_source_to_cells(lines: List[str]) -> List[Dict[str, Any]]:
    """
    Convert Databricks 'notebook source' .py format into Jupyter cells.

    Recognizes:
    - '# COMMAND ----------' as a cell boundary
    - '# MAGIC ...' lines as Databricks magic wrapper
      - '%md' => markdown cell (the '%md' directive is removed)
      - other magics (e.g. %sql, %pip) => code cell (directive kept)
    """

    # Drop the optional header line if present.
    if lines and lines[0].lstrip().startswith("# Databricks notebook source"):
        lines = lines[1:]

    # Partition into raw cell blocks.
    blocks: List[List[str]] = []
    current: List[str] = []

    def flush():
        nonlocal current
        # Keep empty blocks out unless they have content.
        if current:
            # Strip leading/trailing empty lines for cleaner cells.
            while current and current[0].strip() == "":
                current.pop(0)
            while current and current[-1].strip() == "":
                current.pop()
            if current:
                blocks.append(current)
        current = []

    for line in lines:
        if line.startswith("# COMMAND ----------"):
            flush()
            continue
        current.append(line.rstrip("\n"))
    flush()

    cells: List[Dict[str, Any]] = []
    for block in blocks:
        # Detect any MAGIC lines.
        has_magic = any(l.lstrip().startswith("# MAGIC") for l in block)

        if not has_magic:
            source = "\n".join(block).rstrip() + "\n"
            cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": source,
                }
            )
            continue

        # Strip "# MAGIC " prefix from magic lines; keep non-magic lines as-is.
        stripped_lines: List[str] = []
        for l in block:
            s = l.lstrip()
            if s.startswith("# MAGIC"):
                # Keep original indentation before "# MAGIC" out of the cell
                # (Databricks uses MAGIC as wrapper, not meaningful indentation).
                after = s[len("# MAGIC") :].lstrip()
                stripped_lines.append(after)
            else:
                stripped_lines.append(l)

        # Determine cell type based on the first non-empty stripped line.
        first_nonempty: Optional[str] = next((x for x in stripped_lines if x.strip() != ""), None)
        if first_nonempty is None:
            # Purely empty cell: keep as empty code cell to preserve structure.
            cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": "",
                }
            )
            continue

        if first_nonempty.startswith("%md"):
            md_lines: List[str] = []
            for i, x in enumerate(stripped_lines):
                if i == 0 and x.lstrip().startswith("%md"):
                    # Remove the %md directive itself.
                    rest = x.lstrip()[len("%md") :].lstrip()
                    if rest:
                        md_lines.append(rest)
                    continue
                md_lines.append(x)
            md_source = "\n".join(md_lines).rstrip() + "\n"
            cells.append(
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": md_source,
                }
            )
            continue

        # Default: treat as code cell (including other magics like %sql/%pip).
        source = "\n".join(stripped_lines).rstrip() + "\n"
        cells.append(
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source,
            }
        )

    return cells


def _build_notebook(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def convert_file(py_path: Path, ipynb_path: Path, *, overwrite: bool) -> Tuple[bool, str]:
    if ipynb_path.exists() and not overwrite:
        return False, f"skip (exists): {ipynb_path}"

    raw = py_path.read_text(encoding="utf-8")
    cells = _split_databricks_source_to_cells(raw.splitlines(True))
    nb = _build_notebook(cells)

    ipynb_path.parent.mkdir(parents=True, exist_ok=True)
    ipynb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return True, f"wrote: {ipynb_path}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert Databricks .py notebooks to .ipynb for import.")
    parser.add_argument(
        "--root",
        default="src/notebooks",
        help="Root directory to search for .py notebooks (default: src/notebooks)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .ipynb files if present",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change without writing files",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    py_files = sorted(p for p in root.rglob("*.py") if p.is_file())
    if not py_files:
        print(f"No .py files found under: {root}")
        return 0

    wrote = 0
    skipped = 0
    for py_path in py_files:
        ipynb_path = py_path.with_suffix(".ipynb")
        if args.dry_run:
            if ipynb_path.exists() and not args.overwrite:
                print(f"skip (exists): {ipynb_path}")
                skipped += 1
            else:
                print(f"would write: {ipynb_path}")
                wrote += 1
            continue

        did_write, msg = convert_file(py_path, ipynb_path, overwrite=args.overwrite)
        print(msg)
        if did_write:
            wrote += 1
        else:
            skipped += 1

    print(f"done. wrote={wrote} skipped={skipped} root={root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

