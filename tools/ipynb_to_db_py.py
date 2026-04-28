import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DB_HEADER = "# Databricks notebook source"
DB_CELL_SEP = "# COMMAND ----------"


def _as_lines(source: Any) -> List[str]:
    if source is None:
        return []
    if isinstance(source, str):
        # Keep line endings normalized; downstream will add '\n'
        return source.splitlines()
    if isinstance(source, list):
        # Jupyter allows list-of-strings; each may include '\n'
        joined = "".join(source)
        return joined.splitlines()
    raise TypeError(f"Unsupported cell source type: {type(source).__name__}")


def _write_markdown_cell(lines: List[str]) -> List[str]:
    out: List[str] = []
    out.append("# MAGIC %md")
    if not lines:
        return out
    for l in lines:
        # Databricks uses "# MAGIC" line wrapper for markdown cell content.
        out.append("# MAGIC " + l if l != "" else "# MAGIC")
    return out


def _write_magic_code_cell(lines: List[str]) -> List[str]:
    out: List[str] = []
    for l in lines:
        out.append("# MAGIC " + l if l != "" else "# MAGIC")
    return out


def _first_nonempty(lines: List[str]) -> Optional[str]:
    for l in lines:
        if l.strip() != "":
            return l
    return None


def convert_notebook(ipynb_path: Path, py_path: Path, *, overwrite: bool) -> Tuple[bool, str]:
    if py_path.exists() and not overwrite:
        return False, f"skip (exists): {py_path}"

    nb = json.loads(ipynb_path.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])
    if not isinstance(cells, list):
        raise ValueError(f"Invalid notebook (cells not a list): {ipynb_path}")

    lines_out: List[str] = [DB_HEADER, ""]

    for cell in cells:
        if not isinstance(cell, dict):
            continue

        cell_type = cell.get("cell_type")
        src_lines = _as_lines(cell.get("source"))

        lines_out.append(DB_CELL_SEP)
        lines_out.append("")

        if cell_type == "markdown":
            lines_out.extend(_write_markdown_cell(src_lines))
            lines_out.append("")
            continue

        # code (or unknown) => default to code cell
        first = _first_nonempty(src_lines)
        if first is not None and first.lstrip().startswith("%"):
            # Databricks represents magic-command cells in .py as "# MAGIC ..." lines.
            lines_out.extend(_write_magic_code_cell(src_lines))
            lines_out.append("")
        else:
            lines_out.extend(src_lines)
            lines_out.append("")

    py_path.parent.mkdir(parents=True, exist_ok=True)
    py_path.write_text("\n".join(lines_out).rstrip() + "\n", encoding="utf-8")
    return True, f"wrote: {py_path}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert .ipynb to Databricks notebook-source .py files.")
    parser.add_argument(
        "--root",
        default="src/notebooks",
        help="Root directory to search for .ipynb notebooks (default: src/notebooks)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .py files if present")
    parser.add_argument("--delete-ipynb", action="store_true", help="Delete .ipynb after successful conversion")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    ipynb_files = sorted(p for p in root.rglob("*.ipynb") if p.is_file())
    if not ipynb_files:
        print(f"No .ipynb files found under: {root}")
        return 0

    wrote = 0
    skipped = 0
    deleted = 0
    for ipynb_path in ipynb_files:
        py_path = ipynb_path.with_suffix(".py")
        did_write, msg = convert_notebook(ipynb_path, py_path, overwrite=args.overwrite)
        print(msg)
        if did_write:
            wrote += 1
            if args.delete_ipynb:
                ipynb_path.unlink()
                deleted += 1
        else:
            skipped += 1

    print(f"done. wrote={wrote} skipped={skipped} deleted_ipynb={deleted} root={root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

