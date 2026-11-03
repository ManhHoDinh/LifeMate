"""Paths, sample/catalog loading and the per-split zaic_tools import.

Each split ships its own `zaic_tools` package (different apps), so one process works on one split:
call `load_zaic_tools(split)` once before creating any ToolEnv.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

DEFAULT_DATA_ROOT = Path(os.environ.get("LIFEMATE_DATA", r"D:\ZaloAI\train+public_test\train+public_test"))

_loaded_split: str | None = None


def data_root(root: str | Path | None = None) -> Path:
    return Path(root) if root else DEFAULT_DATA_ROOT


def load_zaic_tools(split: str, root: str | Path | None = None):
    """Import <root>/<split>/zaic_tools as `zaic_tools` (the package of that split)."""
    global _loaded_split
    if _loaded_split is not None and _loaded_split != split:
        raise RuntimeError(f"zaic_tools of split {_loaded_split!r} already loaded; use one process per split")
    split_dir = str(data_root(root) / split)
    if split_dir not in sys.path:
        sys.path.insert(0, split_dir)
    _loaded_split = split
    return importlib.import_module("zaic_tools")


def load_samples(split: str, root: str | Path | None = None) -> list[dict]:
    path = data_root(root) / split / f"{split}.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_catalog(split: str, root: str | Path | None = None) -> dict:
    with open(data_root(root) / split / f"{split}_tools_catalog.json", encoding="utf-8") as f:
        return json.load(f)


def catalog_index(catalog: dict) -> tuple[dict, dict]:
    """({tool_name: tool_spec}, {app_id: app_spec_without_tools})"""
    tools = {t["name"]: t for a in catalog["apps"] for t in a["tools"]}
    apps = {a["id"]: {k: v for k, v in a.items() if k != "tools"} for a in catalog["apps"]}
    return tools, apps


def fuzzy_params(tools: dict) -> dict[str, set[str]]:
    """{tool_name: {param names with x-match: fuzzy}}"""
    return {name: {k for k, p in (t.get("parameters") or {}).get("properties", {}).items() if p.get("x-match") == "fuzzy"}
            for name, t in tools.items()}


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
