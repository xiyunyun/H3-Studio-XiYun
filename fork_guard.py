"""Detect conflicting copies of H3 LVM, H3 ClipStream, or H3 Studio in custom_nodes.

ComfyUI registers nodes by node_id and serves HTTP routes globally, so running
ComfyUI-H3-Studio-Xiyun alongside an active legacy H3-Long-Video-Manager or
ComfyUI-H3-ClipStream plugin would cause duplicate routes and overwritten nodes.
This guard scans sibling folders (ignoring *.disabled) and reports every active
conflict with a clear remediation hint.
"""
from __future__ import annotations

import os
from typing import List, Tuple

CONFLICT_MARKERS: Tuple[Tuple[str, str, str], ...] = (
    (
        "H3 Long Video Manager",
        os.path.join("comfyui", "nodes.py"),
        "node_id='H3 Long Video Manager'",
    ),
    (
        "ComfyUI-H3-ClipStream",
        os.path.join("clipbin", "long_nodes.py"),
        "node_id='MiniMaxClipBinLongBuilder'",
    ),
    (
        "ComfyUI-H3-Studio-Xiyun",
        os.path.join("lvm", "comfyui", "nodes.py"),
        "node_id='H3 Long Video Manager'",
    ),
)


def find_other_copy(
    plugin_dir: str,
    marker_rel: str,
    marker_text: str,
    allow_env: str = "H3_XYIYUN_ALLOW_DUPLICATE",
) -> str | None:
    """Return the path of another copy matching (marker_rel, marker_text), or None."""
    if os.environ.get(allow_env) == "1":
        return None

    self_dir = os.path.abspath(plugin_dir)
    custom_nodes = os.path.dirname(self_dir)
    try:
        entries = list(os.scandir(custom_nodes))
    except OSError:
        return None

    for entry in entries:
        if not entry.is_dir():
            continue
        if entry.name == "__pycache__" or entry.name.startswith("."):
            continue
        if entry.name.endswith(".disabled"):
            continue
        other_dir = os.path.abspath(entry.path)
        if other_dir == self_dir:
            continue
        marker = os.path.join(other_dir, marker_rel)
        if not os.path.isfile(marker):
            continue
        try:
            with open(marker, "r", encoding="utf-8", errors="ignore") as stream:
                head = stream.read(200_000)
        except OSError:
            continue
        if marker_text in head:
            return other_dir
    return None


def find_conflicting_plugins(
    plugin_dir: str,
    allow_env: str = "H3_XYIYUN_ALLOW_DUPLICATE",
) -> List[Tuple[str, str]]:
    """Return list of (label, other_dir) for all active conflicting plugins."""
    if os.environ.get(allow_env) == "1":
        return []
    found: List[Tuple[str, str]] = []
    seen_dirs: set[str] = set()
    for label, marker_rel, marker_text in CONFLICT_MARKERS:
        other = find_other_copy(plugin_dir, marker_rel, marker_text, allow_env=allow_env)
        if other and other not in seen_dirs:
            seen_dirs.add(other)
            found.append((label, other))
    return found
