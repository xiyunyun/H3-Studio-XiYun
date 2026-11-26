"""Inherited H3 Long Video Manager (LVM) subsystem."""
from __future__ import annotations

try:
    from .comfyui.nodes import NODE_LIST
except Exception:  # pragma: no cover - lightweight unit test import without ComfyUI
    NODE_LIST = []
