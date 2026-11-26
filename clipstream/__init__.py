"""Inherited H3 ClipStream subsystem."""
from __future__ import annotations

try:
    from comfy_api.latest import ComfyExtension
    from .clipbin import NODE_LIST as CLIP_NODES
    from .motion_context import NODE_LIST as CONTEXT_NODES

    NODE_LIST = [*CONTEXT_NODES, *CLIP_NODES]

    class H3ClipStreamExtension(ComfyExtension):
        async def get_node_list(self):
            return NODE_LIST
except Exception:  # pragma: no cover - lightweight unit test import without ComfyUI
    CLIP_NODES = []
    CONTEXT_NODES = []
    NODE_LIST = []
