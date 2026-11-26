"""HTTP routes under /h3_studio/* for Studio-specific helpers (media combo snapshots & card rename)."""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List

import folder_paths

from ..clipstream.motion_context.csrf_guard import require_same_origin

logger = logging.getLogger("h3_studio.api")

SNAPSHOT_DIR_NAME = "h3-studio-snapshots"
_SAFE_NAME_RE = re.compile(r'[\\/:*?"<>|\x00-\x1f]+')


def _snapshots_root() -> str:
    root = os.path.join(folder_paths.get_output_directory(), SNAPSHOT_DIR_NAME)
    os.makedirs(root, exist_ok=True)
    return root


def sanitize_snapshot_name(name: str) -> str:
    cleaned = _SAFE_NAME_RE.sub("_", str(name or "").strip()).strip(". ")
    if not cleaned:
        raise ValueError("快照名称不能为空")
    if len(cleaned) > 80:
        cleaned = cleaned[:80].strip(". ")
    return cleaned


def _snapshot_path(name: str) -> str:
    safe = sanitize_snapshot_name(name)
    root = os.path.realpath(_snapshots_root())
    candidate = os.path.realpath(os.path.join(root, f"{safe}.json"))
    if not candidate.startswith(root + os.sep):
        raise ValueError("非法的快照名称")
    return candidate


def list_media_snapshots() -> List[Dict[str, Any]]:
    root = _snapshots_root()
    items: List[Dict[str, Any]] = []
    for entry in sorted(os.listdir(root)):
        if not entry.endswith(".json"):
            continue
        fp = os.path.join(root, entry)
        if not os.path.isfile(fp):
            continue
        try:
            with open(fp, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            items.append({
                "name": data.get("name") or os.path.splitext(entry)[0],
                "saved_at": data.get("saved_at", ""),
                "image_project": data.get("image_project", ""),
                "audio_project": data.get("audio_project", ""),
                "image_count": len(data.get("image_files") or []),
                "audio_count": len(data.get("audio_files") or []),
                "image_files": data.get("image_files") or [],
                "audio_files": data.get("audio_files") or [],
            })
        except Exception as exc:
            logger.warning("[H3 Studio API] unreadable snapshot '%s': %s", entry, exc)
    return items


def save_media_snapshot(payload: Dict[str, Any]) -> Dict[str, Any]:
    name = sanitize_snapshot_name(payload.get("name") or "")
    path = _snapshot_path(name)
    record = {
        "schema_version": 1,
        "name": name,
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_project": str(payload.get("image_project") or ""),
        "audio_project": str(payload.get("audio_project") or ""),
        "image_files": payload.get("image_files") if isinstance(payload.get("image_files"), list) else [],
        "audio_files": payload.get("audio_files") if isinstance(payload.get("audio_files"), list) else [],
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
    return record


def delete_media_snapshot(name: str) -> bool:
    path = _snapshot_path(name)
    if os.path.isfile(path):
        os.remove(path)
        return True
    return False


def register_studio_routes() -> None:
    try:
        from aiohttp import web
        from server import PromptServer
    except ImportError:
        return
    server = getattr(PromptServer, "instance", None)
    if server is None or getattr(register_studio_routes, "_done", False):
        return

    @server.routes.get("/h3_studio/media_snapshots")
    async def _get_snapshots(request):
        del request
        try:
            return web.json_response({"snapshots": list_media_snapshots()})
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=500)

    @server.routes.post("/h3_studio/media_snapshots")
    @require_same_origin
    async def _post_snapshots(request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "无效的 JSON 请求体"}, status=400)
        action = str(body.get("action") or "save").strip().lower()
        try:
            if action == "delete":
                deleted = delete_media_snapshot(body.get("name") or "")
                return web.json_response({"deleted": deleted, "snapshots": list_media_snapshots()})
            record = save_media_snapshot(body)
            return web.json_response({"snapshot": record, "snapshots": list_media_snapshots()})
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=500)

    register_studio_routes._done = True
