"""ComfyUI-H3-Studio-Xiyun: Unified MiniMax H3 Studio, Long Video Manager & ClipStream.

Combines:
- All 13 legacy H3-Long-Video-Manager-Xiyun nodes + 27 /h3_lvm/* HTTP routes
- All 15 legacy ComfyUI-H3-ClipStream-Xiyun nodes + 9 /minimax/clip_bin/* & /h3_motion_context/* routes
- New H3 Studio 2+2 workspace hubs, unified encoder with 1-slot caching, fast native 3D AV upscaler,
  and integrated 4-pillar anti-drift continuation pipeline + /h3_studio/* routes.
"""
from __future__ import annotations

import logging
import os

from comfy_api.latest import ComfyExtension

from .fork_guard import find_conflicting_plugins
from .lvm.comfyui.nodes import NODE_LIST as LVM_NODES
from .lvm.comfyui.ref_presets import migrate_input_media
from .lvm.comfyui.server_api import register_h3lvm_routes
from .lvm.comfyui.video_compare import register_video_compare_routes
from .clipstream.clipbin import NODE_LIST as CLIP_NODES
from .clipstream.clipbin.clip_bin_api import register_clip_bin_routes
from .clipstream.motion_context import NODE_LIST as CONTEXT_NODES
from .clipstream.motion_context.nodes import register_chain_routes
from .studio import NODE_LIST as STUDIO_NODES, register_studio_routes

logger = logging.getLogger(__name__)

_CONFLICTS = find_conflicting_plugins(os.path.dirname(os.path.abspath(__file__)))

if _CONFLICTS:
    conflict_desc = "、".join(f"{label} ({path})" for label, path in _CONFLICTS)
    logger.warning(
        "[H3 Studio 熙云版] 检测到仍处于启用状态的旧版冲突插件：%s。"
        "为避免 28 个旧节点和 36 条 HTTP 路由冲突，H3 Studio 已跳过加载。"
        "请先将旧插件目录改名加上 .disabled 后缀（如 H3-Long-Video-Manager-Xiyun.disabled、"
        "ComfyUI-H3-ClipStream-Xiyun.disabled）；"
        "若确需强制同时加载，可设置环境变量 H3_XYIYUN_ALLOW_DUPLICATE=1（不推荐）。",
        conflict_desc,
    )

WEB_DIRECTORY = None if _CONFLICTS else "./web"

NODE_LIST = [*STUDIO_NODES, *LVM_NODES, *CONTEXT_NODES, *CLIP_NODES]


class H3StudioExtension(ComfyExtension):
    async def get_node_list(self):
        if _CONFLICTS:
            return []
        return NODE_LIST


async def comfy_entrypoint():
    if not _CONFLICTS:
        try:
            report = migrate_input_media()
            if report["entries"]:
                logger.info(
                    "[H3 Studio 熙云版] 参考素材迁移完成：%d 个项目、%d 张卡片已指向 input/%s",
                    report["projects"],
                    report["entries"],
                    "h3-lvm-refs",
                )
        except Exception:
            logger.warning("[H3 Studio 熙云版] 参考素材迁移失败（节点仍可使用）", exc_info=True)
        register_h3lvm_routes()
        register_video_compare_routes()
        register_clip_bin_routes()
        register_chain_routes()
        register_studio_routes()
    return H3StudioExtension()
