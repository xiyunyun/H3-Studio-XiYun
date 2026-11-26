"""H3 Studio Node 8 (H3StudioDualClipSaver): Upgraded Dual-Variant Clip Archiver.

- 100% disk-format compatible with MiniMaxClipBinLongBuilder and MiniMaxClipBinDualPicker.
- Automatically reads project_name, parent_clip_id, root_clip_id, chain_depth from
  H3_CONTINUATION_PIPE and latent_p1 from H3_PASS1_RESULT.
- Computes & persists optional metadata fields (latent_stats, root_clip_id, chain_depth)
  into meta.json for Shot #1 color anchoring in multi-segment chains.
- Keeps video_file_p1 optional and unwired by default (fixing the legacy bug where Pass-2
  video was accidentally archived as Pass-1 video).
- Emits a frontend sync event so H3StudioContinuityHub auto-refreshes its card deck.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

import torch
from comfy_api.latest import io

from ..clipstream.clipbin.clip_bin_manager import pil_to_tensor, upsert_clip_into_index
from ..clipstream.clipbin.dual_manager import save_dual_clip_asset
from .anti_drift import compute_latent_stats
from .types import (
    H3ContinuationPipeType,
    H3Pass1ResultType,
    validate_continuation_pipe,
    validate_pass1_result,
)

logger = logging.getLogger("h3_studio.dual_saver")


class H3StudioDualClipSaver(io.ComfyNode):
    """Archive Pass-1 & Pass-2 AV latents, previews, video, and Shot #1 color anchor stats."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="H3StudioDualClipSaver",
            display_name="H3 Studio 双路归档器",
            category="H3 Studio",
            is_output_node=True,
            description=(
                "与 MiniMaxClipBinLongBuilder / DualPicker 100% 磁盘兼容的双路归档器。"
                "自动写入血缘信息与有效新帧 24 通道色彩基准 (latent_stats)，并通知工作台刷新卡片库。"
            ),
            inputs=[
                H3Pass1ResultType.Input("pass1_result"),
                io.Latent.Input("latent_p2", tooltip="二采修正并重叠恢复后的完整二采 AV Latent"),
                io.Image.Input("images_p2", tooltip="二采解码并裁切后的画面帧序列"),
                io.Audio.Input("audio_p2", tooltip="二采解码并裁切后的音频流"),
                H3ContinuationPipeType.Input("continuation_pipe", tooltip="自动读取项目库名、父镜头 ID 与血缘根 ID"),
                io.String.Input("prompt", optional=True, default="", tooltip="本段完整提示词（可接总控台 full_prompt）"),
                io.AnyType.Input("video_file_p2", optional=True, tooltip="二采 VideoCombineV2 的 Filenames 输出"),
                io.AnyType.Input("video_file_p1", optional=True, tooltip="仅当开启一采解码对比时才接入；默认留空，避免误将二采视频存成一采"),
                io.String.Input("shot_tag", optional=True, default="Auto (自动编号)", tooltip="镜头标签，填 Auto 自动递增"),
                io.Boolean.Input("save_video", optional=True, default=True, tooltip="是否归档 MP4 视频文件"),
            ],
            outputs=[
                io.String.Output(display_name="clip_id"),
                io.Image.Output(display_name="preview_image"),
                io.String.Output(display_name="project_name"),
            ],
        )

    @classmethod
    def execute(
        cls,
        pass1_result: Dict[str, Any],
        latent_p2: Dict[str, Any],
        images_p2: torch.Tensor,
        audio_p2: Dict[str, Any],
        continuation_pipe: Dict[str, Any],
        prompt: str = "",
        video_file_p2: Any = "",
        video_file_p1: Any = "",
        shot_tag: str = "Auto (自动编号)",
        save_video: bool = True,
    ) -> io.NodeOutput:
        p1 = validate_pass1_result(pass1_result)
        pipe = validate_continuation_pipe(continuation_pipe)

        project_name = str(pipe["project_name"] or "Default_Project").strip()
        parent_clip_id = str(pipe["parent_clip_id"] or "[INITIAL]")
        trim_frames_p1 = int(p1.get("trim_frames_p1") or 0)

        meta_obj, clip_dir, preview_pil = save_dual_clip_asset(
            latent_a=p1["latent_p1"],
            latent_b=latent_p2,
            images_a=None,
            images_b=images_p2,
            audio_a=None,
            audio_b=audio_p2,
            project_name=project_name,
            shot_tag=shot_tag,
            prompt=prompt if isinstance(prompt, str) else str(prompt or ""),
            parent_clip_id=parent_clip_id,
            video_file_name_a=video_file_p1,
            video_file_name_b=video_file_p2,
            save_video=bool(save_video),
        )

        # Mirror Pass-2 real keyframes onto 一采 thumbnail slots (without duplicating MP4!)
        # so legacy DualPicker first_frame/tail_frame outputs always show real frames.
        try:
            import shutil
            for prefix in ("first_frame", "tail_frame"):
                src_png = os.path.join(clip_dir, f"{prefix}_二采.png")
                dst_png = os.path.join(clip_dir, f"{prefix}_一采.png")
                if os.path.isfile(src_png):
                    shutil.copy2(src_png, dst_png)
        except OSError:
            pass

        # Compute effective new-frame stats (skipping pinned head) & lineage metadata
        root_clip_id = meta_obj.clip_id if pipe["is_initial"] else (pipe.get("root_clip_id") or parent_clip_id)
        chain_depth = 0 if pipe["is_initial"] else int(pipe.get("chain_depth", 1))
        try:
            stats = compute_latent_stats(latent_p2, trim_frames=trim_frames_p1)
        except Exception as exc:
            logger.warning("[H3 Studio DualSaver] latent_stats calculation skipped: %s", exc)
            stats = None

        # Enrich meta.json with backwards-compatible optional fields
        meta_path = os.path.join(clip_dir, "meta.json")
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as handle:
                    meta_dict = json.load(handle)
                meta_dict["root_clip_id"] = root_clip_id
                meta_dict["chain_depth"] = chain_depth
                if stats is not None:
                    meta_dict["latent_stats"] = stats
                with open(meta_path, "w", encoding="utf-8") as handle:
                    json.dump(meta_dict, handle, indent=2, ensure_ascii=False)
                upsert_clip_into_index(project_name, meta_dict)
            except Exception as exc:
                logger.warning("[H3 Studio DualSaver] failed writing studio meta fields: %s", exc)

        # Push frontend refresh notification (non-fatal when headless)
        try:
            import server
            ps = getattr(server.PromptServer, "instance", None)
            if ps is not None:
                ps.send_sync("h3_studio/clip_saved", {
                    "project": project_name,
                    "clip_id": meta_obj.clip_id,
                    "root_clip_id": root_clip_id,
                    "chain_depth": chain_depth,
                })
        except Exception:
            pass

        preview_tensor = pil_to_tensor(preview_pil)
        try:
            import folder_paths
            base_dir = folder_paths.get_output_directory()
            subfolder = os.path.relpath(clip_dir, base_dir)
        except Exception:
            subfolder = ""

        ui_images = [{
            "filename": "preview.png",
            "subfolder": subfolder,
            "type": "output",
        }]
        logger.info(
            "[H3 Studio DualSaver] archived '%s' in '%s' (root='%s', depth=%d)",
            meta_obj.clip_id, project_name, root_clip_id, chain_depth,
        )
        return io.NodeOutput(meta_obj.clip_id, preview_tensor, project_name, ui={"images": ui_images})


NODE_LIST = [H3StudioDualClipSaver]
