"""H3 Studio Node 2 (H3StudioContinuityHub): 2-Tab Continuation & Video Ref Workbench.

Combines:
- Tab 1: ClipStream Dual Picker (manual card selection, delete/retry, 3 modes, lineage root tracing)
- Tab 2: Segment / Uploaded Video Ref Picker + built-in 17n+5 zero-waste padding + enable toggle
- Anti-drift 4-pillar configuration presets + expandable custom tuning parameters
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from comfy_api.latest import io

from ..clipstream.clipbin.clip_bin_manager import (
    clip_asset_kind,
    filter_clips_by_kind,
    get_project_dir,
    list_projects as list_clip_projects,
    load_project_index as load_clip_project_index,
)
from ..clipstream.clipbin.dual_manager import load_dual_clip_asset
from ..lvm.comfyui.nodes import MODE_SEGMENTS, MODE_UPLOADS
from ..lvm.comfyui.segment_store import (
    DEFAULT_PROJECT,
    list_projects as list_segment_projects,
    load_project_index as load_segment_project_index,
    load_segment,
    sanitize_project_name,
)
from ..lvm.comfyui.uploaded_video_store import resolve_upload
from ..lvm.comfyui.video_decode import (
    CUSTOM_MAX_SIDE_DEFAULT,
    DEFAULT_RESIZE,
    MAX_MAX_SIDE,
    MIN_MAX_SIDE,
    RESIZE_OPTIONS,
    TARGET_FPS,
    decode_video,
    parse_max_side,
)
from .anti_drift import compute_latent_stats
from .cache import stable_hash
from .types import (
    ANTI_DRIFT_PRESET_OPTIONS,
    PRESET_BALANCED,
    SCHEMA_VERSION,
    H3ContinuationPipeType,
    H3VideoRefBundleType,
    pad_video_to_17n_plus_5,
    resolve_anti_drift_settings,
)

logger = logging.getLogger("h3_studio.continuity_hub")

CLIP_MODES = (
    "Auto (首段全新 / 后续自动接续)",
    "Force Initial (强制首段，无上下文)",
    "Strict Chaining (严格接续，空库报错)",
)
CONTEXT_LENGTH_OPTIONS = ("22", "5", "39", "56")


def _is_initial_marker(clip_id: Optional[str]) -> bool:
    if not clip_id:
        return True
    norm = str(clip_id).strip().upper()
    return norm in ("", "[INITIAL]", "[INITIAL_GENERATION]", "INITIAL", "NONE")


def _find_dual_clips(project_name: str) -> List[Dict[str, Any]]:
    """Find dual clips in project_name from index with disk-scan fallback."""
    try:
        idx = load_clip_project_index(project_name)
        clips = filter_clips_by_kind(project_name, idx.get("clips", []), "dual")
        if clips:
            return clips
    except Exception as exc:
        logger.warning("[H3 Studio Continuity] index load failed for '%s': %s", project_name, exc)

    try:
        proj_dir = get_project_dir(project_name)
        scanned: List[Dict[str, Any]] = []
        if os.path.isdir(proj_dir):
            for entry in sorted(os.listdir(proj_dir), reverse=True):
                entry_path = os.path.join(proj_dir, entry)
                if not os.path.isdir(entry_path) or entry.startswith("."):
                    continue
                if clip_asset_kind(entry_path) == "dual":
                    meta_path = os.path.join(entry_path, "meta.json")
                    meta: Dict[str, Any] = {"clip_id": entry, "shot_tag": entry}
                    if os.path.isfile(meta_path):
                        try:
                            with open(meta_path, "r", encoding="utf-8") as handle:
                                meta = json.load(handle)
                        except Exception:
                            pass
                    scanned.append(meta)
        return scanned
    except Exception as exc:
        logger.warning("[H3 Studio Continuity] disk scan failed for '%s': %s", project_name, exc)
        return []


def _read_clip_meta(project_name: str, clip_id: str) -> Tuple[Dict[str, Any], str]:
    proj_dir = get_project_dir(project_name)
    clip_dir = os.path.join(proj_dir, clip_id)
    meta_path = os.path.join(clip_dir, "meta.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as handle:
                return json.load(handle), meta_path
        except Exception:
            pass
    return {}, meta_path


def _trace_root_lineage(
    project_name: str,
    selected_clip_id: str,
    selected_meta: Dict[str, Any],
    selected_variants: Dict[str, Any],
) -> Tuple[str, int, Optional[Dict[str, Any]]]:
    """Walk parent_clip_id to Shot #1 root card, compute chain_depth, and obtain root_stats.

    If the root card predates H3 Studio and lacks meta['latent_stats'], compute it once
    from the root card's latent_二采.safetensors (fallback 一采) and cache it back to meta.json.
    """
    visited = {selected_clip_id}
    cur_id = selected_clip_id
    cur_meta = selected_meta or {}

    # Fast path if chain_depth and root_clip_id were recorded by H3StudioDualClipSaver
    steps_up = 0
    while True:
        parent = cur_meta.get("parent_clip_id")
        if _is_initial_marker(parent) or parent in visited:
            break
        parent_meta, _ = _read_clip_meta(project_name, str(parent))
        if not parent_meta:
            break
        visited.add(str(parent))
        cur_id = str(parent)
        cur_meta = parent_meta
        steps_up += 1
        if steps_up >= 128:
            break

    root_clip_id = cur_id
    # chain_depth of the *next* clip we are about to generate:
    # if selected is Shot #1 (steps_up == 0), next clip is depth 1.
    chain_depth = steps_up + 1

    root_stats = cur_meta.get("latent_stats")
    if not isinstance(root_stats, dict) or not root_stats.get("mean") or not root_stats.get("std"):
        root_latent = None
        if root_clip_id == selected_clip_id:
            root_latent = (
                selected_variants.get("二采", {}).get("latent")
                or selected_variants.get("一采", {}).get("latent")
            )
        else:
            try:
                root_asset = load_dual_clip_asset(project_name, root_clip_id)
                rv = root_asset.get("variants", {})
                root_latent = rv.get("二采", {}).get("latent") or rv.get("一采", {}).get("latent")
            except Exception as exc:
                logger.warning("[H3 Studio Continuity] failed loading root clip '%s': %s", root_clip_id, exc)
        if root_latent is not None:
            try:
                root_stats = compute_latent_stats(root_latent, trim_frames=0)
                # Backfill cache into root card meta.json safely (non-destructive)
                root_meta_disk, root_meta_path = _read_clip_meta(project_name, root_clip_id)
                if root_meta_disk and os.path.isfile(root_meta_path):
                    root_meta_disk["latent_stats"] = root_stats
                    try:
                        with open(root_meta_path, "w", encoding="utf-8") as handle:
                            json.dump(root_meta_disk, handle, indent=2, ensure_ascii=False)
                    except OSError:
                        pass
            except Exception as exc:
                logger.warning("[H3 Studio Continuity] failed computing root_stats for '%s': %s", root_clip_id, exc)
                root_stats = None

    return root_clip_id, chain_depth, root_stats


def _clip_mtime(project_name: str, clip_id: str) -> float:
    try:
        clip_dir = os.path.join(get_project_dir(project_name), clip_id)
        mtimes = []
        for fn in ("meta.json", "latent_一采.safetensors", "latent_二采.safetensors"):
            fp = os.path.join(clip_dir, fn)
            if os.path.isfile(fp):
                mtimes.append(os.path.getmtime(fp))
        return max(mtimes) if mtimes else 0.0
    except OSError:
        return 0.0


def _resolve_clip_choice(
    project_name: str,
    clip_mode: str,
    clip_selection: str,
) -> Tuple[bool, Optional[str]]:
    """Return (is_initial, resolved_clip_id_or_None)."""
    mode_str = str(clip_mode or CLIP_MODES[0])
    if mode_str.startswith("Force Initial"):
        return True, None

    sel = str(clip_selection or "latest").strip()
    auto_latest = sel.lower() in ("latest", "", "0", "auto", "default")
    clips = _find_dual_clips(project_name)

    if not clips:
        if mode_str.startswith("Strict"):
            return False, None  # will raise during execute
        return True, None

    if auto_latest:
        return False, clips[0]["clip_id"]

    matched = [c for c in clips if sel in c.get("clip_id", "") or sel in c.get("shot_tag", "")]
    if not matched:
        return False, f"__MISSING__:{sel}"
    return False, matched[0]["clip_id"]


class H3StudioContinuityHub(io.ComfyNode):
    """2-Tab Continuation & Video Reference Hub with integrated 4-pillar anti-drift config."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        clip_projects = list_clip_projects()
        default_clip_proj = clip_projects[0] if clip_projects else "Default_Project"
        return io.Schema(
            node_id="H3StudioContinuityHub",
            display_name="H3 Studio 接续与视频工作台",
            category="H3 Studio",
            description=(
                "双 Tab 整合生成接续库 (DualPicker) 与参考视频/切片 (SegmentPicker)，"
                "内置 17n+5 零浪费补帧、首段原色锚点追踪与四柱抗漂移配置。"
                "完全保留手动选卡/删卡/三模式，不强制自动推进。"
            ),
            inputs=[
                # Tab 1: Continuation Clip Bin
                io.String.Input("clip_project", default=default_clip_proj, tooltip="生成接续项目库名"),
                io.Combo.Input("clip_mode", options=list(CLIP_MODES), default=CLIP_MODES[0], tooltip="接续模式：Auto / Force Initial / Strict Chaining"),
                io.String.Input("clip_selection", default="latest", tooltip="接续源镜头 ID 或 'latest'"),
                # Tab 2: Video Reference / Segment Picker
                io.Boolean.Input("enable_video_ref", default=False, label_on="开启视频参考", label_off="关闭", tooltip="是否启用参考视频/切片输入（替代外部 Fast Groups Bypasser）"),
                io.Boolean.Input("use_video_duration", default=False, label_on="同步参考视频时长", label_off="使用手动时长", tooltip="开启视频参考时，是否用参考视频帧数覆盖总控台的手动时长"),
                io.Combo.Input("video_source_mode", options=[MODE_SEGMENTS, MODE_UPLOADS], default=MODE_SEGMENTS, tooltip="裁切片段 / 投放完整视频"),
                io.String.Input("video_project", default=DEFAULT_PROJECT, tooltip="参考视频/切片项目库名"),
                io.Int.Input("segment_id", default=1, min=1, max=999, tooltip="当前选中的切片/投放视频编号"),
                io.Combo.Input("video_output_size", options=list(RESIZE_OPTIONS), default=DEFAULT_RESIZE, tooltip="投放视频模式的输出分辨率上限"),
                io.Int.Input("video_custom_max_side", default=CUSTOM_MAX_SIDE_DEFAULT, min=MIN_MAX_SIDE, max=MAX_MAX_SIDE, step=2, tooltip="自定义最长边像素（仅投放视频自定义尺寸时生效）"),
                io.Boolean.Input("auto_advance_segment", default=False, label_on="成功后自动下一段", label_off="手动控制 (默认)", tooltip="默认关闭！仅在手动勾选且单次生成成功完成时才自动将 segment_id + 1。"),
                # Anti-drift preset + advanced tuning
                io.Combo.Input("anti_drift_preset", options=list(ANTI_DRIFT_PRESET_OPTIONS), default=PRESET_BALANCED, tooltip="接续抗色偏/抗劣化与提速预设"),
                io.Combo.Input("pass1_context_length", options=list(CONTEXT_LENGTH_OPTIONS), default="22", advanced=True, tooltip="一采 Motion Context 帧数（仅自定义预设生效）"),
                io.Combo.Input("pass2_context_length", options=list(CONTEXT_LENGTH_OPTIONS), default="22", advanced=True, tooltip="二采 Motion Context 帧数（极速档自动为 5，自定义预设可微调）"),
                io.Int.Input("audio_context_length", default=18, min=0, max=240, advanced=True, tooltip="音频接续帧数（默认 18，0 跟随一采视频窗口）"),
                io.Boolean.Input("enable_audio_context", default=True, advanced=True, tooltip="是否开启音频接续上下文"),
                io.Float.Input("p1_taper_start", default=0.82, min=0.10, max=1.0, step=0.01, advanced=True, tooltip="算法① 一采远端旧帧置信度（仅自定义预设生效）"),
                io.Float.Input("p1_taper_seam", default=0.99, min=0.10, max=1.0, step=0.01, advanced=True, tooltip="算法① 一采接缝帧置信度（仅自定义预设生效）"),
                io.Float.Input("p2_taper_start", default=0.85, min=0.10, max=1.0, step=0.01, advanced=True, tooltip="算法① 二采远端旧帧置信度（仅自定义预设生效）"),
                io.Float.Input("p2_taper_seam", default=0.995, min=0.10, max=1.0, step=0.005, advanced=True, tooltip="算法① 二采接缝帧置信度（仅自定义预设生效）"),
                io.Int.Input("p2_release_last_steps", default=1, min=0, max=3, advanced=True, tooltip="算法② 二采最后 N 步动态释放 Motion Context（仅自定义预设生效）"),
                io.Float.Input("join_match_strength", default=1.0, min=0.0, max=1.5, step=0.05, advanced=True, tooltip="算法④ 接缝校色强度（仅自定义预设生效）"),
                io.Float.Input("root_anchor_strength", default=0.30, min=0.0, max=0.80, step=0.05, advanced=True, tooltip="算法④ 首段原色锚点拉回强度（仅自定义预设生效）"),
                io.Boolean.Input("overlap_restore_p1", default=False, advanced=True, tooltip="一采后是否恢复重叠区（经验默认关闭）"),
                io.Boolean.Input("overlap_restore_p2", default=True, advanced=True, tooltip="二采后是否恢复重叠区（经验默认开启）"),
            ],
            outputs=[
                H3ContinuationPipeType.Output(display_name="continuation_pipe"),
                H3VideoRefBundleType.Output(display_name="video_ref_bundle"),
            ],
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        clip_project="Default_Project",
        clip_mode=CLIP_MODES[0],
        clip_selection="latest",
        enable_video_ref=False,
        use_video_duration=False,
        video_source_mode=MODE_SEGMENTS,
        video_project=DEFAULT_PROJECT,
        segment_id=1,
        video_output_size=DEFAULT_RESIZE,
        video_custom_max_side=CUSTOM_MAX_SIDE_DEFAULT,
        auto_advance_segment=False,
        anti_drift_preset=PRESET_BALANCED,
        pass1_context_length="22",
        pass2_context_length="22",
        audio_context_length=18,
        enable_audio_context=True,
        p1_taper_start=0.82,
        p1_taper_seam=0.99,
        p2_taper_start=0.85,
        p2_taper_seam=0.995,
        p2_release_last_steps=1,
        join_match_strength=1.0,
        root_anchor_strength=0.30,
        overlap_restore_p1=False,
        overlap_restore_p2=True,
    ):
        del auto_advance_segment
        p_name = (clip_project or "Default_Project").strip()
        is_initial, resolved_id = _resolve_clip_choice(p_name, clip_mode, clip_selection)
        clip_mtime = _clip_mtime(p_name, resolved_id) if (not is_initial and resolved_id and not resolved_id.startswith("__MISSING__:")) else 0.0

        video_mtime = 0.0
        if enable_video_ref:
            try:
                mode = str(video_source_mode or MODE_SEGMENTS).strip().lower()
                if mode == MODE_UPLOADS:
                    path, _rec = resolve_upload(video_project, int(segment_id))
                    video_mtime = os.path.getmtime(path) if os.path.isfile(path) else 0.0
                else:
                    vproj = sanitize_project_name(video_project or DEFAULT_PROJECT)
                    idx = load_segment_project_index(vproj)
                    for seg in idx.get("segments", []):
                        if seg.get("segment_id") == int(segment_id):
                            video_mtime = float(seg.get("saved_at_ts") or len(str(seg)))
                            break
            except Exception:
                video_mtime = -1.0

        settings = resolve_anti_drift_settings(
            anti_drift_preset,
            pass1_context_length=int(pass1_context_length),
            pass2_context_length=int(pass2_context_length),
            audio_context_length=int(audio_context_length),
            enable_audio_context=bool(enable_audio_context),
            p1_taper_start=float(p1_taper_start),
            p1_taper_seam=float(p1_taper_seam),
            p2_taper_start=float(p2_taper_start),
            p2_taper_seam=float(p2_taper_seam),
            p2_release_last_steps=int(p2_release_last_steps),
            join_match_strength=float(join_match_strength),
            root_anchor_strength=float(root_anchor_strength),
            overlap_restore_p1=bool(overlap_restore_p1),
            overlap_restore_p2=bool(overlap_restore_p2),
        )
        return stable_hash({
            "clip_project": p_name,
            "clip_mode": clip_mode,
            "is_initial": is_initial,
            "resolved_id": resolved_id,
            "clip_mtime": clip_mtime,
            "enable_video_ref": bool(enable_video_ref),
            "use_video_duration": bool(use_video_duration),
            "video_source_mode": video_source_mode,
            "video_project": video_project,
            "segment_id": int(segment_id),
            "video_output_size": video_output_size,
            "video_custom_max_side": int(video_custom_max_side),
            "video_mtime": video_mtime,
            "settings": settings,
        })

    @classmethod
    def _load_video_ref(
        cls,
        *,
        enable_video_ref: bool,
        use_video_duration: bool,
        video_source_mode: str,
        video_project: str,
        segment_id: int,
        video_output_size: str,
        video_custom_max_side: int,
    ) -> Dict[str, Any]:
        if not enable_video_ref:
            return {
                "schema_version": SCHEMA_VERSION,
                "enabled": False,
                "video": None,
                "audio": None,
                "raw_frame_count": 0,
                "aligned_frame_count": 0,
                "use_video_duration": False,
                "source_meta": {
                    "source_mode": str(video_source_mode),
                    "project_name": str(video_project),
                    "segment_id": int(segment_id),
                    "context_frames": 0,
                },
                "video_fingerprint": "video_ref:off",
            }

        mode = str(video_source_mode or MODE_SEGMENTS).strip().lower()
        seg_id = int(segment_id)
        context_frames = 0
        file_mtime = 0.0

        if mode == MODE_UPLOADS:
            path, record = resolve_upload(video_project, seg_id)
            max_side = parse_max_side(video_output_size, video_custom_max_side)
            raw_video, audio, _info = decode_video(path, max_side=max_side)
            file_mtime = os.path.getmtime(path) if os.path.isfile(path) else 0.0
            proj_clean = sanitize_project_name(video_project or DEFAULT_PROJECT)
            if audio is None:
                dur = int(raw_video.shape[0]) / float(TARGET_FPS)
                audio = {"waveform": torch.zeros(1, 1, max(1, int(dur * 44100))), "sample_rate": 44100}
            source_meta = {
                "source_mode": MODE_UPLOADS,
                "project_name": proj_clean,
                "segment_id": int(record["segment_id"]),
                "original_name": record.get("original_name") or os.path.basename(path),
                "context_frames": 0,
            }
        elif mode == MODE_SEGMENTS:
            proj_clean = sanitize_project_name(video_project or DEFAULT_PROJECT)
            idx = load_segment_project_index(proj_clean)
            if idx.get("total_segments", 0) == 0:
                projects = list_segment_projects()
                raise ValueError(
                    f"H3 Studio: 视频片段库 '{proj_clean}' 为空。可用项目：{projects or '(无)'}"
                )
            available = [s.get("segment_id") for s in idx.get("segments", [])]
            if seg_id not in available:
                raise ValueError(
                    f"H3 Studio: 片段 #{seg_id} 不存在于项目 '{proj_clean}' 中。可选编号：{available}"
                )
            raw_video, audio, meta = load_segment(proj_clean, seg_id)
            context_frames = int(meta.get("context_frames") or 0)
            if audio is None:
                dur = int(raw_video.shape[0]) / float(meta.get("fps", 24))
                audio = {"waveform": torch.zeros(1, 1, max(1, int(dur * 44100))), "sample_rate": 44100}
            source_meta = {
                "source_mode": MODE_SEGMENTS,
                "project_name": proj_clean,
                "segment_id": seg_id,
                "context_frames": context_frames,
            }
        else:
            raise ValueError(f"H3 Studio: 未知的参考视频模式 {video_source_mode!r}")

        # Pitfall #12: zero-waste 17n+5 padding
        aligned_video, raw_count, aligned_count = pad_video_to_17n_plus_5(raw_video)
        video_fp = stable_hash({
            "enabled": True,
            "mode": mode,
            "project": source_meta["project_name"],
            "segment_id": source_meta["segment_id"],
            "raw_frame_count": raw_count,
            "aligned_frame_count": aligned_count,
            "shape": list(aligned_video.shape),
            "output_size": video_output_size if mode == MODE_UPLOADS else "",
            "custom_max_side": int(video_custom_max_side) if mode == MODE_UPLOADS else 0,
            "mtime": file_mtime,
        })
        logger.info(
            "[H3 Studio Continuity] loaded video ref (%s/%s #%d): %df -> %df (pad=%d, use_duration=%s)",
            mode, source_meta["project_name"], source_meta["segment_id"],
            raw_count, aligned_count, aligned_count - raw_count, bool(use_video_duration),
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "enabled": True,
            "video": aligned_video,
            "audio": audio,
            "raw_frame_count": raw_count,
            "aligned_frame_count": aligned_count,
            "use_video_duration": bool(use_video_duration),
            "source_meta": source_meta,
            "video_fingerprint": video_fp,
        }

    @classmethod
    def execute(
        cls,
        clip_project: str = "Default_Project",
        clip_mode: str = CLIP_MODES[0],
        clip_selection: str = "latest",
        enable_video_ref: bool = False,
        use_video_duration: bool = False,
        video_source_mode: str = MODE_SEGMENTS,
        video_project: str = DEFAULT_PROJECT,
        segment_id: int = 1,
        video_output_size: str = DEFAULT_RESIZE,
        video_custom_max_side: int = CUSTOM_MAX_SIDE_DEFAULT,
        auto_advance_segment: bool = False,
        anti_drift_preset: str = PRESET_BALANCED,
        pass1_context_length: str = "22",
        pass2_context_length: str = "22",
        audio_context_length: int = 18,
        enable_audio_context: bool = True,
        p1_taper_start: float = 0.82,
        p1_taper_seam: float = 0.99,
        p2_taper_start: float = 0.85,
        p2_taper_seam: float = 0.995,
        p2_release_last_steps: int = 1,
        join_match_strength: float = 1.0,
        root_anchor_strength: float = 0.30,
        overlap_restore_p1: bool = False,
        overlap_restore_p2: bool = True,
    ) -> io.NodeOutput:
        del auto_advance_segment
        p_name = (clip_project or "Default_Project").strip()
        settings = resolve_anti_drift_settings(
            anti_drift_preset,
            pass1_context_length=int(pass1_context_length),
            pass2_context_length=int(pass2_context_length),
            audio_context_length=int(audio_context_length),
            enable_audio_context=bool(enable_audio_context),
            p1_taper_start=float(p1_taper_start),
            p1_taper_seam=float(p1_taper_seam),
            p2_taper_start=float(p2_taper_start),
            p2_taper_seam=float(p2_taper_seam),
            p2_release_last_steps=int(p2_release_last_steps),
            join_match_strength=float(join_match_strength),
            root_anchor_strength=float(root_anchor_strength),
            overlap_restore_p1=bool(overlap_restore_p1),
            overlap_restore_p2=bool(overlap_restore_p2),
        )

        is_initial, resolved_id = _resolve_clip_choice(p_name, clip_mode, clip_selection)
        if not is_initial:
            if resolved_id is None:
                raise ValueError(
                    f"H3 Studio Continuity: 项目 '{p_name}' 无卡片（严格接续模式）。请切换为 Auto 模式生成首段。"
                )
            if resolved_id.startswith("__MISSING__:"):
                missing_sel = resolved_id.split(":", 1)[1]
                raise ValueError(
                    f"H3 Studio Continuity: 项目 '{p_name}' 中找不到匹配卡片 '{missing_sel}'，请重新从卡片列表选择。"
                )

        if is_initial:
            continuation_pipe: Dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "is_initial": True,
                "project_name": p_name,
                "parent_clip_id": "[INITIAL]",
                "root_clip_id": None,
                "chain_depth": 0,
                "latent_p1": None,
                "latent_p2": None,
                "root_stats": None,
                "context_fingerprint": stable_hash({"initial": True, "project": p_name, "settings": settings}),
                "settings": settings,
            }
            logger.info("[H3 Studio Continuity] initial shot in project '%s' (preset=%s)", p_name, settings["preset"])
        else:
            asset = load_dual_clip_asset(p_name, resolved_id)
            meta = asset.get("meta") or {}
            variants = asset.get("variants") or {}
            latent_p1 = variants.get("一采", {}).get("latent")
            latent_p2 = variants.get("二采", {}).get("latent")
            root_clip_id, chain_depth, root_stats = _trace_root_lineage(
                p_name, resolved_id, meta, variants
            )
            ctx_fp = stable_hash({
                "initial": False,
                "project": p_name,
                "parent_clip_id": resolved_id,
                "root_clip_id": root_clip_id,
                "chain_depth": chain_depth,
                "mtime": _clip_mtime(p_name, resolved_id),
                "settings": settings,
            })
            continuation_pipe = {
                "schema_version": SCHEMA_VERSION,
                "is_initial": False,
                "project_name": p_name,
                "parent_clip_id": resolved_id,
                "root_clip_id": root_clip_id,
                "chain_depth": chain_depth,
                "latent_p1": latent_p1,
                "latent_p2": latent_p2,
                "root_stats": root_stats,
                "context_fingerprint": ctx_fp,
                "settings": settings,
            }
            logger.info(
                "[H3 Studio Continuity] chaining from '%s' (root='%s', depth=%d, preset=%s, p1=%df/p2=%df)",
                resolved_id, root_clip_id, chain_depth, settings["preset"],
                settings["pass1_context_length"], settings["pass2_context_length"],
            )

        video_ref_bundle = cls._load_video_ref(
            enable_video_ref=bool(enable_video_ref),
            use_video_duration=bool(use_video_duration),
            video_source_mode=video_source_mode,
            video_project=video_project,
            segment_id=int(segment_id),
            video_output_size=video_output_size,
            video_custom_max_side=int(video_custom_max_side),
        )
        return io.NodeOutput(
            continuation_pipe,
            video_ref_bundle,
            ui={"h3_studio_continuity": [{"segment_id": int(segment_id), "enable_video_ref": bool(enable_video_ref)}]},
        )


NODE_LIST = [H3StudioContinuityHub]
