"""Strong-typed custom bus definitions, schema validators, and shared helpers for H3 Studio."""
from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import torch
from comfy_api.latest import io

SCHEMA_VERSION = 1

# Strong-typed custom ComfyUI V3 bus types (no generic '*' sockets).
H3MediaBundleType = io.Custom("H3_MEDIA_BUNDLE")
H3VideoRefBundleType = io.Custom("H3_VIDEO_REF_BUNDLE")
H3ContinuationPipeType = io.Custom("H3_CONTINUATION_PIPE")
H3EncodeEnvType = io.Custom("H3_ENCODE_ENV")
H3SamplingPlanType = io.Custom("H3_SAMPLING_PLAN")
H3CondPackType = io.Custom("H3_COND_PACK")
H3Pass1ResultType = io.Custom("H3_PASS1_RESULT")

# Anti-drift preset names (displayed in H3StudioContinuityHub combo).
PRESET_BALANCED = "均衡护色 (推荐)"
PRESET_STRONG = "强力护色"
PRESET_FAST = "极速接续 (二采5帧)"
PRESET_OFF = "关闭 (原始模式)"
PRESET_CUSTOM = "自定义"

ANTI_DRIFT_PRESET_OPTIONS = (
    PRESET_BALANCED,
    PRESET_STRONG,
    PRESET_FAST,
    PRESET_OFF,
    PRESET_CUSTOM,
)

ANTI_DRIFT_PRESETS: Dict[str, Dict[str, Any]] = {
    PRESET_BALANCED: {
        "preset": "balanced",
        "pass1_context_length": 22,
        "pass2_context_length": 22,
        "p1_taper_enabled": True,
        "p1_taper_start": 0.82,
        "p1_taper_seam": 0.99,
        "p1_taper_ramp_steps": 2,
        "p2_taper_enabled": True,
        "p2_taper_start": 0.85,
        "p2_taper_seam": 0.995,
        "p2_taper_ramp_steps": 2,
        "p2_release_last_steps": 1,
        "p2_release_keep_audio": True,
        "join_match_enabled": True,
        "join_match_strength": 1.0,
        "root_anchor_strength": 0.30,
        "overlap_restore_p1": False,
        "overlap_restore_p2": True,
    },
    PRESET_STRONG: {
        "preset": "strong",
        "pass1_context_length": 22,
        "pass2_context_length": 22,
        "p1_taper_enabled": True,
        "p1_taper_start": 0.74,
        "p1_taper_seam": 0.97,
        "p1_taper_ramp_steps": 2,
        "p2_taper_enabled": True,
        "p2_taper_start": 0.78,
        "p2_taper_seam": 0.98,
        "p2_taper_ramp_steps": 2,
        "p2_release_last_steps": 2,
        "p2_release_keep_audio": True,
        "join_match_enabled": True,
        "join_match_strength": 1.0,
        "root_anchor_strength": 0.45,
        "overlap_restore_p1": False,
        "overlap_restore_p2": True,
    },
    PRESET_FAST: {
        "preset": "fast",
        "pass1_context_length": 22,
        "pass2_context_length": 5,
        "p1_taper_enabled": True,
        "p1_taper_start": 0.82,
        "p1_taper_seam": 0.99,
        "p1_taper_ramp_steps": 2,
        "p2_taper_enabled": True,
        "p2_taper_start": 0.88,
        "p2_taper_seam": 0.995,
        "p2_taper_ramp_steps": 2,
        "p2_release_last_steps": 1,
        "p2_release_keep_audio": True,
        "join_match_enabled": True,
        "join_match_strength": 1.0,
        "root_anchor_strength": 0.30,
        "overlap_restore_p1": False,
        "overlap_restore_p2": True,
    },
    PRESET_OFF: {
        "preset": "off",
        "pass1_context_length": 22,
        "pass2_context_length": 22,
        "p1_taper_enabled": False,
        "p1_taper_start": 1.0,
        "p1_taper_seam": 1.0,
        "p1_taper_ramp_steps": 0,
        "p2_taper_enabled": False,
        "p2_taper_start": 1.0,
        "p2_taper_seam": 1.0,
        "p2_taper_ramp_steps": 0,
        "p2_release_last_steps": 0,
        "p2_release_keep_audio": True,
        "join_match_enabled": False,
        "join_match_strength": 0.0,
        "root_anchor_strength": 0.0,
        "overlap_restore_p1": False,
        "overlap_restore_p2": True,
    },
}


def resolve_anti_drift_settings(
    preset_name: str,
    *,
    pass1_context_length: int = 22,
    pass2_context_length: int = 22,
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
    p1_taper_ramp_steps: int = 2,
    p2_taper_ramp_steps: int = 2,
    p2_release_keep_audio: bool = True,
) -> Dict[str, Any]:
    """Build normalized continuation settings dict from preset or custom overrides."""
    if preset_name in ANTI_DRIFT_PRESETS:
        cfg = dict(ANTI_DRIFT_PRESETS[preset_name])
        cfg["audio_context_length"] = max(0, int(audio_context_length))
        cfg["enable_audio_context"] = bool(enable_audio_context)
        return cfg

    # Custom preset: honor every individual parameter.
    p1_start = float(p1_taper_start)
    p1_seam = float(p1_taper_seam)
    p2_start = float(p2_taper_start)
    p2_seam = float(p2_taper_seam)
    jm_strength = max(0.0, float(join_match_strength))
    return {
        "preset": "custom",
        "pass1_context_length": int(pass1_context_length),
        "pass2_context_length": int(pass2_context_length),
        "audio_context_length": max(0, int(audio_context_length)),
        "enable_audio_context": bool(enable_audio_context),
        "p1_taper_enabled": p1_start < 0.999 or p1_seam < 0.999,
        "p1_taper_start": p1_start,
        "p1_taper_seam": p1_seam,
        "p1_taper_ramp_steps": max(1, int(p1_taper_ramp_steps)),
        "p2_taper_enabled": p2_start < 0.999 or p2_seam < 0.999,
        "p2_taper_start": p2_start,
        "p2_taper_seam": p2_seam,
        "p2_taper_ramp_steps": max(1, int(p2_taper_ramp_steps)),
        "p2_release_last_steps": max(0, int(p2_release_last_steps)),
        "p2_release_keep_audio": bool(p2_release_keep_audio),
        "join_match_enabled": jm_strength > 1e-6,
        "join_match_strength": jm_strength,
        "root_anchor_strength": max(0.0, float(root_anchor_strength)),
        "overlap_restore_p1": bool(overlap_restore_p1),
        "overlap_restore_p2": bool(overlap_restore_p2),
    }


def align_frames_17n_plus_5(frame_count: int) -> int:
    """Snap frame_count up to the nearest H3 17k+5 grid point (minimum 5)."""
    n = max(5, int(frame_count))
    rem = (n - 5) % 17
    if rem == 0:
        return n
    return n + (17 - rem)


def pad_video_to_17n_plus_5(video: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    """Pad video [F, H, W, C] by repeating its last frame up to 17k+5.

    Pitfall #12 guarantee: if F is already on 17k+5 (pad == 0), return the tensor
    unchanged with ZERO duplicated frames.
    """
    if not isinstance(video, torch.Tensor) or video.ndim != 4 or video.shape[0] <= 0:
        raise ValueError("H3 Studio: expected video tensor [F, H, W, C] with F >= 1")
    raw = int(video.shape[0])
    aligned = align_frames_17n_plus_5(raw)
    pad = aligned - raw
    if pad == 0:
        return video, raw, aligned
    tail = video[-1:].expand(pad, -1, -1, -1).clone()
    return torch.cat([video, tail], dim=0), raw, aligned


def strip_filename_ext(filename: str) -> str:
    """Return basename without its final extension (e.g. '露帕-四视图.png' -> '露帕-四视图')."""
    base = os.path.basename(str(filename or "").replace("\\", "/")).strip()
    if not base:
        return ""
    stem, _ext = os.path.splitext(base)
    return stem or base


def display_card_label(entry: Dict[str, Any]) -> str:
    """Resolve card display name: custom label if non-empty, else stem of filename."""
    custom = str(entry.get("label") or "").strip()
    if custom:
        return custom
    return strip_filename_ext(str(entry.get("filename") or entry.get("original_name") or ""))


def _validate_bus(obj: Any, expected_type: str, required_keys: Tuple[str, ...]) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValueError(f"H3 Studio: expected {expected_type} dict, got {type(obj).__name__}")
    version = obj.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"H3 Studio: {expected_type} schema_version mismatch (got {version!r}, expected {SCHEMA_VERSION})"
        )
    missing = [k for k in required_keys if k not in obj]
    if missing:
        raise ValueError(f"H3 Studio: {expected_type} missing required key(s): {missing}")
    return obj


def validate_media_bundle(bundle: Any) -> Dict[str, Any]:
    return _validate_bus(bundle, "H3_MEDIA_BUNDLE", ("images", "audios", "media_fingerprint"))


def validate_video_ref_bundle(bundle: Any) -> Dict[str, Any]:
    return _validate_bus(
        bundle,
        "H3_VIDEO_REF_BUNDLE",
        ("enabled", "video", "audio", "raw_frame_count", "aligned_frame_count",
         "use_video_duration", "source_meta", "video_fingerprint"),
    )


def validate_continuation_pipe(pipe: Any) -> Dict[str, Any]:
    return _validate_bus(
        pipe,
        "H3_CONTINUATION_PIPE",
        ("is_initial", "project_name", "parent_clip_id", "root_clip_id",
         "chain_depth", "latent_p1", "latent_p2", "root_stats",
         "context_fingerprint", "settings"),
    )


def validate_encode_env(env: Any) -> Dict[str, Any]:
    return _validate_bus(
        env,
        "H3_ENCODE_ENV",
        ("clip", "video_vae", "audio_vae", "width", "height",
         "manual_aligned_frames", "prompt", "ref_image_size"),
    )


def validate_sampling_plan(plan: Any) -> Dict[str, Any]:
    return _validate_bus(
        plan,
        "H3_SAMPLING_PLAN",
        ("seed", "pass1_steps", "pass2_steps", "upscale_factor",
         "scheduler", "sampler_name", "sigma_refiner_extra_steps",
         "sigma_refiner_start", "sigma_refiner_end", "sigma_refiner_spacing"),
    )


def validate_cond_pack(pack: Any) -> Dict[str, Any]:
    return _validate_bus(
        pack,
        "H3_COND_PACK",
        ("positive", "latent_pass1", "width_p1", "height_p1",
         "frame_count", "frame_cap", "ref_image_size", "raw_ref_images",
         "video_vae", "audio_vae", "encode_fingerprint"),
    )


def validate_pass1_result(result: Any) -> Dict[str, Any]:
    return _validate_bus(
        result,
        "H3_PASS1_RESULT",
        ("latent_p1", "low_sigmas", "high_sigmas", "trim_frames_p1",
         "seed", "pass1_fingerprint"),
    )
