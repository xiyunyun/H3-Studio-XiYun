"""Four-pillar anti-drift & acceleration algorithms for H3 Studio continuation.

Pillar 1 (Algorithm ①): Flow-matched progressive confidence-declared noise tapering
  - Blends Motion Context video keyframes with a_k * z + (1 - a_k) * eps
  - Synchronously installs the single shared h3_strength_patch so DiT AdaLN relabels
    each cond segment's timestep to max(t_v, a_k) (Pitfall #4). Never touches audio.

Pillar 2 (Algorithm ②): Pass-2 tail dynamic Motion Context release
  - Computes exact midpoint sigma threshold between keep and drop steps of low_sigmas
  - Tags ONLY Motion Context video keyframes with MC_RELEASE_KEY (isolated from Guide
    Timeline's RELEASE_KEY, refs, and audio — Pitfall #7)
  - Installs instance-local DIFFUSION_MODEL wrapper + _window_payload metadata preserver
    that sets layout=None and rebuilds cond_video_latents on release.

Pillar 3 (Algorithm ③): Asymmetric Pass-1 / Pass-2 context window (22 -> 5 alignment)
  - Anchors Pass-2 shorter context at frame_offset = trim_p1 - trim_p2 (e.g. 22 - 5 = 17)
  - Keeps audio context end_frame locked at trim_p1 (e.g. 22) for 0ms audio alignment
  - Keeps OverlapRestore & Trim locked to trim_p1 (Pitfall #6).

Pillar 4 (Algorithm ④): Root-Anchor Join Match + OverlapRestore synergy
  - Strict execution order (Pitfall #8):
      Pass-2 Sampler output -> Root-Anchor Join Match (vs raw un-noised parent + root_stats)
      -> OverlapRestore (vs raw un-noised parent, trim_p1) -> VAEDecode & DualSaver.
  - Scene-Shift Guard suppresses root-anchor pull when motion > 0.75 or rel_shift > 0.60.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..clipstream.motion_context.join_match import (
    MiniMaxH3JoinMatchClipStream,
    as5,
    blur,
    repack,
    rms,
    shot_motion,
    steps_for_frames,
    streams_of,
)
from ..clipstream.motion_context.nodes import (
    AUDIO_HZ,
    FPS,
    FRAME_RESCALE,
    _audio_tail_from_latent,
    _ensure_layout_ok,
    _pixel_frames,
    _steps_for_frames,
    _video_from_latent,
    _video_tail_from_latent,
)
from ..clipstream.motion_context.overlap_restore import restore_overlap
from ..lvm.comfyui.guide_release import condition_latents
from ..lvm.comfyui.h3_strength_patch import (
    NOISE_AUG_KEY,
    flow_blend,
    install as install_strength_patch,
    install_reason,
)
from ..lvm.core.guide_release import RELEASE_KEY as GUIDE_RELEASE_KEY
from .cache import clone_conditioning_safe

logger = logging.getLogger("h3_studio.anti_drift")

MC_RELEASE_KEY = "h3_mc_release_sigma"
MC_WRAPPER_KEY = "h3_studio_mc_release"
MC_WINDOW_MARKER = "_h3_studio_mc_native_window"
MC_TAPER_SEED_BASE = 91700
DEFAULT_VISUAL_COND_TIMESTEP = 0.999


def _visual_cond_timestep() -> float:
    try:
        from comfy.ldm.minimax.model import VISUAL_COND_TIMESTEP
        return float(VISUAL_COND_TIMESTEP)
    except Exception:
        return DEFAULT_VISUAL_COND_TIMESTEP


# ---------------------------------------------------------------------------
# Algorithm ①: Flow-Matched Progressive Confidence-Declared Noise Tapering
# ---------------------------------------------------------------------------

def compute_taper_schedule(
    num_steps: int,
    taper_start: float,
    taper_seam: float,
    ramp_steps: int = 2,
) -> List[float]:
    """Compute per-step confidence schedule s_0 .. s_{K-1} (oldest -> seam)."""
    k_total = int(num_steps)
    if k_total <= 0:
        return []
    start = max(0.05, min(1.0, float(taper_start)))
    seam = max(start, min(1.0, float(taper_seam)))
    ramp = max(0, int(ramp_steps))
    if k_total == 1:
        return [seam]
    schedule: List[float] = []
    for k in range(k_total):
        from_end = (k_total - 1) - k
        if ramp <= 0:
            s_k = seam if from_end == 0 else start
        elif from_end >= ramp:
            s_k = start
        else:
            frac = (ramp - from_end) / float(ramp)
            s_k = start + (seam - start) * frac
        schedule.append(round(s_k, 6))
    return schedule


def apply_progressive_taper(
    video_keyframes: List[Dict[str, Any]],
    taper_start: float,
    taper_seam: float,
    ramp_steps: int = 2,
    seed_base: int = MC_TAPER_SEED_BASE,
    install_patch: bool = True,
) -> List[float]:
    """Apply flow-matched noise + declared noise_aug to Motion Context video keyframes only.

    Returns the list of applied coefficients (or 1.0 when untouched).
    """
    if not video_keyframes:
        return []
    schedule = compute_taper_schedule(len(video_keyframes), taper_start, taper_seam, ramp_steps)
    need_patch = any(s < 0.999 for s in schedule)
    if need_patch and install_patch:
        if not install_strength_patch():
            raise RuntimeError(
                "H3 Studio 算法① (渐进声明加噪) 需要启用逐行时间步重标定补丁，"
                f"但当前 ComfyUI 拒绝安装：{install_reason()}"
            )

    vct = _visual_cond_timestep()
    applied: List[float] = []
    for k, (kf, s_k) in enumerate(zip(video_keyframes, schedule)):
        if kf.get("latent") is None or s_k >= 0.999:
            applied.append(1.0)
            continue
        coeff = round(float(s_k) * vct, 6)
        kf[NOISE_AUG_KEY] = coeff
        kf["latent"] = flow_blend(kf["latent"], coeff, int(seed_base) + k)
        applied.append(coeff)
    return applied


# ---------------------------------------------------------------------------
# Algorithm ②: Pass-2 Tail Dynamic Motion Context Release
# ---------------------------------------------------------------------------

def compute_mc_release_threshold(
    low_sigmas: Any,
    release_last_steps: int,
) -> Optional[float]:
    """Compute exact midpoint sigma threshold to release MC during the last N steps."""
    r = int(release_last_steps)
    if r <= 0:
        return None
    if isinstance(low_sigmas, torch.Tensor):
        vals = [float(x) for x in low_sigmas.detach().cpu().tolist()]
    else:
        vals = [float(x) for x in low_sigmas]
    nonzero = [s for s in vals if s > 1e-6]
    if len(nonzero) <= 1:
        return None
    if r >= len(nonzero):
        logger.warning(
            "[H3 Studio] p2_release_last_steps=%d >= active pass-2 steps %d; clamping to %d",
            r, len(nonzero), len(nonzero) - 1,
        )
        r = len(nonzero) - 1
    if r <= 0:
        return None
    s_keep = nonzero[-r - 1]
    s_drop = nonzero[-r]
    return 0.5 * (s_keep + s_drop)


def tag_mc_release(video_keyframes: List[Dict[str, Any]], threshold: Optional[float]) -> None:
    """Attach MC_RELEASE_KEY exclusively to Motion Context video keyframes."""
    if threshold is None:
        return
    t_val = float(threshold)
    if not math.isfinite(t_val) or not (0.0 <= t_val <= 1.0):
        raise ValueError(f"H3 Studio: invalid MC release sigma threshold {threshold!r}")
    for kf in video_keyframes:
        if kf.get("latent") is not None:
            kf[MC_RELEASE_KEY] = t_val


def release_mc_payload(
    payload: Optional[Dict[str, Any]],
    sigma: float,
    keep_audio: bool = True,
) -> Optional[Dict[str, Any]]:
    """Pure filter: remove ONLY MC video keyframes whose MC_RELEASE_KEY > sigma.

    Guarantees (Pitfall #7):
    - Never touches payload['refs'] (all reference images / reference videos / standalone audio)
    - Never touches Guide Timeline keyframes (which lack MC_RELEASE_KEY)
    - Never touches audio_latent when keep_audio=True
    - Sets layout=None and rebuilds cond_video_latents / cond_audio_latents on any change
    """
    if not payload or not payload.get("keyframes"):
        return payload
    retained = []
    changed = False
    for kf in payload["keyframes"]:
        threshold = kf.get(MC_RELEASE_KEY)
        if threshold is not None and kf.get("latent") is not None:
            t_val = float(threshold)
            if not math.isfinite(t_val) or not (0.0 <= t_val <= 1.0):
                raise ValueError(f"H3 Studio: invalid MC release threshold {threshold!r}")
            if float(sigma) < t_val:
                changed = True
                if keep_audio and kf.get("audio_latent") is not None:
                    retained.append({
                        k: v for k, v in kf.items()
                        if k not in ("latent", MC_RELEASE_KEY, NOISE_AUG_KEY)
                    })
                continue
        retained.append(kf)
    if not changed:
        return payload
    result = dict(payload, keyframes=retained, layout=None)
    result.update(condition_latents(retained, payload.get("refs")))
    return result


def preserve_mc_window_metadata(native_window):
    """Wrap _window_payload so MC_RELEASE_KEY (plus guide RELEASE_KEY & noise_aug) survives slicing."""
    base_window = getattr(native_window, MC_WINDOW_MARKER, native_window)

    def window_payload(payload, window):
        keyframes = payload.get("keyframes") or []
        if not any(MC_RELEASE_KEY in k for k in keyframes):
            return base_window(payload, window)
        retained = []
        result = None
        for original in keyframes:
            result = base_window(dict(payload, keyframes=[original]), window)
            for cropped in result.get("keyframes") or []:
                if cropped.get("latent") is not None:
                    cropped = dict(cropped)
                    for meta_key in (MC_RELEASE_KEY, GUIDE_RELEASE_KEY, NOISE_AUG_KEY):
                        if meta_key in original and meta_key not in cropped:
                            cropped[meta_key] = original[meta_key]
                retained.append(cropped)
        if result is None:
            return base_window(payload, window)
        result = dict(result, keyframes=retained, layout=None)
        result.update(condition_latents(retained, payload.get("refs")))
        return result

    setattr(window_payload, MC_WINDOW_MARKER, base_window)
    return window_payload


def patch_model_mc_release(model: Any, keep_audio: bool = True, log_forwards: bool = False) -> Any:
    """Clone model and attach instance-local DIFFUSION_MODEL release wrapper + window preserver."""
    import comfy.patcher_extension

    try:
        native_window = model.get_model_object("_window_payload")
    except (AttributeError, KeyError):
        native_window = None

    def mc_release_wrapper(executor, x, timestep, context, transformer_options, minimax_payload=None, **kwargs):
        sigma = float(timestep.flatten()[0]) / 1000.0
        filtered = release_mc_payload(minimax_payload, sigma, keep_audio=keep_audio)
        if log_forwards and any(MC_RELEASE_KEY in k for k in (minimax_payload or {}).get("keyframes", [])):
            before = sum(k.get("latent") is not None for k in minimax_payload["keyframes"])
            after = sum(k.get("latent") is not None for k in filtered["keyframes"])
            logger.info("[H3 Studio MC Release] sigma=%.6f video_keyframes=%d->%d", sigma, before, after)
        return executor(x, timestep, context, transformer_options, minimax_payload=filtered, **kwargs)

    patched = model.clone()
    wrapper_type = comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL
    patched.remove_wrappers_with_key(wrapper_type, MC_WRAPPER_KEY)
    patched.add_wrapper_with_key(wrapper_type, MC_WRAPPER_KEY, mc_release_wrapper)
    if callable(native_window):
        patched.add_object_patch("_window_payload", preserve_mc_window_metadata(native_window))
    return patched


# ---------------------------------------------------------------------------
# Algorithm ③: Asymmetric Pass-1 / Pass-2 Motion Context Injection
# ---------------------------------------------------------------------------

def inject_motion_context_studio(
    conditioning: Any,
    target_latent: Dict[str, Any],
    context_latent: Optional[Dict[str, Any]],
    *,
    context_length: int = 22,
    trim_frames_p1: Optional[int] = None,
    audio_context_length: int = 18,
    enable_audio_context: bool = True,
    taper_enabled: bool = False,
    taper_start: float = 0.82,
    taper_seam: float = 0.99,
    taper_ramp_steps: int = 2,
    taper_seed_base: int = MC_TAPER_SEED_BASE,
    release_threshold: Optional[float] = None,
    verify_layout: bool = True,
    install_patch: bool = True,
) -> Tuple[list, int, Dict[str, Any]]:
    """Inject Motion Context with full support for ① Taper, ② Release tag, and ③ 22->5 offset.

    Args:
        conditioning: Incoming conditioning list (never mutated in place).
        target_latent: Target AV latent for this pass.
        context_latent: Previous clip's raw un-noised AV latent for this pass.
        context_length: Desired video context window for this pass (5 / 22 / 39 / 56).
        trim_frames_p1: Authoritative head span from Pass 1 (None in Pass 1 -> equals context_length).
            When Pass 2 uses a shorter window (e.g. context_length=5, trim_frames_p1=22):
            - Video keyframes are anchored at frame_offset = trim_frames_p1 - context_length (17).
            - Audio end_frame stays locked to trim_frames_p1 (22) for 0ms audio alignment.
        audio_context_length: Frames of tail audio to pin (0 follows trim_frames_p1).
        enable_audio_context: Whether to pin tail audio on timeline.

    Returns:
        (new_conditioning, authoritative_trim_frames, info_dict)
    """
    if context_latent is None:
        return clone_conditioning_safe(conditioning), 0, {"active": False}

    if verify_layout:
        _ensure_layout_ok()

    target_video = _video_from_latent(target_latent)
    target_t = int(target_video.shape[2])
    target_w = int(target_video.shape[4]) * 16
    target_h = int(target_video.shape[3]) * 16
    frame_count = _pixel_frames(target_t)

    src_video = _video_from_latent(context_latent)
    src_w = int(src_video.shape[4]) * 16
    src_h = int(src_video.shape[3]) * 16
    if src_w != target_w or src_h != target_h:
        raise ValueError(
            f"H3 Studio Motion Context: context_latent is {src_w}x{src_h} but target is {target_w}x{target_h}."
        )
    if int(src_video.shape[1]) != int(target_video.shape[1]):
        raise ValueError(
            f"H3 Studio Motion Context: channel mismatch ({src_video.shape[1]} vs {target_video.shape[1]})."
        )

    available = _pixel_frames(int(src_video.shape[2]))
    req_len = int(context_length)
    if trim_frames_p1 is not None and int(trim_frames_p1) > 0:
        head_span = int(trim_frames_p1)
        if req_len > head_span:
            logger.warning(
                "[H3 Studio] pass2_context_length (%d) > pass1 trim_frames (%d); clamping to %d",
                req_len, head_span, head_span,
            )
            req_len = head_span
    else:
        head_span = req_len

    # Clamp to available frames on valid step grid
    valid_grid = (56, 39, 22, 5)
    n = min(req_len, available)
    if n not in valid_grid:
        snapped = next((g for g in valid_grid if g <= n), 5)
        logger.warning("[H3 Studio] context_length %d snapped to grid %d", n, snapped)
        n = snapped

    if trim_frames_p1 is None:
        head_span = n

    if head_span >= frame_count:
        raise ValueError(
            f"H3 Studio Motion Context: pinned head {head_span} frames >= target clip {frame_count} frames."
        )

    # Pitfall #6 invariant: frame_offset = trim_p1 - trim_p2 (e.g. 22 - 5 = 17)
    frame_offset = head_span - n
    if frame_offset < 0:
        raise ValueError(
            f"H3 Studio Motion Context: negative frame_offset ({frame_offset}); "
            f"trim_p1={head_span}, context_length={n}"
        )
    if frame_offset > 0:
        steps_p1 = _steps_for_frames(head_span)
        steps_p2 = _steps_for_frames(n)
        if steps_p1 is None or steps_p2 is None or (steps_p1 - steps_p2) % 5 != 0:
            raise ValueError(
                f"H3 Studio Motion Context: asymmetric pair (p1={head_span}, p2={n}) does not align on 5-step cycle."
            )

    blocks, local_offsets, covered = _video_tail_from_latent(context_latent, n)
    if covered != n:
        raise RuntimeError(f"H3 Studio Motion Context: expected {n} covered frames, got {covered}")

    video_keyframes: List[Dict[str, Any]] = []
    for local_off, blk in zip(local_offsets, blocks):
        video_keyframes.append({
            "resolved_frame_index": int(frame_offset + local_off),
            "latent": blk,
        })

    # Algorithm ①: progressive flow-matched confidence tapering (video only)
    taper_coeffs: List[float] = []
    if taper_enabled:
        taper_coeffs = apply_progressive_taper(
            video_keyframes,
            taper_start=taper_start,
            taper_seam=taper_seam,
            ramp_steps=taper_ramp_steps,
            seed_base=taper_seed_base,
            install_patch=install_patch,
        )

    # Algorithm ②: tag MC video keyframes for tail dynamic release (video only)
    if release_threshold is not None:
        tag_mc_release(video_keyframes, release_threshold)

    # Audio context: ALWAYS end-aligned at head_span (trim_p1) for 0ms alignment!
    audio_kf = None
    audio_end_frame = None
    ref_audio_t = 0
    if enable_audio_context:
        a_frames = int(audio_context_length) or head_span
        audio_latent, ref_audio_t, overhang = _audio_tail_from_latent(context_latent, a_frames)
        end_frame = float(head_span) + overhang / FRAME_RESCALE
        end_coord = round(FRAME_RESCALE * end_frame)
        end_frame = end_coord / FRAME_RESCALE
        audio_kf = {
            "resolved_frame_index": end_frame - ref_audio_t / FRAME_RESCALE,
            "audio_latent": audio_latent,
        }
        audio_end_frame = end_frame

    # Merge into cloned conditioning without mutating cached input
    base_cond = clone_conditioning_safe(conditioning)
    tail_kfs = [audio_kf] if audio_kf is not None else []
    dropped = []
    out_cond = []
    for emb, extra in base_cond:
        prior = extra.get("minimax_keyframes") or []
        kept = []
        for kf in prior:
            p = kf.get("resolved_frame_index", 0)
            if p >= frame_count:
                raise ValueError(
                    f"H3 Studio Motion Context: upstream keyframe at frame {p} exceeds clip length {frame_count}."
                )
            if p < head_span:
                dropped.append(p)
                continue
            kept.append(dict(kf))
        extra["minimax_keyframes"] = kept + [dict(k) for k in video_keyframes] + [dict(k) for k in tail_kfs]
        out_cond.append([emb, extra])

    if dropped:
        logger.warning(
            "[H3 Studio] dropped %d upstream keyframe(s) inside pinned head 0..%d: %s",
            len(dropped), head_span - 1, sorted(set(dropped)),
        )

    info = {
        "active": True,
        "context_length": n,
        "trim_frames_p1": head_span,
        "frame_offset": frame_offset,
        "video_indices": [kf["resolved_frame_index"] for kf in video_keyframes],
        "taper_coeffs": taper_coeffs,
        "release_threshold": release_threshold,
        "audio_end_frame": audio_end_frame,
        "audio_steps": ref_audio_t,
    }
    return out_cond, head_span, info


# ---------------------------------------------------------------------------
# Algorithm ④: Root-Anchor Join Match + OverlapRestore Synergy
# ---------------------------------------------------------------------------

def compute_latent_stats(latent: Dict[str, Any], trim_frames: int = 0) -> Dict[str, Any]:
    """Compute 24-channel mean, std, and 3x3 detail RMS on effective new frames of a video latent."""
    video, _ = streams_of(latent)
    v5 = as5(video).detach().to(device="cpu", dtype=torch.float32)
    total_steps = int(v5.shape[2])
    skip_steps = 0
    if int(trim_frames) > 0:
        s = steps_for_frames(int(trim_frames))
        if s is not None and s < total_steps:
            skip_steps = s
    region = v5[:, :, skip_steps:]
    if region.shape[2] == 0:
        region = v5

    # Per-channel mean and std over (B, T, H, W)
    mean = region.mean(dim=(0, 2, 3, 4))
    std = region.std(dim=(0, 2, 3, 4))
    smooth = blur(region)
    fine = region - smooth
    detail_rms = (fine * fine).mean(dim=(0, 2, 3, 4)).sqrt()

    return {
        "mean": [round(float(x), 6) for x in mean.tolist()],
        "std": [round(float(x), 6) for x in std.tolist()],
        "detail_rms": [round(float(x), 6) for x in detail_rms.tolist()],
        "steps_measured": int(region.shape[2]),
        "skip_steps": int(skip_steps),
    }


def apply_root_anchor_pull(
    samples: Dict[str, Any],
    root_stats: Optional[Dict[str, Any]],
    *,
    trim_frames: int,
    chain_depth: int,
    root_anchor_strength: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Gently pull new frames' 24-channel mean/std toward Shot #1 root_stats with Scene-Shift Guard."""
    diag: Dict[str, Any] = {
        "applied": False,
        "reason": "disabled",
        "eff_weight": 0.0,
        "motion": None,
        "rel_shift": None,
    }
    strength = float(root_anchor_strength)
    if not root_stats or int(chain_depth) < 1 or strength <= 1e-6:
        return samples, diag

    root_mean_list = root_stats.get("mean")
    root_std_list = root_stats.get("std")
    if not root_mean_list or not root_std_list:
        diag["reason"] = "missing_root_stats"
        return samples, diag

    video, audio = streams_of(samples)
    raw_ndim = video.ndim
    orig_dtype = video.dtype
    b = as5(video).to(dtype=torch.float32)
    channels = int(b.shape[1])
    if len(root_mean_list) != channels or len(root_std_list) != channels:
        diag["reason"] = "channel_mismatch"
        return samples, diag

    steps = steps_for_frames(int(trim_frames)) if int(trim_frames) > 0 else 0
    if steps is None or steps >= int(b.shape[2]):
        steps = 0

    new_region = b[:, :, steps:]
    m_cur = new_region.mean(dim=(0, 2, 3, 4))
    s_cur = new_region.std(dim=(0, 2, 3, 4))
    m_root = torch.tensor(root_mean_list, device=b.device, dtype=torch.float32)
    s_root = torch.tensor(root_std_list, device=b.device, dtype=torch.float32)

    motion = shot_motion(b, steps)
    rel_shift = rms(m_cur - m_root) / max(rms(s_root), 1e-6)
    diag["motion"] = None if motion is None else round(float(motion), 4)
    diag["rel_shift"] = round(float(rel_shift), 4)

    # Scene-Shift Guard: linearly attenuate to 0 on large camera moves or deliberate lighting shifts.
    if motion is None or motion <= 0.48:
        g_motion = 1.0
    elif motion >= 0.75:
        g_motion = 0.0
    else:
        g_motion = (0.75 - float(motion)) / (0.75 - 0.48)

    if rel_shift <= 0.30:
        g_shift = 1.0
    elif rel_shift >= 0.60:
        g_shift = 0.0
    else:
        g_shift = (0.60 - float(rel_shift)) / (0.60 - 0.30)

    eff_w = max(0.0, min(1.0, strength * min(g_motion, g_shift)))
    diag["eff_weight"] = round(eff_w, 4)

    if eff_w <= 1e-6:
        diag["reason"] = "scene_shift_guard"
        return samples, diag

    # Bounded gentle std correction [0.94, 1.06] and bounded mean pull.
    std_gain = (1.0 + eff_w * (s_root / s_cur.clamp(min=1e-6) - 1.0)).clamp(0.94, 1.06)
    max_mean_step = 0.25 * s_root.clamp(min=1e-3)
    mean_delta = (eff_w * (m_root - m_cur)).clamp(-max_mean_step, max_mean_step)

    view = lambda v: v.view(1, -1, 1, 1, 1)
    out = (b - view(m_cur)) * view(std_gain) + view(m_cur) + view(mean_delta)
    out = out.to(dtype=orig_dtype)
    if raw_ndim == 4:
        out = out.squeeze(0)

    result = dict(samples)
    result["samples"] = repack(samples["samples"], out, audio)
    diag["applied"] = True
    diag["reason"] = "applied"
    diag["mean_delta_rms"] = round(rms(mean_delta), 5)
    diag["std_gain_range"] = [round(float(std_gain.min()), 4), round(float(std_gain.max()), 4)]
    return result, diag


def apply_pass2_anti_drift_postprocess(
    samples: Dict[str, Any],
    *,
    context_latent: Optional[Dict[str, Any]],
    trim_frames_p1: int,
    root_stats: Optional[Dict[str, Any]] = None,
    chain_depth: int = 0,
    join_match_enabled: bool = True,
    join_match_strength: float = 1.0,
    root_anchor_strength: float = 0.30,
    overlap_restore_p2: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Execute strict Pitfall #8 post-processing pipeline on Pass-2 sampler output:

      1. JoinMatch (against raw un-noised context_latent)
      2. Root-Anchor gentle pull (against Shot #1 root_stats, with Scene-Shift Guard)
      3. OverlapRestore (restores trim_frames_p1 head tokens from raw context_latent)

    Returns (final_latent_p2, diagnostics_dict). Audio is NEVER modified.
    """
    steps_order: List[str] = []
    cur = samples
    diag: Dict[str, Any] = {
        "order": steps_order,
        "join_match_applied": False,
        "root_anchor": {"applied": False},
        "overlap_restore_applied": False,
    }

    if context_latent is None or int(trim_frames_p1) <= 0:
        return cur, diag

    # Step 1: Join Match (eliminates local seam step in fine detail, mean, spread)
    if join_match_enabled and abs(float(join_match_strength)) > 1e-6:
        out_jm = MiniMaxH3JoinMatchClipStream.execute(
            samples=cur,
            strength=float(join_match_strength),
            context_length=int(trim_frames_p1),
            min_excess=1.5,
            match_spread=True,
            min_excess_spread=1.5,
            match_detail=True,
            min_excess_detail=1.5,
            tuning="auto",
            context_latent=context_latent,
        )
        jm_latent = out_jm[0] if hasattr(out_jm, "__getitem__") else out_jm.args[0]
        diag["join_match_applied"] = jm_latent is not cur
        cur = jm_latent
        steps_order.append("join_match")

    # Step 2: Root-Anchor gentle color anchor pull
    if root_stats and int(chain_depth) >= 1 and float(root_anchor_strength) > 1e-6:
        cur, ra_diag = apply_root_anchor_pull(
            cur,
            root_stats,
            trim_frames=int(trim_frames_p1),
            chain_depth=int(chain_depth),
            root_anchor_strength=float(root_anchor_strength),
        )
        diag["root_anchor"] = ra_diag
        steps_order.append("root_anchor")

    # Step 3: OverlapRestore (MUST run AFTER JoinMatch & Root-Anchor!)
    if overlap_restore_p2:
        cur = restore_overlap(
            cur,
            trim_frames=int(trim_frames_p1),
            enabled=True,
            context_latent=context_latent,
        )
        diag["overlap_restore_applied"] = True
        steps_order.append("overlap_restore")

    return cur, diag
