"""H3 Studio Node 6 (H3StudioAVLatentUpscaler): Fast Native-Core Joint AV 3D Upscaler.

Fixes Performance Bug #2 (Pitfall #3):
- Uses ComfyUI core comfy.ldm.minimax.latent_upscaler.MiniMaxH3LatentUpscaler
  (channels_last_3d + comfy_kitchen fused GroupNorm + exact SLAB=16 slabbing).
- Wraps model in CoreModelPatcher and schedules VRAM cleanly via
  comfy.model_management.load_models_gpu([patcher], memory_required=...), preventing
  Windows WDDM shared-VRAM thrashing!
- Unloads ONLY itself via unload_model_and_clones(patcher) when release_after=True;
  NEVER calls unload_all_models() so the DiT stays hot in VRAM.
- Accepts & emits joint AV NestedTensor directly, eliminating LTXVSeparateAVLatent
  and LTXVConcatAVLatent completely!
"""
from __future__ import annotations

import logging
import os
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

import torch
from comfy_api.latest import io

import folder_paths

from .types import (
    H3Pass1ResultType,
    H3SamplingPlanType,
    validate_pass1_result,
    validate_sampling_plan,
)

logger = logging.getLogger("h3_studio.av_upscaler")

LATENT_UPSCALE_FOLDER = "latent_upscale_models"
DEFAULT_UPSCALER_NAME = "minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors"
VAE_DOWNSAMPLE = 16

_PATCHER_CACHE: Dict[Tuple[str, int, int, str], Any] = {}
_CACHE_LOCK = threading.RLock()


def scan_upscale_models() -> List[str]:
    try:
        names = [
            name for name in folder_paths.get_filename_list(LATENT_UPSCALE_FOLDER)
            if os.path.splitext(name)[1].lower() in (".safetensors", ".pth", ".pt", ".ckpt")
        ]
    except Exception:
        names = []
    return names if names else [DEFAULT_UPSCALER_NAME]


def normalize_upscaler_state_dict(raw_sd: Dict[str, Any]) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    """Unwrap 'model' wrapper, strip 'upscaler.' prefix, cast fp8->fp16, and infer arch config."""
    sd = raw_sd
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]

    if any(k.startswith("upscaler.") for k in sd):
        sd = {k[len("upscaler."):]: v for k, v in sd.items() if k.startswith("upscaler.")}

    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    clean_sd: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if isinstance(v, torch.Tensor) and fp8_dtype is not None and v.dtype == fp8_dtype:
            clean_sd[k] = v.to(torch.float16)
        else:
            clean_sd[k] = v

    if "conv_in.weight" not in clean_sd:
        raise ValueError(
            "H3 Studio AV Upscaler: checkpoint missing 'conv_in.weight' after normalization."
        )

    in_channels = int(clean_sd["conv_in.weight"].shape[1])
    channels = int(clean_sd["conv_in.weight"].shape[0])
    in_ids = {
        int(m.group(1))
        for k in clean_sd
        for m in (re.match(r"in_blocks\.(\d+)\.in_layers\.", k),)
        if m
    }
    out_ids = {
        int(m.group(1))
        for k in clean_sd
        for m in (re.match(r"out_blocks\.(\d+)\.in_layers\.", k),)
        if m
    }
    temporal_kernel = 5
    for k, v in clean_sd.items():
        if k.endswith("dwconv.weight") and isinstance(v, torch.Tensor) and v.ndim >= 3:
            temporal_kernel = int(v.shape[2])
            break

    cfg = {
        "in_channels": in_channels,
        "channels": channels,
        "in_blocks": len(in_ids) if in_ids else 12,
        "out_blocks": len(out_ids) if out_ids else 12,
        "temporal_kernel": temporal_kernel,
    }
    return clean_sd, cfg


def compute_upscaled_pixel_size(
    latent_h: int,
    latent_w: int,
    upscale_factor: float,
    align: int = 32,
) -> Tuple[int, int, int, int]:
    """Compute aligned pixel (w_px, h_px) and latent (w_out, h_out) dimensions."""
    # Ensure alignment is at least 16 and divisible by 16 so w_px // 16 is exact.
    eff_align = max(16, int(align))
    if eff_align % VAE_DOWNSAMPLE != 0:
        eff_align = ((eff_align + VAE_DOWNSAMPLE - 1) // VAE_DOWNSAMPLE) * VAE_DOWNSAMPLE

    scale = max(1.0, float(upscale_factor))
    w_px = max(eff_align, round((int(latent_w) * VAE_DOWNSAMPLE * scale) / eff_align) * eff_align)
    h_px = max(eff_align, round((int(latent_h) * VAE_DOWNSAMPLE * scale) / eff_align) * eff_align)
    return int(w_px), int(h_px), int(w_px // VAE_DOWNSAMPLE), int(h_px // VAE_DOWNSAMPLE)


def _resolve_upscaler_dtype(precision: str) -> torch.dtype:
    import comfy.model_management as mm

    p = str(precision or "fp16").lower()
    if p == "fp16":
        return torch.float16
    if p == "bf16":
        return torch.bfloat16
    if p == "fp32":
        return torch.float32
    return mm.vae_dtype(allowed_dtypes=[torch.float16, torch.bfloat16, torch.float32])


def _load_or_get_upscaler_patcher(model_name: str, precision: str):
    """Load MiniMaxH3LatentUpscaler wrapped in CoreModelPatcher with mtime-aware caching."""
    import comfy.model_management as mm
    import comfy.model_patcher
    import comfy.ops
    import comfy.storage
    import comfy.utils
    from comfy.ldm.minimax.latent_upscaler import MiniMaxH3LatentUpscaler

    model_path = folder_paths.get_full_path_or_raise(LATENT_UPSCALE_FOLDER, model_name)
    st = os.stat(model_path)
    cache_key = (os.path.normcase(os.path.abspath(model_path)), int(st.st_size), int(st.st_mtime_ns), str(precision))

    with _CACHE_LOCK:
        cached = _PATCHER_CACHE.get(cache_key)
        if cached is not None:
            return cached

    raw_sd = comfy.utils.load_torch_file(model_path, safe_load=True)
    clean_sd, cfg = normalize_upscaler_state_dict(raw_sd)
    fast_disk = comfy.storage.state_dict_fast_disk(clean_sd)
    dtype = _resolve_upscaler_dtype(precision)

    model = MiniMaxH3LatentUpscaler(
        in_channels=cfg["in_channels"],
        channels=cfg["channels"],
        in_blocks=cfg["in_blocks"],
        out_blocks=cfg["out_blocks"],
        temporal_kernel=cfg["temporal_kernel"],
        dtype=dtype,
        operations=comfy.ops.disable_weight_init,
    )
    mm.archive_model_dtypes(model)
    patcher_cls = getattr(comfy.model_patcher, "CoreModelPatcher", comfy.model_patcher.ModelPatcher)
    patcher = patcher_cls(
        model,
        load_device=mm.get_torch_device(),
        offload_device=mm.unet_offload_device(),
        fast_disk=fast_disk,
    )
    assign_flag = patcher.is_dynamic() if hasattr(patcher, "is_dynamic") else False
    model.load_state_dict(clean_sd, strict=True, assign=assign_flag)

    with _CACHE_LOCK:
        # Evict stale entries for other checkpoints to keep RAM bounded.
        old_patchers = [p for k, p in _PATCHER_CACHE.items() if k != cache_key]
        _PATCHER_CACHE.clear()
        _PATCHER_CACHE[cache_key] = patcher

    for old in old_patchers:
        try:
            mm.unload_model_and_clones(old)
        except Exception:
            pass

    logger.info(
        "[H3 Studio AV Upscaler] loaded core MiniMaxH3LatentUpscaler '%s' (cfg=%s, dtype=%s)",
        model_name, cfg, dtype,
    )
    return patcher


def _unpack_av_samples(samples: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor], bool]:
    """Return (video_tensor, audio_tensor_or_None, is_nested)."""
    if getattr(samples, "is_nested", False) and hasattr(samples, "unbind"):
        parts = list(samples.unbind())
        return parts[0], (parts[1] if len(parts) > 1 else None), True
    if isinstance(samples, (list, tuple)):
        return samples[0], (samples[1] if len(samples) > 1 else None), True
    return samples, None, False


def _repack_av_samples(original: Any, video: torch.Tensor, audio: Optional[torch.Tensor], is_nested: bool):
    if not is_nested or audio is None:
        return video
    if isinstance(original, list):
        return [video, audio]
    if isinstance(original, tuple):
        return (video, audio)
    import comfy.nested_tensor
    return comfy.nested_tensor.NestedTensor((video, audio))


class H3StudioAVLatentUpscaler(io.ComfyNode):
    """Joint AV 3D latent upscaler built on ComfyUI core MiniMaxH3LatentUpscaler."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        models = scan_upscale_models()
        default_model = DEFAULT_UPSCALER_NAME if DEFAULT_UPSCALER_NAME in models else models[0]
        return io.Schema(
            node_id="H3StudioAVLatentUpscaler",
            display_name="H3 Studio 联合 3D 放大器",
            category="H3 Studio",
            description=(
                "基于 ComfyUI 原生高速内核 (MiniMaxH3LatentUpscaler + CoreModelPatcher + load_models_gpu) "
                "直接吞吐音画联合 AV Latent，消除 WDDM 共享显存换页暴慢，同时省去音画拆分/合并节点。"
            ),
            inputs=[
                H3Pass1ResultType.Input("pass1_result"),
                H3SamplingPlanType.Input("sampling_plan"),
                io.Combo.Input("model_name", options=models, default=default_model, tooltip="models/latent_upscale_models 下的 H3 3D Latent 放大模型"),
                io.Int.Input("align", default=32, min=16, max=256, step=16, tooltip="像素空间对齐倍数（H3 强烈建议 32）"),
                io.Combo.Input("precision", options=["auto", "fp16", "bf16", "fp32"], default="fp16", tooltip="3D 放大计算精度"),
                io.Boolean.Input("release_after", default=True, label_on="跑完仅卸载自身", label_off="常驻显存", tooltip="推理完成后精准卸载本放大器以释放显存，绝不踢出主 DiT 模型"),
            ],
            outputs=[
                io.Latent.Output(display_name="upscaled_av_latent"),
                io.Int.Output(display_name="highres_width"),
                io.Int.Output(display_name="highres_height"),
            ],
        )

    @classmethod
    def execute(
        cls,
        pass1_result: Dict[str, Any],
        sampling_plan: Dict[str, Any],
        model_name: str = DEFAULT_UPSCALER_NAME,
        align: int = 32,
        precision: str = "fp16",
        release_after: bool = True,
    ) -> io.NodeOutput:
        import comfy.model_management as mm

        p1 = validate_pass1_result(pass1_result)
        plan = validate_sampling_plan(sampling_plan)

        latent_in = p1["latent_p1"]
        orig_samples = latent_in["samples"]
        video, audio, is_nested = _unpack_av_samples(orig_samples)

        was_4d = video.ndim == 4
        video5 = video.unsqueeze(0) if was_4d else video
        _, _, t_lat, h_in, w_in = video5.shape

        w_px, h_px, w_out, h_out = compute_upscaled_pixel_size(
            h_in, w_in, float(plan["upscale_factor"]), align=int(align)
        )
        target_size = (int(t_lat), int(h_out), int(w_out))

        if (h_out, w_out) == (h_in, w_in):
            logger.info("[H3 Studio AV Upscaler] scale=1.0 (%dx%d), zero-cost passthrough.", w_px, h_px)
            return io.NodeOutput(latent_in, w_px, h_px)

        patcher = _load_or_get_upscaler_patcher(model_name, precision)
        try:
            mem_req = patcher.model.memory_required(video5.shape, target_size)
            mm.load_models_gpu([patcher], memory_required=mem_req)
            run_dtype = patcher.model_dtype() if hasattr(patcher, "model_dtype") else patcher.model.get_dtype()
            with torch.inference_mode():
                inp = video5.to(device=patcher.load_device, dtype=run_dtype)
                upscaled = patcher.model(inp, target_size)
                upscaled = upscaled.to(device=mm.intermediate_device(), dtype=video.dtype)
        finally:
            if release_after:
                # Pitfall #3: unload ONLY this upscaler patcher, NEVER unload_all_models()!
                try:
                    mm.unload_model_and_clones(patcher)
                except Exception:
                    pass

        if was_4d:
            upscaled = upscaled.squeeze(0)

        out = latent_in.copy()
        out["samples"] = _repack_av_samples(orig_samples, upscaled, audio, is_nested)
        out.pop("noise_mask", None)
        logger.info(
            "[H3 Studio AV Upscaler] %dx%d -> %dx%d (latent %dx%d -> %dx%d, T=%d, audio=%s)",
            w_in * VAE_DOWNSAMPLE, h_in * VAE_DOWNSAMPLE, w_px, h_px,
            w_in, h_in, w_out, h_out, t_lat,
            "preserved" if audio is not None else "none",
        )
        return io.NodeOutput(out, w_px, h_px)


NODE_LIST = [H3StudioAVLatentUpscaler]
