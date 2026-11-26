"""H3 Studio Node 3 (H3StudioMasterConfig): All-in-One Master Parameter Hub.

Consolidates encoding-level parameters (CLIP, Video/Audio VAE, resolution, duration,
prompt, ref_image_size) and sampling-level parameters (seed, pass1/pass2 steps,
upscale_factor, scheduler, sampler, sigma refiner) into a single clean console node.
Downstream 1-slot content-hash caches prevent changing seed/pass-2 settings from
re-running 32B encoding or Pass 1 sampling (Pitfall #2).
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

from comfy_api.latest import io

from .sigma_plan import SCHEDULER_OPTIONS, SIGMA_REFINER_SPACINGS
from .types import (
    SCHEMA_VERSION,
    H3EncodeEnvType,
    H3SamplingPlanType,
    align_frames_17n_plus_5,
)

# Exact match with ComfyUI native comfy_extras/nodes_resolution.py
ASPECT_RATIOS: Dict[str, Tuple[int, int]] = {
    "1:1 (Square)": (1, 1),
    "2:3 (Portrait Photo)": (2, 3),
    "3:2 (Photo)": (3, 2),
    "3:4 (Portrait Standard)": (3, 4),
    "4:3 (Standard)": (4, 3),
    "9:16 (Portrait Widescreen)": (9, 16),
    "16:9 (Widescreen)": (16, 9),
    "21:9 (Ultrawide)": (21, 9),
}
ASPECT_RATIO_OPTIONS = tuple(ASPECT_RATIOS.keys())
DEFAULT_ASPECT_RATIO = "9:16 (Portrait Widescreen)"


def _sampler_options() -> Tuple[str, ...]:
    try:
        import comfy.samplers
        samplers = tuple(comfy.samplers.KSampler.SAMPLERS)
        if samplers:
            return samplers
    except Exception:
        pass
    return ("euler", "euler_ancestral", "heun", "dpm_2", "dpmpp_2m", "uni_pc", "res_multistep")


def compute_resolution(aspect_ratio: str, megapixels: float, multiple: int) -> Tuple[int, int]:
    """Exact ComfyUI ResolutionSelector math (1.0 MP = 1024*1024 px)."""
    if aspect_ratio not in ASPECT_RATIOS:
        raise ValueError(f"H3 Studio: unknown aspect_ratio {aspect_ratio!r}")
    w_ratio, h_ratio = ASPECT_RATIOS[aspect_ratio]
    mult = max(16, int(multiple))
    total_pixels = float(megapixels) * 1024.0 * 1024.0
    scale = math.sqrt(total_pixels / float(w_ratio * h_ratio))
    width = max(mult, round(w_ratio * scale / mult) * mult)
    height = max(mult, round(h_ratio * scale / mult) * mult)
    return int(width), int(height)


def compute_aligned_frames(duration_seconds: float, fps: float = 24.0) -> int:
    """Convert duration in seconds at 24fps and snap up to H3 17n+5 grid."""
    raw = max(5, int(round(float(duration_seconds) * float(fps))))
    return align_frames_17n_plus_5(raw)


def combine_prompts(custom_prompt: str, prompt: str) -> str:
    """Join optional custom/LoRA trigger prompt with the main prompt cleanly."""
    c = (custom_prompt or "").strip()
    p = (prompt or "").strip()
    if c and p:
        return f"{c}\n{p}"
    return c or p


class H3StudioMasterConfig(io.ComfyNode):
    """Unified encoding + sampling parameter console."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        samplers = list(_sampler_options())
        default_sampler = "euler" if "euler" in samplers else samplers[0]
        return io.Schema(
            node_id="H3StudioMasterConfig",
            display_name="H3 Studio 一体化总控参数台",
            category="H3 Studio",
            description=(
                "一屏掌控分辨率、时长、提示词、Seed、一采/二采步数、二采倍率与 Sigma 低噪精修。"
                "下游统一编码器与一采引擎内置内容哈希缓存，改 Seed 不重跑 32B，改二采参数不重跑一采。"
            ),
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("video_vae"),
                io.Vae.Input("audio_vae"),
                io.Combo.Input("aspect_ratio", options=list(ASPECT_RATIO_OPTIONS), default=DEFAULT_ASPECT_RATIO, tooltip="画幅比例（与 ComfyUI 原生 ResolutionSelector 完全一致）"),
                io.Float.Input("megapixels", default=0.6, min=0.1, max=16.0, step=0.01, tooltip="一采目标像素量 (MP)，1.0 MP = 1024×1024"),
                io.Int.Input("multiple", default=32, min=16, max=128, step=16, tooltip="宽高对齐倍数（H3 要求 32）"),
                io.Float.Input("duration_seconds", default=5.0, min=0.2, max=150.0, step=0.1, tooltip="目标生成时长（秒），按 24fps 自动向上对齐 17n+5（5s→124f）"),
                io.String.Input("custom_prompt", default="", multiline=True, dynamic_prompts=True, tooltip="自定义前缀 / LoRA 触发词（可留空）"),
                io.String.Input("prompt", default="", multiline=True, dynamic_prompts=True, tooltip="主提示词（支持 <Picture 1> / <Audio 1> / <Video 1> 标签）"),
                io.Combo.Input("ref_image_size", options=["max", "match"], default="max", tooltip="参考图尺寸模式：max 为 2048 短边最高保真（默认），match 按生成画幅面积缩放"),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True, tooltip="采样随机种子"),
                io.Int.Input("pass1_steps", default=6, min=1, max=100, tooltip="一采低分粗采步数"),
                io.Int.Input("pass2_steps", default=4, min=1, max=100, tooltip="二采高分精修基础步数"),
                io.Float.Input("upscale_factor", default=1.5, min=1.0, max=4.0, step=0.05, tooltip="3D Latent 放大倍率（用于二采分辨率）"),
                io.Combo.Input("scheduler", options=list(SCHEDULER_OPTIONS), default="simple", tooltip="Sigma 调度器"),
                io.Combo.Input("sampler_name", options=samplers, default=default_sampler, tooltip="采样器算法"),
                io.Int.Input("sigma_refiner_extra_steps", default=1, min=0, max=15, advanced=True, tooltip="低噪区间额外微雕步数（原 H3SigmaRefiner）"),
                io.Float.Input("sigma_refiner_start", default=0.7, min=0.0, max=20.0, step=0.01, advanced=True, tooltip="启动低噪精修的 Sigma 阈值"),
                io.Float.Input("sigma_refiner_end", default=0.0, min=0.0, max=5.0, step=0.01, advanced=True, tooltip="低噪精修结束 Sigma 边界"),
                io.Combo.Input("sigma_refiner_spacing", options=list(SIGMA_REFINER_SPACINGS), default="cosine", advanced=True, tooltip="低噪加密插值曲线"),
            ],
            outputs=[
                H3EncodeEnvType.Output(display_name="encode_env"),
                H3SamplingPlanType.Output(display_name="sampling_plan"),
                io.String.Output(display_name="full_prompt"),
            ],
        )

    @classmethod
    def execute(
        cls,
        clip,
        video_vae,
        audio_vae,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        megapixels: float = 0.6,
        multiple: int = 32,
        duration_seconds: float = 5.0,
        custom_prompt: str = "",
        prompt: str = "",
        ref_image_size: str = "max",
        seed: int = 0,
        pass1_steps: int = 6,
        pass2_steps: int = 4,
        upscale_factor: float = 1.5,
        scheduler: str = "simple",
        sampler_name: str = "euler",
        sigma_refiner_extra_steps: int = 1,
        sigma_refiner_start: float = 0.7,
        sigma_refiner_end: float = 0.0,
        sigma_refiner_spacing: str = "cosine",
    ) -> io.NodeOutput:
        width, height = compute_resolution(aspect_ratio, megapixels, multiple)
        aligned_frames = compute_aligned_frames(duration_seconds)
        full_prompt = combine_prompts(custom_prompt, prompt)

        encode_env = {
            "schema_version": SCHEMA_VERSION,
            "clip": clip,
            "video_vae": video_vae,
            "audio_vae": audio_vae,
            "width": width,
            "height": height,
            "manual_aligned_frames": aligned_frames,
            "prompt": full_prompt,
            "ref_image_size": str(ref_image_size or "max"),
        }
        sampling_plan = {
            "schema_version": SCHEMA_VERSION,
            "seed": int(seed),
            "pass1_steps": max(1, int(pass1_steps)),
            "pass2_steps": max(1, int(pass2_steps)),
            "upscale_factor": max(1.0, float(upscale_factor)),
            "scheduler": str(scheduler or "simple"),
            "sampler_name": str(sampler_name or "euler"),
            "sigma_refiner_extra_steps": max(0, int(sigma_refiner_extra_steps)),
            "sigma_refiner_start": float(sigma_refiner_start),
            "sigma_refiner_end": float(sigma_refiner_end),
            "sigma_refiner_spacing": str(sigma_refiner_spacing or "cosine"),
        }
        return io.NodeOutput(encode_env, sampling_plan, full_prompt)


NODE_LIST = [H3StudioMasterConfig]
