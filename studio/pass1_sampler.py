"""H3 Studio Node 5 (H3StudioPass1Sampler): Pass-1 Sampling Engine with 1-Slot Cache.

- Unified sigma scheduling: BasicScheduler + self-contained H3SigmaRefiner + SplitSigmas
- 1-slot content-hash cache: changing Pass-2 params / upscale factor / anti-drift Pass-2
  settings skips Pass 1 in <1ms!
- Supports optional Guide Timeline v2 shared document + Motion Context Algorithm ①
- Pitfall #5 enforced: ALWAYS returns slot 1 (denoised_output) because high_sigmas[-1] > 0!
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from comfy_api.latest import io

from ..clipstream.motion_context.overlap_restore import restore_overlap
from ..lvm.comfyui.guide_nodes import H3GuideTimelineV2
from ..lvm.comfyui.guide_release import patch_model as patch_model_guide_release
from ..lvm.core.guide_release import parse_release as parse_guide_release
from .anti_drift import inject_motion_context_studio
from .cache import (
    PASS1_CACHE,
    clone_conditioning_safe,
    model_patch_fingerprint,
    stable_hash,
)
from .sigma_plan import build_studio_sigmas
from .types import (
    SCHEMA_VERSION,
    H3CondPackType,
    H3ContinuationPipeType,
    H3Pass1ResultType,
    H3SamplingPlanType,
    validate_cond_pack,
    validate_continuation_pipe,
    validate_sampling_plan,
)

logger = logging.getLogger("h3_studio.pass1_sampler")


def run_stock_sampler(model: Any, conditioning: list, latent: Dict[str, Any], sigmas, seed: int, sampler_name: str):
    """Execute ComfyUI's stock BasicGuider + RandomNoise + KSamplerSelect + SamplerCustomAdvanced.

    Using ComfyUI's own nodes ensures ModelPreviewOverrideKJ (taeh3) callbacks and
    all model wrappers run identically to hand-wired graphs.
    Returns (noisy_output, denoised_output).
    """
    from comfy_extras.nodes_custom_sampler import (
        BasicGuider,
        KSamplerSelect,
        RandomNoise,
        SamplerCustomAdvanced,
    )

    noise = RandomNoise.execute(int(seed))[0]
    guider = BasicGuider.execute(model, conditioning)[0]
    sampler = KSamplerSelect.execute(str(sampler_name))[0]
    res = SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent)
    return res[0], res[1]


def compute_pass1_cache_key(
    model: Any,
    cond_pack: Dict[str, Any],
    sampling_plan: Dict[str, Any],
    high_sigmas,
    continuation_pipe: Optional[Dict[str, Any]],
    shared_guides: Optional[str],
    enable_pass1_guide: bool,
) -> str:
    """Pass-1 cache key: sensitive ONLY to inputs that affect Pass-1 output."""
    p1_cont = None
    if continuation_pipe is not None and not continuation_pipe["is_initial"] and continuation_pipe["latent_p1"] is not None:
        st = continuation_pipe["settings"]
        p1_cont = {
            "parent": continuation_pipe["parent_clip_id"],
            "p1_ctx": st["pass1_context_length"],
            "aud_ctx": st["audio_context_length"],
            "en_aud": st["enable_audio_context"],
            "taper_en": st["p1_taper_enabled"],
            "taper_start": st["p1_taper_start"],
            "taper_seam": st["p1_taper_seam"],
            "taper_ramp": st["p1_taper_ramp_steps"],
            "ov_p1": st["overlap_restore_p1"],
        }

    guide_fp = ""
    if enable_pass1_guide and shared_guides:
        guide_fp = H3GuideTimelineV2.fingerprint_inputs(shared_guides, enable_guide=True)

    return stable_hash({
        "model": model_patch_fingerprint(model),
        "encode_fp": cond_pack["encode_fingerprint"],
        "w": cond_pack["width_p1"],
        "h": cond_pack["height_p1"],
        "frames": cond_pack["frame_count"],
        "seed": int(sampling_plan["seed"]),
        "sampler": str(sampling_plan["sampler_name"]),
        "high_sigmas": [round(float(x), 7) for x in high_sigmas.detach().cpu().tolist()],
        "cont": p1_cont,
        "guide": guide_fp,
    })


class H3StudioPass1Sampler(io.ComfyNode):
    """Pass-1 coarse AV sampler with integrated scheduler, Guide v2, Motion Context ①, and 1-slot cache."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="H3StudioPass1Sampler",
            display_name="H3 Studio 一采采样引擎",
            category="H3 Studio",
            description=(
                "内置 BasicScheduler + H3SigmaRefiner + SplitSigmas 与 1-slot 内容哈希缓存。"
                "支持共享 Guide 文档与一采渐进声明加噪接续，强制输出干净的 denoised_output 供 3D 放大。"
            ),
            inputs=[
                io.Model.Input("model", tooltip="一采模型明线（接一采 Power Lora Loader + taeh3 预览）"),
                H3CondPackType.Input("cond_pack"),
                H3SamplingPlanType.Input("sampling_plan"),
                H3ContinuationPipeType.Input("continuation_pipe", optional=True),
                io.String.Input("shared_guides", optional=True, force_input=True, tooltip="直连 H3 Guide Timeline v2 的「共享 Guide 文档」输出"),
                io.Boolean.Input("enable_pass1_guide", default=False, label_on="开启一采 Guide", label_off="关闭", tooltip="是否在一采阶段应用共享 Guide 时间轴"),
            ],
            outputs=[
                H3Pass1ResultType.Output(display_name="pass1_result"),
                io.Latent.Output(display_name="latent_p1"),
                io.Int.Output(display_name="trim_frames_p1"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        cond_pack: Dict[str, Any],
        sampling_plan: Dict[str, Any],
        continuation_pipe: Optional[Dict[str, Any]] = None,
        shared_guides: Optional[str] = None,
        enable_pass1_guide: bool = False,
    ) -> io.NodeOutput:
        pack = validate_cond_pack(cond_pack)
        plan = validate_sampling_plan(sampling_plan)
        pipe = validate_continuation_pipe(continuation_pipe) if continuation_pipe is not None else None

        _full_sigmas, high_sigmas, low_sigmas = build_studio_sigmas(model, plan)

        pass1_key = compute_pass1_cache_key(
            model, pack, plan, high_sigmas, pipe, shared_guides, bool(enable_pass1_guide)
        )
        hit, cached = PASS1_CACHE.get(pass1_key)
        if hit:
            latent_p1 = cached["latent_p1"]
            trim_frames_p1 = int(cached["trim_frames_p1"])
            logger.info(
                "[H3 Studio Pass1] CACHE HIT (key=%s, 0ms, trim=%df) — skipping Pass 1!",
                pass1_key[:10], trim_frames_p1,
            )
        else:
            work_model = model
            positive = clone_conditioning_safe(pack["positive"])
            latent_in = pack["latent_pass1"]

            # Optional Guide Timeline v2 injection at Pass 1 resolution
            if enable_pass1_guide and shared_guides:
                guide_res = H3GuideTimelineV2._run(
                    positive=positive,
                    latent=latent_in,
                    guides=shared_guides,
                    vae=pack["video_vae"],
                    audio_vae=pack["audio_vae"],
                )
                positive = guide_res[0]
                rel_cfg = parse_guide_release(shared_guides)
                if rel_cfg.get("enabled"):
                    work_model = patch_model_guide_release(work_model)

            # Optional Motion Context + Algorithm ① (Flow-Matched Progressive Taper)
            trim_frames_p1 = 0
            if pipe is not None and not pipe["is_initial"] and pipe["latent_p1"] is not None:
                st = pipe["settings"]
                positive, trim_frames_p1, mc_info = inject_motion_context_studio(
                    positive,
                    latent_in,
                    pipe["latent_p1"],
                    context_length=int(st["pass1_context_length"]),
                    trim_frames_p1=None,
                    audio_context_length=int(st["audio_context_length"]),
                    enable_audio_context=bool(st["enable_audio_context"]),
                    taper_enabled=bool(st["p1_taper_enabled"]),
                    taper_start=float(st["p1_taper_start"]),
                    taper_seam=float(st["p1_taper_seam"]),
                    taper_ramp_steps=int(st["p1_taper_ramp_steps"]),
                )
                logger.info(
                    "[H3 Studio Pass1] Motion Context injected: %df (taper=%s)",
                    trim_frames_p1, mc_info.get("taper_coeffs"),
                )

            # Sample with ComfyUI native sampler; Pitfall #5: MUST take slot 1 (denoised_output)!
            _noisy_out, denoised_out = run_stock_sampler(
                work_model,
                positive,
                latent_in,
                high_sigmas,
                seed=int(plan["seed"]),
                sampler_name=str(plan["sampler_name"]),
            )
            latent_p1 = denoised_out

            # Optional Pass-1 OverlapRestore (default False per empirical testing)
            if (
                pipe is not None
                and not pipe["is_initial"]
                and pipe["latent_p1"] is not None
                and trim_frames_p1 > 0
                and bool(pipe["settings"].get("overlap_restore_p1", False))
            ):
                latent_p1 = restore_overlap(
                    latent_p1,
                    trim_frames=trim_frames_p1,
                    enabled=True,
                    context_latent=pipe["latent_p1"],
                )

            PASS1_CACHE.put(pass1_key, {
                "latent_p1": latent_p1,
                "trim_frames_p1": trim_frames_p1,
            })
            logger.info(
                "[H3 Studio Pass1] sampled & cached (key=%s, steps=%d, trim=%df)",
                pass1_key[:10], len(high_sigmas) - 1, trim_frames_p1,
            )

        pass1_result: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "latent_p1": latent_p1,
            "low_sigmas": low_sigmas,
            "high_sigmas": high_sigmas,
            "trim_frames_p1": trim_frames_p1,
            "seed": int(plan["seed"]),
            "pass1_fingerprint": pass1_key,
        }
        return io.NodeOutput(pass1_result, latent_p1, trim_frames_p1)


NODE_LIST = [H3StudioPass1Sampler]
