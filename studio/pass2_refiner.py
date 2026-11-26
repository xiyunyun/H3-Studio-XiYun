"""H3 Studio Node 7 (H3StudioPass2Refiner): Pass-2 High-Res Refiner, Anti-Drift & Decoder.

Strict execution pipeline (Pitfalls #1, #5, #6, #7, #8):
1. Build 0.1ms empty high-res AV template directly from upscaled_latent dimensions (ZERO 32B calls!).
2. Reuse cond_pack['positive'] directly (only re-encode image ref VAEs if ref_image_size == 'match').
3. Optional Guide Timeline v2 injection at high-res dimensions.
4. Optional Motion Context with Algorithm ③ (22 -> 5 offset=17 + audio anchored at 22),
   Algorithm ① (progressive flow-matched taper), and Algorithm ② (tail dynamic release wrapper).
5. Internal two-pass Reconcile verification (shapes, finiteness, spatial keyframe/ref contracts,
   preserving Pass-1 learned audio).
6. Native ComfyUI sampling on low_sigmas — Pitfall #5: MUST take slot 0 (output)!
7. Algorithm ④ strict order (Pitfall #8):
     JoinMatch (vs raw un-noised parent) -> Root-Anchor pull (vs root_stats)
     -> OverlapRestore (trim_frames_p1, vs raw un-noised parent).
8. Decode video + audio from the EXACT corrected latent_p2, then apply MotionContextTrim
   (primary + optional extra_trim_frames comparison stream).
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional, Tuple

import torch
from comfy_api.latest import io
from comfy_extras.nodes_minimax_h3 import _empty_av_latent

from ..clipstream.motion_context.nodes import MiniMaxH3MotionContextTrim
from ..lvm.comfyui.guide_nodes import H3GuideTimelineV2
from ..lvm.comfyui.guide_release import patch_model as patch_model_guide_release
from ..lvm.core.guide_release import parse_release as parse_guide_release
from .anti_drift import (
    apply_pass2_anti_drift_postprocess,
    compute_mc_release_threshold,
    inject_motion_context_studio,
    patch_model_mc_release,
)
from .av_upscaler import _repack_av_samples, _unpack_av_samples
from .cache import clone_conditioning_safe
from .pass1_sampler import run_stock_sampler
from .types import (
    H3CondPackType,
    H3ContinuationPipeType,
    H3Pass1ResultType,
    H3SamplingPlanType,
    validate_cond_pack,
    validate_continuation_pipe,
    validate_pass1_result,
    validate_sampling_plan,
)
from .unified_encoder import reencode_match_image_refs_for_highres

logger = logging.getLogger("h3_studio.pass2_refiner")


def reconcile_studio_pass2_latent(
    upscaled_latent: Dict[str, Any],
    highres_template: Dict[str, Any],
    positive: list,
) -> Tuple[Dict[str, Any], list]:
    """Strict internal two-pass spatial, audio, finiteness & conditioning contract validation."""
    up_video, up_audio, is_nested = _unpack_av_samples(upscaled_latent["samples"])
    tpl_video, tpl_audio, _ = _unpack_av_samples(highres_template["samples"])

    if tuple(up_video.shape) != tuple(tpl_video.shape):
        raise ValueError(
            f"H3 Studio Pass2 Reconcile: upscaled video shape {tuple(up_video.shape)} "
            f"!= highres template shape {tuple(tpl_video.shape)}."
        )
    if up_audio is not None and tpl_audio is not None and tuple(up_audio.shape) != tuple(tpl_audio.shape):
        raise ValueError(
            f"H3 Studio Pass2 Reconcile: audio shape {tuple(up_audio.shape)} "
            f"!= highres template audio shape {tuple(tpl_audio.shape)}."
        )
    if not torch.isfinite(up_video).all():
        raise ValueError("H3 Studio Pass2 Reconcile: upscaled video latent contains NaN or Inf.")
    if up_audio is not None and not torch.isfinite(up_audio).all():
        raise ValueError("H3 Studio Pass2 Reconcile: Pass-1 audio latent contains NaN or Inf.")

    h_lat, w_lat = int(up_video.shape[-2]), int(up_video.shape[-1])
    pos_out = clone_conditioning_safe(positive)
    for _emb, extra in pos_out:
        for idx, kf in enumerate(extra.get("minimax_keyframes") or []):
            lat = kf.get("latent")
            if lat is not None:
                if not isinstance(lat, torch.Tensor) or lat.ndim != 5:
                    raise ValueError(f"H3 Studio Pass2 Reconcile: keyframe #{idx} latent must be 5D.")
                if tuple(int(v) for v in lat.shape[-2:]) != (h_lat, w_lat):
                    raise ValueError(
                        f"H3 Studio Pass2 Reconcile: keyframe #{idx} spatial shape {tuple(lat.shape[-2:])} "
                        f"!= highres latent {(h_lat, w_lat)}."
                    )
        for idx, ref in enumerate(extra.get("minimax_refs") or []):
            kind = ref.get("kind")
            lat = ref.get("latent")
            if kind in ("image", "video", "video_audio") and lat is not None:
                declared = (int(ref.get("latent_h", -1)), int(ref.get("latent_w", -1)))
                actual = tuple(int(v) for v in lat.shape[-2:])
                if declared != actual:
                    raise ValueError(
                        f"H3 Studio Pass2 Reconcile: ref #{idx} metadata {declared} != latent {actual}."
                    )

    selected_audio = up_audio if up_audio is not None else tpl_audio
    out = highres_template.copy()
    out["samples"] = _repack_av_samples(highres_template["samples"], up_video, selected_audio, is_nested)
    return out, pos_out


def decode_av_latent(
    av_latent: Dict[str, Any],
    video_vae: Any,
    audio_vae: Any,
) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
    """Decode H3 AV NestedTensor latent into (IMAGE [F,H,W,3], AUDIO dict)."""
    video_lat, audio_lat, _ = _unpack_av_samples(av_latent["samples"])

    images = video_vae.decode(video_lat)
    if images.ndim == 5:
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])

    audio_out = None
    if audio_lat is not None and audio_vae is not None:
        try:
            from comfy_extras.nodes_audio import vae_decode_audio
            audio_out = vae_decode_audio(audio_vae, {"samples": audio_lat})
        except Exception:
            # Clean fallback for custom/mock audio VAEs in tests
            decoded = audio_vae.decode(audio_lat)
            if isinstance(decoded, dict) and "waveform" in decoded:
                audio_out = decoded
            else:
                wf = decoded
                if wf.ndim == 2:
                    wf = wf.unsqueeze(0)
                sr = int(getattr(audio_vae, "audio_sample_rate", 32000))
                audio_out = {"waveform": wf, "sample_rate": sr}

    return images, audio_out


class H3StudioPass2Refiner(io.ComfyNode):
    """High-res Pass-2 Refiner with zero 32B reload, 4-pillar anti-drift, and aligned AV decode/trim."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="H3StudioPass2Refiner",
            display_name="H3 Studio 二采精修与解码引擎",
            category="H3 Studio",
            description=(
                "0ms 复用一采 32B 条件、原地构建高分模板，完整内置算法①渐进加噪、②尾步松手、"
                "③22→5精确对齐、④首段锚定校色+OverlapRestore，并完成音画解码与精确对齐裁切。"
            ),
            inputs=[
                io.Model.Input("model", tooltip="二采模型明线（接二采 Power Lora Loader + taeh3 预览）"),
                io.Latent.Input("upscaled_latent", tooltip="来自 H3 Studio 联合 3D 放大器的高分 AV Latent"),
                H3Pass1ResultType.Input("pass1_result"),
                H3CondPackType.Input("cond_pack"),
                H3SamplingPlanType.Input("sampling_plan"),
                H3ContinuationPipeType.Input("continuation_pipe", optional=True),
                io.String.Input("shared_guides", optional=True, force_input=True, tooltip="直连 H3 Guide Timeline v2 的「共享 Guide 文档」输出"),
                io.Boolean.Input("enable_pass2_guide", default=False, label_on="开启二采 Guide", label_off="关闭", tooltip="是否在二采阶段应用共享 Guide 时间轴"),
                io.Int.Input("extra_trim_frames", default=2, min=0, max=32, tooltip="额外多裁 N 帧对比输出（用于跳变色对比；设 0 则零拷贝复用主输出）"),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.Audio.Output(display_name="audio"),
                io.Latent.Output(display_name="latent_p2"),
                io.Image.Output(display_name="extra_trimmed_images"),
                io.Audio.Output(display_name="extra_trimmed_audio"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        upscaled_latent: Dict[str, Any],
        pass1_result: Dict[str, Any],
        cond_pack: Dict[str, Any],
        sampling_plan: Dict[str, Any],
        continuation_pipe: Optional[Dict[str, Any]] = None,
        shared_guides: Optional[str] = None,
        enable_pass2_guide: bool = False,
        extra_trim_frames: int = 2,
    ) -> io.NodeOutput:
        t0 = time.perf_counter()
        p1 = validate_pass1_result(pass1_result)
        pack = validate_cond_pack(cond_pack)
        plan = validate_sampling_plan(sampling_plan)
        pipe = validate_continuation_pipe(continuation_pipe) if continuation_pipe is not None else None

        low_sigmas = p1["low_sigmas"]
        trim_frames_p1 = int(p1["trim_frames_p1"])

        # Step 1: 0.1ms empty high-res template directly from upscaled latent shape (ZERO 32B calls!)
        up_video, _, _ = _unpack_av_samples(upscaled_latent["samples"])
        h_lat = int(up_video.shape[-2])
        w_lat = int(up_video.shape[-1])
        highres_h = h_lat * 16
        highres_w = w_lat * 16
        highres_template, _fc = _empty_av_latent(highres_w, highres_h, int(pack["frame_count"]))

        # Step 2: Reuse Pass-1 32B positive conditioning (Pitfall #1: NEVER re-run 32B!)
        positive = clone_conditioning_safe(pack["positive"])
        match_reencoded = False
        if pack["ref_image_size"] == "match" and pack["raw_ref_images"]:
            positive = reencode_match_image_refs_for_highres(
                positive, pack["raw_ref_images"], pack["video_vae"], highres_w, highres_h
            )
            match_reencoded = True

        work_model = model

        # Step 3: Optional Pass-2 Guide Timeline v2 injection at high-res dimensions
        if enable_pass2_guide and shared_guides:
            guide_res = H3GuideTimelineV2._run(
                positive=positive,
                latent=highres_template,
                guides=shared_guides,
                vae=pack["video_vae"],
                audio_vae=pack["audio_vae"],
            )
            positive = guide_res[0]
            rel_cfg = parse_guide_release(shared_guides)
            if rel_cfg.get("enabled"):
                work_model = patch_model_guide_release(work_model)

        # Step 4: Motion Context + Algorithm ③ (22->5 offset=17) + ① (Taper) + ② (Dynamic Release)
        mc_info: Dict[str, Any] = {"active": False}
        release_threshold: Optional[float] = None
        if pipe is not None and not pipe["is_initial"] and pipe["latent_p2"] is not None:
            st = pipe["settings"]
            release_threshold = compute_mc_release_threshold(
                low_sigmas, int(st.get("p2_release_last_steps", 0))
            )
            effective_trim_p1 = trim_frames_p1 if trim_frames_p1 > 0 else int(st["pass1_context_length"])
            positive, trim_frames_p1, mc_info = inject_motion_context_studio(
                positive,
                highres_template,
                pipe["latent_p2"],
                context_length=int(st["pass2_context_length"]),
                trim_frames_p1=effective_trim_p1,
                audio_context_length=int(st["audio_context_length"]),
                enable_audio_context=bool(st["enable_audio_context"]),
                taper_enabled=bool(st["p2_taper_enabled"]),
                taper_start=float(st["p2_taper_start"]),
                taper_seam=float(st["p2_taper_seam"]),
                taper_ramp_steps=int(st["p2_taper_ramp_steps"]),
                release_threshold=release_threshold,
            )
            if release_threshold is not None:
                work_model = patch_model_mc_release(
                    work_model,
                    keep_audio=bool(st.get("p2_release_keep_audio", True)),
                )

        # Step 5: Internal Reconcile verification
        av_latent_in, positive = reconcile_studio_pass2_latent(
            upscaled_latent, highres_template, positive
        )

        # Step 6: Pass-2 sampling — Pitfall #5: MUST take slot 0 (output)!
        sampled_out, _denoised_out = run_stock_sampler(
            work_model,
            positive,
            av_latent_in,
            low_sigmas,
            seed=int(plan["seed"]),
            sampler_name=str(plan["sampler_name"]),
        )

        # Step 7: Algorithm ④ strict order (Pitfall #8): JoinMatch -> RootAnchor -> OverlapRestore
        post_diag: Dict[str, Any] = {"order": []}
        if pipe is not None and not pipe["is_initial"] and pipe["latent_p2"] is not None and trim_frames_p1 > 0:
            st = pipe["settings"]
            latent_p2, post_diag = apply_pass2_anti_drift_postprocess(
                sampled_out,
                context_latent=pipe["latent_p2"],
                trim_frames_p1=trim_frames_p1,
                root_stats=pipe.get("root_stats"),
                chain_depth=int(pipe.get("chain_depth", 0)),
                join_match_enabled=bool(st.get("join_match_enabled", True)),
                join_match_strength=float(st.get("join_match_strength", 1.0)),
                root_anchor_strength=float(st.get("root_anchor_strength", 0.30)),
                overlap_restore_p2=bool(st.get("overlap_restore_p2", True)),
            )
        else:
            latent_p2 = sampled_out

        # Step 8: Decode video & audio from the EXACT corrected latent_p2
        decoded_images, decoded_audio = decode_av_latent(
            latent_p2, pack["video_vae"], pack["audio_vae"]
        )

        # Step 9: Main aligned trim (trim_frames_p1 + frames_cap + match_tail=True)
        frame_cap = int(pack.get("frame_cap") or 0)
        trim_res = MiniMaxH3MotionContextTrim.execute(
            images=decoded_images,
            trim_frames=trim_frames_p1,
            audio=decoded_audio,
            fps=24.0,
            match_tail=True,
            frames_cap=frame_cap,
        )
        out_images, out_audio = trim_res[0], trim_res[1]

        # Step 10: Optional extra_trim_frames comparison stream (0 -> zero-copy reuse)
        extra = max(0, int(extra_trim_frames))
        if extra == 0:
            extra_images, extra_audio = out_images, out_audio
        else:
            effective_total = min(int(decoded_images.shape[0]), frame_cap) if frame_cap > 0 else int(decoded_images.shape[0])
            safe_trim = min(trim_frames_p1 + extra, max(0, effective_total - 1))
            extra_res = MiniMaxH3MotionContextTrim.execute(
                images=decoded_images,
                trim_frames=safe_trim,
                audio=decoded_audio,
                fps=24.0,
                match_tail=True,
                frames_cap=frame_cap,
            )
            extra_images, extra_audio = extra_res[0], extra_res[1]

        elapsed = round(time.perf_counter() - t0, 3)
        report_dict = {
            "pass2_seconds": elapsed,
            "skipped_32b_reload": True,
            "match_ref_vae_reencoded": match_reencoded,
            "highres_size": [highres_w, highres_h],
            "steps": max(0, len(low_sigmas) - 1),
            "trim_frames_p1": trim_frames_p1,
            "frame_cap": frame_cap,
            "delivered_frames": int(out_images.shape[0]),
            "motion_context": {
                "active": bool(mc_info.get("active")),
                "context_length": mc_info.get("context_length", 0),
                "frame_offset": mc_info.get("frame_offset", 0),
                "release_threshold": release_threshold,
            },
            "anti_drift_postprocess": post_diag,
        }
        report_str = json.dumps(report_dict, ensure_ascii=False, indent=2)
        logger.info(
            "[H3 Studio Pass2] done in %.2fs (%dx%d, delivered=%df, 32B_reused=True, post=%s)",
            elapsed, highres_w, highres_h, int(out_images.shape[0]), post_diag.get("order"),
        )
        return io.NodeOutput(out_images, out_audio, latent_p2, extra_images, extra_audio, report_str)


NODE_LIST = [H3StudioPass2Refiner]
