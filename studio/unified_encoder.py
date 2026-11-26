"""H3 Studio Node 4 (H3StudioUnifiedEncoder): Single-Pass 32B Conditioning Encoder with 1-Slot Cache.

Fixes Performance Bug #1 (Pitfalls #1 & #2):
- Runs Qwen3-VL 32B + reference VAE encoding ONCE per prompt/media combination.
- Pass 2 reuses cond_pack['positive'] directly (0ms 32B cost, zero DiT <-> 32B thrashing).
- Internal 1-slot content-hash cache ensures changing seed, steps, upscale factor, or
  continuation card never triggers redundant 32B encoding.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import node_helpers
import torch
from comfy_api.latest import io
from comfy_extras.nodes_minimax_h3 import (
    CANVAS_MULTIPLE,
    FPS,
    REF_IMAGE_SHORT_EDGE,
    _empty_av_latent,
    _encode_ref_audio,
    _resize,
    adapt_canvas,
)

from .cache import ENCODE_CACHE, clone_conditioning_safe, stable_hash
from .types import (
    SCHEMA_VERSION,
    H3CondPackType,
    H3EncodeEnvType,
    H3MediaBundleType,
    H3VideoRefBundleType,
    validate_encode_env,
    validate_media_bundle,
    validate_video_ref_bundle,
)

logger = logging.getLogger("h3_studio.unified_encoder")


def encode_image_ref_blocks(
    raw_ref_images: List[torch.Tensor],
    video_vae: Any,
    width: int,
    height: int,
    ref_image_size: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build Qwen ref_items and DiT ref_blocks for reference images (exact stock math)."""
    ref_items: List[Dict[str, Any]] = []
    ref_blocks: List[Dict[str, Any]] = []
    for img in raw_ref_images:
        if img is None:
            continue
        h, w = int(img.shape[1]), int(img.shape[2])
        if ref_image_size == "match":
            scale = min(1.0, math.sqrt((width * height) / float(w * h)))
        else:
            scale = min(1.0, REF_IMAGE_SHORT_EDGE / float(min(w, h)))
        tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        resized = _resize(img[:1], tw, th, "disabled")
        ref_items.append({"type": "image", "data": resized})
        if video_vae is not None:
            z = video_vae.encode(resized)
            ref_blocks.append({
                "kind": "image",
                "latent_h": th // 16,
                "latent_w": tw // 16,
                "latent": z,
            })
    return ref_items, ref_blocks


def reencode_match_image_refs_for_highres(
    positive: list,
    raw_ref_images: List[torch.Tensor],
    video_vae: Any,
    highres_width: int,
    highres_height: int,
) -> list:
    """For ref_image_size == 'match' only: re-encode image ref blocks at highres area.

    Reuses the Qwen3-VL 32B text embeddings without EVER invoking CLIP again (Pitfall #1).
    """
    if not raw_ref_images or video_vae is None:
        return positive
    _, new_img_blocks = encode_image_ref_blocks(
        raw_ref_images, video_vae, highres_width, highres_height, "match"
    )
    out = clone_conditioning_safe(positive)
    for _emb, extra in out:
        old_refs = extra.get("minimax_refs") or []
        non_img_refs = [r for r in old_refs if r.get("kind") != "image"]
        extra["minimax_refs"] = [dict(b) for b in new_img_blocks] + non_img_refs
    return out


def compute_encode_cache_key(
    env: Dict[str, Any],
    media_bundle: Optional[Dict[str, Any]],
    video_ref_bundle: Optional[Dict[str, Any]],
    frame_count: int,
) -> str:
    """Precise 32B cache key: ignores seed, pass2 params, continuation, and max-mode canvas size."""
    ref_size = env["ref_image_size"]
    video_enabled = bool(video_ref_bundle and video_ref_bundle.get("enabled"))
    payload = {
        "clip_id": id(env["clip"]),
        "video_vae_id": id(env["video_vae"]),
        "audio_vae_id": id(env["audio_vae"]),
        "prompt": env["prompt"],
        "ref_image_size": ref_size,
        # Only match mode depends on canvas pixel area
        "canvas": (env["width"], env["height"]) if ref_size == "match" else None,
        "media_fp": media_bundle["media_fingerprint"] if media_bundle is not None else "none",
        "video_fp": video_ref_bundle["video_fingerprint"] if video_enabled else "off",
        # Only reference video truncation depends on frame_count
        "video_frame_count": int(frame_count) if video_enabled else None,
    }
    return stable_hash(payload)


class H3StudioUnifiedEncoder(io.ComfyNode):
    """Single-pass MiniMax H3 32B encoder with 1-slot content-hash memory cache."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="H3StudioUnifiedEncoder",
            display_name="H3 Studio 统一条件编码器",
            category="H3 Studio",
            description=(
                "全程唯一 Qwen3-VL 32B 编码入口，内置 1-slot 内容哈希缓存。"
                "二采 0ms 复用本节点输出的 positive，彻底消灭二采前踢模型重载 32B 的性能灾难。"
            ),
            inputs=[
                H3EncodeEnvType.Input("encode_env"),
                H3MediaBundleType.Input("media_bundle", optional=True),
                H3VideoRefBundleType.Input("video_ref_bundle", optional=True),
            ],
            outputs=[
                H3CondPackType.Output(display_name="cond_pack"),
                io.Latent.Output(display_name="latent_pass1"),
                io.Conditioning.Output(display_name="positive"),
            ],
        )

    @classmethod
    def execute(
        cls,
        encode_env: Dict[str, Any],
        media_bundle: Optional[Dict[str, Any]] = None,
        video_ref_bundle: Optional[Dict[str, Any]] = None,
    ) -> io.NodeOutput:
        env = validate_encode_env(encode_env)
        mb = validate_media_bundle(media_bundle) if media_bundle is not None else None
        vb = validate_video_ref_bundle(video_ref_bundle) if video_ref_bundle is not None else None

        video_enabled = bool(vb is not None and vb["enabled"] and vb["video"] is not None)
        if video_enabled and vb["use_video_duration"]:
            target_length = int(vb["aligned_frame_count"])
            frame_cap = int(vb["raw_frame_count"])
        else:
            target_length = int(env["manual_aligned_frames"])
            # When NOT overriding duration with reference video, never trim manual duration!
            frame_cap = 0

        width = int(env["width"])
        height = int(env["height"])
        latent_pass1, frame_count = _empty_av_latent(width, height, target_length)

        encode_key = compute_encode_cache_key(env, mb, vb, frame_count)
        hit, cached = ENCODE_CACHE.get(encode_key)

        raw_ref_images: List[torch.Tensor] = []
        if mb is not None:
            raw_ref_images = [item["tensor"] for item in mb["images"] if item.get("tensor") is not None]

        if hit:
            positive = clone_conditioning_safe(cached["positive"])
            logger.info(
                "[H3 Studio UnifiedEncoder] 32B CACHE HIT (key=%s, 0ms encode, %dx%d, %df)",
                encode_key[:10], width, height, frame_count,
            )
        else:
            clip = env["clip"]
            video_vae = env["video_vae"]
            audio_vae = env["audio_vae"]
            prompt = env["prompt"]
            ref_image_size = env["ref_image_size"]

            ref_items: List[Dict[str, Any]] = []
            ref_blocks: List[Dict[str, Any]] = []

            # 1. Reference images (order 1..9)
            img_items, img_blocks = encode_image_ref_blocks(
                raw_ref_images, video_vae, width, height, ref_image_size
            )
            ref_items.extend(img_items)
            ref_blocks.extend(img_blocks)

            # 2. Optional reference video + paired soundtrack (exact stock L331-L366 logic)
            if video_enabled:
                video_frames = vb["video"]
                soundtrack = vb["audio"]
                vh, vw = int(video_frames.shape[1]), int(video_frames.shape[2])
                cw, ch = adapt_canvas(vw, vh)
                if vw * vh < cw * ch:
                    cw = max(CANVAS_MULTIPLE, round(vw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
                    ch = max(CANVAS_MULTIPLE, round(vh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
                frames = _resize(video_frames, cw, ch, "disabled")
                if frames.shape[0] > frame_count:
                    frames = frames[:frame_count]
                n = int(frames.shape[0])
                if n < 5:
                    raise ValueError("H3 Studio: 参考视频至少需要 5 帧 (~0.2s @ 24fps)")
                while n % 17 != 5:
                    n -= 1
                frames = frames[:n]

                if soundtrack is not None:
                    ref_items.append({"type": "audio"})
                sample_idx = list(range(0, int(frames.shape[0]), FPS // 2))
                qwen_frames = frames[sample_idx]
                ref_items.append({
                    "type": "video",
                    "data": qwen_frames,
                    "timestamps": [i / 2.0 for i in range(len(sample_idx))],
                })
                if video_vae is not None:
                    z = video_vae.encode(frames)
                    audio_latent, ref_audio_t = (None, 0)
                    if soundtrack is not None and audio_vae is not None:
                        audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, soundtrack)
                    ref_blocks.append({
                        "kind": "video_audio" if ref_audio_t else "video",
                        "latent_t": int(z.shape[2]),
                        "latent_h": ch // 16,
                        "latent_w": cw // 16,
                        "ref_audio_t": int(ref_audio_t),
                        "latent": z,
                        "audio_latent": audio_latent,
                    })

            # 3. Standalone reference audios (order 1..3)
            if mb is not None:
                for item in mb["audios"]:
                    audio = item.get("audio")
                    if audio is None:
                        continue
                    ref_items.append({"type": "audio"})
                    if audio_vae is not None:
                        audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, audio)
                        ref_blocks.append({
                            "kind": "audio",
                            "ref_audio_t": int(ref_audio_t),
                            "audio_latent": audio_latent,
                        })

            tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
            cond = clip.encode_from_tokens_scheduled(tokens)
            if ref_blocks:
                cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": ref_blocks})

            ENCODE_CACHE.put(encode_key, {"positive": cond})
            positive = clone_conditioning_safe(cond)
            logger.info(
                "[H3 Studio UnifiedEncoder] 32B encoded & cached (key=%s, imgs=%d, vid=%s, auds=%d, %dx%d, %df)",
                encode_key[:10],
                len(raw_ref_images),
                "on" if video_enabled else "off",
                len(mb["audios"]) if mb is not None else 0,
                width, height, frame_count,
            )

        cond_pack: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "positive": positive,
            "latent_pass1": latent_pass1,
            "width_p1": width,
            "height_p1": height,
            "frame_count": int(frame_count),
            "frame_cap": int(frame_cap),
            "ref_image_size": env["ref_image_size"],
            "raw_ref_images": raw_ref_images if env["ref_image_size"] == "match" else [],
            "video_vae": env["video_vae"],
            "audio_vae": env["audio_vae"],
            "encode_fingerprint": encode_key,
        }
        return io.NodeOutput(cond_pack, latent_pass1, clone_conditioning_safe(positive))


NODE_LIST = [H3StudioUnifiedEncoder]
