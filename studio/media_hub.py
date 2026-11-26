"""H3 Studio Node 1 (H3StudioMediaHub) and Node 9 (H3StudioBundleUnpack).

Merges H3 Image Reference Loader (0..8) and H3 Audio Reference Loader (0..2) into
a single 2-Tab media hub outputting a strong-typed H3_MEDIA_BUNDLE, plus a compatibility
unpacker escape hatch for bridging stock or third-party nodes.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from comfy_api.latest import io

from ..lvm.comfyui.ref_nodes import (
    _media_roots,
    _safe_fingerprint,
    build_audio_slot,
    build_image_slot,
)
from ..lvm.comfyui.segment_store import DEFAULT_PROJECT
from ..lvm.core.ref_media import (
    MAX_REF_AUDIOS,
    MAX_REF_IMAGES,
    active_entries,
    parse_media_list,
    resolve_media_path,
    trim_range,
)
from ..lvm.core.ref_size import (
    DEFAULT_MEGAPIXELS,
    DEFAULT_RESAMPLE_METHOD,
    DEFAULT_UPSCALE_SMALL_IMAGES,
    MAX_MEGAPIXELS,
    MIN_MEGAPIXELS,
    RESAMPLE_METHODS,
    clamp_megapixels,
    clamp_resampling_method,
)
from .cache import stable_hash
from .types import (
    SCHEMA_VERSION,
    H3MediaBundleType,
    H3VideoRefBundleType,
    display_card_label,
    validate_media_bundle,
    validate_video_ref_bundle,
)

logger = logging.getLogger("h3_studio.media_hub")


class H3StudioMediaHub(io.ComfyNode):
    """Unified character & reference media workbench (Images 0-8 + Audios 0-2)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="H3StudioMediaHub",
            display_name="H3 Studio 角色与素材台",
            category="H3 Studio",
            description=(
                "双 Tab 整合参考图 (0-9) 与参考音频 (0-3)，上方常驻已启用素材栏。"
                "导入素材默认显示去后缀文件名并支持自定义改名，直接输出强类型 H3_MEDIA_BUNDLE。"
            ),
            inputs=[
                io.String.Input(
                    "image_files",
                    default="[]",
                    tooltip="面板维护的参考图 JSON 列表（支持 skip、label、项目预设读写）。",
                ),
                io.String.Input(
                    "image_project",
                    default=DEFAULT_PROJECT,
                    tooltip="参考图项目预设库名称（对应 output/h3-lvm-image-refs/<project>）。",
                ),
                io.Float.Input(
                    "target_megapixels",
                    default=DEFAULT_MEGAPIXELS,
                    min=MIN_MEGAPIXELS,
                    max=MAX_MEGAPIXELS,
                    step=0.01,
                    round=0.01,
                    tooltip="每张参考图的目标像素量（MP），按比例对齐 32px 网格，不裁切。",
                ),
                io.Combo.Input(
                    "resampling_method",
                    options=list(RESAMPLE_METHODS),
                    default=DEFAULT_RESAMPLE_METHOD,
                    tooltip="参考图缩放重采样算法。",
                ),
                io.Boolean.Input(
                    "upscale_small_images",
                    default=DEFAULT_UPSCALE_SMALL_IMAGES,
                    label_on="放大小图至目标 MP",
                    label_off="不放大",
                    tooltip="关闭时小图不放大到目标 MP，仅对齐 32px 网格。",
                ),
                io.String.Input(
                    "audio_files",
                    default="[]",
                    tooltip="面板维护的参考音频 JSON 列表（支持 skip、label、start/duration 区间裁切）。",
                ),
                io.String.Input(
                    "audio_project",
                    default=DEFAULT_PROJECT,
                    tooltip="参考音频项目预设库名称（对应 output/h3-lvm-audio-refs/<project>）。",
                ),
            ],
            outputs=[
                H3MediaBundleType.Output(display_name="media_bundle"),
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        image_files="[]",
        image_project=DEFAULT_PROJECT,
        target_megapixels=DEFAULT_MEGAPIXELS,
        resampling_method=DEFAULT_RESAMPLE_METHOD,
        upscale_small_images=DEFAULT_UPSCALE_SMALL_IMAGES,
        audio_files="[]",
        audio_project=DEFAULT_PROJECT,
    ):
        try:
            parse_media_list(image_files, MAX_REF_IMAGES)
            parse_media_list(audio_files, MAX_REF_AUDIOS)
            clamp_megapixels(target_megapixels)
            clamp_resampling_method(resampling_method)
        except ValueError as exc:
            return str(exc)
        return True

    @classmethod
    def fingerprint_inputs(
        cls,
        image_files="[]",
        image_project=DEFAULT_PROJECT,
        target_megapixels=DEFAULT_MEGAPIXELS,
        resampling_method=DEFAULT_RESAMPLE_METHOD,
        upscale_small_images=DEFAULT_UPSCALE_SMALL_IMAGES,
        audio_files="[]",
        audio_project=DEFAULT_PROJECT,
    ):
        try:
            img_extra = repr((
                clamp_megapixels(target_megapixels),
                clamp_resampling_method(resampling_method),
                bool(upscale_small_images),
            ))
        except ValueError:
            img_extra = repr((target_megapixels, resampling_method, upscale_small_images))
        img_fp = _safe_fingerprint(image_files, MAX_REF_IMAGES, extra=img_extra)
        aud_fp = _safe_fingerprint(audio_files, MAX_REF_AUDIOS)
        return stable_hash({"images": img_fp, "audios": aud_fp})

    @classmethod
    def execute(
        cls,
        image_files: str = "[]",
        image_project: str = DEFAULT_PROJECT,
        target_megapixels: float = DEFAULT_MEGAPIXELS,
        resampling_method: str = DEFAULT_RESAMPLE_METHOD,
        upscale_small_images: bool = DEFAULT_UPSCALE_SMALL_IMAGES,
        audio_files: str = "[]",
        audio_project: str = DEFAULT_PROJECT,
    ) -> io.NodeOutput:
        del image_project, audio_project
        img_entries = active_entries(parse_media_list(image_files, MAX_REF_IMAGES))[:MAX_REF_IMAGES]
        aud_entries = active_entries(parse_media_list(audio_files, MAX_REF_AUDIOS))[:MAX_REF_AUDIOS]

        megapixels = clamp_megapixels(target_megapixels)
        method = clamp_resampling_method(resampling_method)
        upscale = bool(upscale_small_images)
        roots = _media_roots()

        images_out: List[Dict[str, Any]] = []
        for entry in img_entries:
            path = resolve_media_path(roots, entry)
            tensor, plan, source = build_image_slot(path, megapixels, method, upscale)
            fp = stable_hash({
                "path": path,
                "plan": plan,
                "method": method,
                "upscale": upscale,
                "shape": list(tensor.shape),
            })
            images_out.append({
                "tensor": tensor,
                "label": display_card_label(entry),
                "filename": entry["filename"],
                "plan": plan,
                "source_size": list(source),
                "fingerprint": fp,
            })

        audios_out: List[Dict[str, Any]] = []
        for entry in aud_entries:
            path = resolve_media_path(roots, entry)
            start, duration = trim_range(entry)
            audio = build_audio_slot(path, start, duration)
            wf = audio["waveform"]
            fp = stable_hash({
                "path": path,
                "trim": [start, duration],
                "shape": list(wf.shape),
                "sr": int(audio["sample_rate"]),
            })
            audios_out.append({
                "audio": audio,
                "label": display_card_label(entry),
                "filename": entry["filename"],
                "trim": (start, duration),
                "fingerprint": fp,
            })

        media_fp = stable_hash({
            "images": [item["fingerprint"] for item in images_out],
            "audios": [item["fingerprint"] for item in audios_out],
        })
        bundle: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "images": images_out,
            "audios": audios_out,
            "media_fingerprint": media_fp,
        }
        logger.info(
            "[H3 Studio MediaHub] active references: %d image(s), %d audio(s)",
            len(images_out), len(audios_out),
        )
        return io.NodeOutput(bundle)


class H3StudioBundleUnpack(io.ComfyNode):
    """Compatibility escape hatch: unpack Studio bundles into standard ComfyUI ports."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        outputs = [
            *(io.Image.Output(display_name=f"ref_image_{i}") for i in range(MAX_REF_IMAGES)),
            *(io.Audio.Output(display_name=f"ref_audio_{i}") for i in range(MAX_REF_AUDIOS)),
            io.Image.Output(display_name="ref_video_0"),
            io.Audio.Output(display_name="ref_video_audio_0"),
            io.Int.Output(display_name="frame_count"),
        ]
        return io.Schema(
            node_id="H3StudioBundleUnpack",
            display_name="H3 Studio 素材拆包器",
            category="H3 Studio/Compat",
            description="兼容逃生舱：将 H3_MEDIA_BUNDLE 与 H3_VIDEO_REF_BUNDLE 拆包为标准 ref_image_0..8 / ref_audio_0..2 / ref_video_0。",
            inputs=[
                H3MediaBundleType.Input("media_bundle", optional=True),
                H3VideoRefBundleType.Input("video_ref_bundle", optional=True),
            ],
            outputs=outputs,
        )

    @classmethod
    def execute(
        cls,
        media_bundle: Optional[Dict[str, Any]] = None,
        video_ref_bundle: Optional[Dict[str, Any]] = None,
    ) -> io.NodeOutput:
        img_slots: List[Any] = [None] * MAX_REF_IMAGES
        aud_slots: List[Any] = [None] * MAX_REF_AUDIOS
        if media_bundle is not None:
            mb = validate_media_bundle(media_bundle)
            for idx, item in enumerate(mb["images"][:MAX_REF_IMAGES]):
                img_slots[idx] = item.get("tensor")
            for idx, item in enumerate(mb["audios"][:MAX_REF_AUDIOS]):
                aud_slots[idx] = item.get("audio")

        ref_video = None
        ref_video_audio = None
        frame_count = 0
        if video_ref_bundle is not None:
            vb = validate_video_ref_bundle(video_ref_bundle)
            if vb["enabled"]:
                ref_video = vb["video"]
                ref_video_audio = vb["audio"]
                frame_count = int(vb["aligned_frame_count"] or 0)

        return io.NodeOutput(*img_slots, *aud_slots, ref_video, ref_video_audio, frame_count)


NODE_LIST = [H3StudioMediaHub, H3StudioBundleUnpack]
