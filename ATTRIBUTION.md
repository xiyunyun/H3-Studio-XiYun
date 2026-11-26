# Attribution & Upstream Credits

`ComfyUI-H3-Studio-Xiyun` (`H3-Studio-XiYun`) integrates and builds upon open-source MiniMax H3 ComfyUI ecosystem projects, unified into a single next-generation Studio plugin with zero disk migration and full backwards compatibility.

Because GPLv3-licensed code is incorporated into this combined work, the repository as a whole is distributed under **GNU General Public License v3.0 (GPLv3)** (see [LICENSE](./LICENSE)).

## 1. H3 Studio Core Architecture & Workbenches (`studio/`, `web/studio/`)
- **Author**: 熙云 (Xiyun)
- **Original work**:
  - 7 strong-typed ComfyUI V3 custom buses (`H3_MEDIA_BUNDLE`, `H3_VIDEO_REF_BUNDLE`, `H3_CONTINUATION_PIPE`, `H3_ENCODE_ENV`, `H3_SAMPLING_PLAN`, `H3_COND_PACK`, `H3_PASS1_RESULT`)
  - 9 Studio core nodes (`H3StudioMediaHub`, `H3StudioContinuityHub`, `H3StudioMasterConfig`, `H3StudioUnifiedEncoder`, `H3StudioPass1Sampler`, `H3StudioAVLatentUpscaler`, `H3StudioPass2Refiner`, `H3StudioDualClipSaver`, `H3StudioBundleUnpack`)
  - Single-pass Qwen3-VL 32B encoding + dual 1-slot content-hash memory caches
  - Native-core `MiniMaxH3LatentUpscaler` + `CoreModelPatcher` joint AV 3D upscaler
  - 4-pillar anti-drift continuation engine (Flow-Matched Progressive Taper, Per-Step MC Release, Asymmetric `22→5` Seam-Aligned Fast Chaining, Root-Anchor Join Match + Scene-Shift Guard)
  - 2+2 Studio frontend workbenches, drag-and-drop reordering, media combo snapshots, and full bilingual parameter labels

## 2. Long Video Manager, Reference Loaders & Guide Timeline (`lvm/`, `web/lvm/`)
- **Source**: `H3-Long-Video-Manager-Xiyun` (MIT License)
- **Included modules**: Long Video Manager / v2, Smart Split / v2, Segment Picker, Image & Audio Reference Loaders, Guide Timeline / v2 / v2 B / Keyframe Release, Video Compare Slider, and `/h3_lvm/*` API routes.

## 3. ClipStream — Motion-Context & Clip Bin (`clipstream/`, `web/clipstream/`)
- **Source**: `ComfyUI-H3-ClipStream-Xiyun` (GPLv3), combining:
  - **Motion-Context continuation engine** (`clipstream/motion_context/`):
    - Upstream: [NikoDemon80/ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context) by **NikoDemon80** (**GPLv3**)
  - **Clip Bin visual media pool** (`clipstream/clipbin/`):
    - Upstream: [knoic/ComfyUI-MiniMaxH3-PrefixStream](https://github.com/knoic/ComfyUI-MiniMaxH3-PrefixStream) by **knoic** (**MIT**)
  - **Dual Saver / Dual Picker & Long Builder**:
    - Original work by **AraneaQwQ** (from `ComfyUI-H3-ClipStream`)
