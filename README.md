# ComfyUI-H3-Studio-Xiyun（H3 Studio 熙云版）

> **一站式 MiniMax H3 长视频接续、参考素材工作台、原生高速 3D 联合放大与四柱抗漂移引擎**  
> 将原 `H3-Long-Video-Manager-Xiyun` 与 `ComfyUI-H3-ClipStream-Xiyun` 深度融合并全面重构为统一 ComfyUI V3 插件（**37 节点 · 38 HTTP 路由 · 0 数据迁移**）。

[![Version](https://img.shields.io/badge/version-v0.1.0-blue.svg)](./VERSION.md)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-V3%20API-green.svg)](https://github.com/comfyanonymous/ComfyUI)
[![License](https://img.shields.io/badge/license-GPLv3-orange.svg)](./LICENSE)

---

## ✨ 核心特性一览

### 1. 强类型总线 + 2+2 对称工作台，彻底消灭 `SetNode`/`GetNode` 蜘蛛网
- 引入 **7 种带版本号校验的强类型自定义总线**（`H3_MEDIA_BUNDLE`、`H3_VIDEO_REF_BUNDLE`、`H3_CONTINUATION_PIPE`、`H3_ENCODE_ENV`、`H3_SAMPLING_PLAN`、`H3_COND_PACK`、`H3_PASS1_RESULT`），大幅压缩复杂接续工作流规模（实测从 **151 节点 / 171 连线 / 50+ Set/Get** 精简至 **44 节点 / 67 连线 / 0 Set/Get**）。
- 模型主干坚持明线可视化（UNET → 加速 LoRA → 显存与稀疏优化链 → 一采/二采独立 LoRA → `taeh3` 实时预览）。
- 全套 Studio 节点参数提供**中英双语标签**（如 `一采步数 (pass1_steps)`、`二采放大倍率 (upscale_factor)`），不改变底层字段名，兼容工作流保存与 API 调用。

### 2. 根治「进入二采前异常缓慢」的双重性能瓶颈 + 双层 1-Slot 智能缓存
- **全程仅跑 1 次 Qwen3-VL 32B 编码**：`H3StudioUnifiedEncoder` 单入口编码，`H3StudioPass2Refiner` 0.1ms 原地构建高分空模板、**0ms 直接复用一采 `positive`**（`match` 模式下也仅重编参考图 VAE，绝不重载 32B）。
- **原生高速 3D AV 联合放大器**：`H3StudioAVLatentUpscaler` 直接复用 ComfyUI 官方 `comfy.ldm.minimax.latent_upscaler.MiniMaxH3LatentUpscaler` + `CoreModelPatcher` + `load_models_gpu(..., memory_required=...)`，根治 Windows WDDM 共享显存换页暴慢；跑完通过 `unload_model_and_clones` **仅卸载放大器自身**不踢出主 DiT，且音画 `NestedTensor` 直进直出，消灭外部拆合节点。
- **双层 1-Slot 内容哈希缓存**：
  - 改 Seed / 步数 / 放大倍率 / 接续母片 → **0ms 命中 32B 编码缓存**；
  - 仅调二采参数 / 二采 LoRA / 抗漂移档位 → **0ms 命中一采采样缓存**；
  - 严格 1-Slot 主动释放旧张量引用，防内存膨胀。

### 3. 多段长视频接续「四柱抗漂移 + 极速接续」黄金算法（① + ② + ③ + ④）
全部内置集成，无需外挂零碎小节点，并在前端提供 **5 套一键预设** + **折叠双语高级参数抽屉**：

| 算法支柱 | 解决痛点 | 核心实现要点 |
| :--- | :--- | :--- |
| **① Flow-Matched 渐进置信度声明加噪** | 历史旧帧过强锁定导致画面发脏/色污染 | 按距接缝远近渐进混合加噪，并共享 `h3_strength_patch` 同步重标定 DiT AdaLN 时间步，绝不触碰音频流 |
| **② 二采尾步动态解除约束** | 低噪精修阶段死锁旧段肤色/色偏且拖慢注意力 | 按 `low_sigmas` 自动算释放阈值，最后 1~2 步卸载 Motion Context 视频 `cond` 行（专属键 `h3_mc_release_sigma` 隔离，**零误伤参考图/音频/Guide**） |
| **③ 非对称窗口极速接续 (`22→5`)** | 二采高分携带 22 帧历史上下文计算沉重 | 基于 H3 Video VAE `(1,4,4,4,4)` 周期不变量，二采 5 帧精准锚定在 `frame_offset=17`，音频终点与重叠恢复仍锁死 `22` 帧，**0ms 音画错位** |
| **④ 首段原色锚定校色 + 场景切换保护** | 5~10 段长链累积发灰、偏黄、显老 | 严格执行 `JoinMatch（未加噪父段） → Root-Anchor（Shot #1 24 通道基准） → OverlapRestore`；大运动/大转场自动通过 Scene-Shift Guard 衰减归零防误伤夜景/转场 |

#### 内置抗漂移预设一览
| 预设 | 一采 / 二采帧数 | 渐进加噪 (一采 / 二采) | 二采末尾解除步数 | 接缝校色 / 首段拉回 | 重叠恢复 (一采 / 二采) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`均衡护色 (推荐)`** | `22 / 22` | `0.82→0.99` / `0.85→0.995` | `1` 步 | `1.0` / `0.30` | 关 / 开 |
| **`强力护色`** | `22 / 22` | `0.74→0.97` / `0.78→0.98` | `2` 步 | `1.0` / `0.45` | 关 / 开 |
| **`极速接续 (二采5帧)`** | `22 / 5` | `0.82→0.99` / `0.88→0.995` | `1` 步 | `1.0` / `0.30` | 关 / 开 |
| **`关闭 (原始模式)`** | `22 / 22` | 关 / 关 | `0` 步 | `0.0` / `0.0` | 关 / 开 |
| **`自定义`** | 自由调整 | 自由调整 | `0~3` 步 | 自由调整 | 自由调整 |

### 4. 2+2 前端可视化工作台
- **🎨 H3 Studio 角色与素材台 (`H3StudioMediaHub`)**：
  - 双 Tab（参考图 0–9 + 参考音频 0–3）+ 顶部**常驻已启用素材栏**（悬停预览、点击跳转高亮、一键忽略）；
  - 完整支持**按住卡片拖拽排序**、`忽略 / 替换 / 删除` 与 `启用 / 删除`、默认去后缀文件名 + 双击改名（清空回退原名）、竖图模糊背景适配、真实输出像素预览、编辑防抖自动同步项目预设，以及跨图+音 **📸 组合快照**一键存取。
- **🔗 H3 Studio 接续与视频工作台 (`H3StudioContinuityHub`)**：
  - 双 Tab（Dual 生成接续库 + 参考视频/切片库）+ 顶部快捷 `参考视频: 开/关` 胶囊 + `同步参考时长` 开关 + 内置零浪费 `17n+5` 自动补帧；
  - **100% 保留手动选卡 / 删卡重抽 / 三模式（自动/首段/严格）**，`auto_advance_segment` 默认关闭，提供一键手动 **`⏭ 下一段并接续最新`**；
  - Dual 卡片显示「一采·原始 / 二采·裁切后」双路时长、血缘来源与悬停静音预览。

### 5. 100% 继承旧生态与磁盘资产（零数据迁移）
- 完整内置原 LVM **13 个节点** + ClipStream **15 个节点**（含 `MiniMaxClipBinLongBuilder` 长视频拼接器、`H3 Guide Timeline v2`、`H3 Long Video Manager v2`、`H3 Smart Split v2`、`H3 Video Compare Slider` 等）及全部 **36 条旧 HTTP 路由** + 2 条 Studio 新路由。
- 原有磁盘目录（`output/h3-lvm/`、`output/h3-lvm-video-uploads/`、`output/h3-clipstream/`、`input/h3-lvm-refs/`）开箱直接读取，无需迁移任何旧文件。

---

## 📦 安装与升级指南

### 1. 克隆到 `ComfyUI/custom_nodes/`
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/xiyunyun/H3-Studio-XiYun.git ComfyUI-H3-Studio-Xiyun
```

### 2. 安装可选依赖（仅 `H3 Smart Split` 场景切分需要）
```bash
pip install -r ComfyUI-H3-Studio-Xiyun/requirements.txt
```

### 3. ⚠️ 从旧版 LVM / ClipStream 升级必读（重要！）
由于本插件已全量包含旧版 28 个节点与 36 条路由，为避免冲突，请先将旧目录改名加上 `.disabled` 后缀作为冷备份（**不要删除磁盘 `output/` 和 `input/` 下的素材库！**）：
- `H3-Long-Video-Manager-Xiyun` → `H3-Long-Video-Manager-Xiyun.disabled`
- `ComfyUI-H3-ClipStream-Xiyun` → `ComfyUI-H3-ClipStream-Xiyun.disabled`

> 插件内置 `fork_guard.py` 冲突检测守卫：若检测到未禁用的旧版插件，会在 ComfyUI 启动控制台给出明确中文提示并安全跳过加载。

### 4. 重启 ComfyUI 并强刷浏览器
重启 ComfyUI 后，在浏览器页面按 **`Ctrl + F5`** 强制刷新前端脚本即可使用。

---

## 🧩 节点总览（共 37 个节点）

### 全新 Studio 核心节点（9 个）
| 节点 ID | 中文名称 | 作用 |
| :--- | :--- | :--- |
| `H3StudioMediaHub` | H3 Studio 角色与素材台 | 双 Tab 参考图 (0–9) + 参考音频 (0–3) 工作台 |
| `H3StudioContinuityHub` | H3 Studio 接续与视频工作台 | 双 Tab Dual 接续库 + 参考视频/切片库 + 抗漂移总控 |
| `H3StudioMasterConfig` | H3 Studio 一体化总控参数台 | 分辨率、时长、提示词、Seed、步数、倍率与 Sigma 精修总控 |
| `H3StudioUnifiedEncoder` | H3 Studio 统一条件编码器 | 单次 Qwen3-VL 32B 编码 + 1-Slot 缓存 |
| `H3StudioPass1Sampler` | H3 Studio 一采采样引擎 | 一采采样 + 1-Slot 缓存 + Guide v2 + 算法① |
| `H3StudioAVLatentUpscaler` | H3 Studio 联合 3D 放大器 | 原生内核音画联合 AV 3D 放大器（防 WDDM 换页） |
| `H3StudioPass2Refiner` | H3 Studio 二采精修与解码引擎 | 0ms 复用 32B + 四柱抗漂移 ①②③④ + AV 解码与对齐裁切 |
| `H3StudioDualClipSaver` | H3 Studio 双路归档器 | 兼容归档 + 自动写入首段色彩锚点 `latent_stats` |
| `H3StudioBundleUnpack` | H3 Studio 素材拆包器 | 将总线拆回标准 `ref_image_0..8` / `ref_audio_0..2` / `ref_video_0` |

### 完整继承的 28 个旧节点
- **LVM (13)**：`H3 Video Loader`、`H3 Long Video Manager`、`H3 Long Video Manager v2`、`H3 Smart Split`、`H3 Smart Split v2`、`H3 Segment Picker`、`H3 Image Reference Loader`、`H3 Audio Reference Loader`、`H3 Guide Timeline`、`H3 Guide Timeline v2`、`H3 Guide Timeline v2 B`、`H3 Guide Keyframe Release`、`H3 Video Compare Slider`
- **ClipStream (15)**：`MiniMaxClipBinLongBuilder`、`MiniMaxClipBinSaver`、`MiniMaxClipBinPicker`、`MiniMaxClipBinDualSaver`、`MiniMaxClipBinDualPicker`、`MiniMaxH3MotionContextClipStream`、`MiniMaxH3MotionContextTrimClipStream`、`MiniMaxH3MotionContextSaveLatentClipStream`、`MiniMaxH3MotionContextLoadLatentClipStream`、`MiniMaxH3MotionContextChainClipStream`、`MiniMaxH3MotionContextSeamProbeClipStream`、`MiniMaxH3OverlapRestoreClipStream`、`MiniMaxH3JoinMatchClipStream`、`MiniMaxH3RefineBlendClipStream`、`MiniMaxH3SigmasTailClipStream`

---

## 🧪 运行自动化测试

```powershell
# 1. Studio 后端单元测试 (18 tests)
$env:PYTHONPATH = "<ComfyUI>/custom_nodes/ComfyUI-H3-Studio-Xiyun;<ComfyUI>"
python -m unittest discover -s tests/studio -v

# 2. Studio + 继承前端状态机单元测试 (Node.js)
node --test tests/studio/studio_state.test.mjs
```

---

## 📜 开源协议与致谢

本项目整合了来自 [ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context)（GPLv3）、[ComfyUI-MiniMaxH3-PrefixStream](https://github.com/knoic/ComfyUI-MiniMaxH3-PrefixStream)（MIT）、`ComfyUI-H3-ClipStream` 与 `H3-Long-Video-Manager` 的开源成果，整体遵循 **GNU General Public License v3.0 (GPLv3)** 协议发布。详见 [LICENSE](./LICENSE) 与 [ATTRIBUTION.md](./ATTRIBUTION.md)。
