# Changelog / 版本记录

## v0.1.0 (2026-11-26) — Initial Studio Release

### ✨ New Architecture & Core Nodes (9 Studio Nodes)
- **`H3StudioMediaHub`**：双 Tab 整合参考图 (0–9) 与参考音频 (0–3)，带常驻已启用素材栏（悬停预览/点击跳转/一键忽略）、卡片拖拽排序、`忽略 / 替换 / 删除` 与 `启用 / 删除` 按钮、去后缀默认命名与改名回退、竖图模糊背景适配、真实 `image_metadata + ref_plan` 像素规划、自动防抖同步项目预设，以及跨图+音 **📸 组合快照**（`/h3_studio/media_snapshots`）。
- **`H3StudioContinuityHub`**：双 Tab 整合 Dual 生成接续库与参考视频/切片库，内置 `17n+5` 零浪费补帧、首段原色血缘溯源与老卡 `latent_stats` 自动回填、5 套抗漂移预设 + 中英双语高级参数抽屉、100% 保留手动选卡/删卡重试/三模式、`auto_advance_segment` 默认关闭，以及一键手动 `⏭ 下一段并接续最新`。
- **`H3StudioMasterConfig`**：一体化总控参数台（画幅、MP、对齐倍数、时长自动 `17n+5`、前后置提示词、参考图模式、Seed、一二采步数、放大倍率、调度器、采样器与自包含 `H3SigmaRefiner` 低噪精修）。
- **`H3StudioUnifiedEncoder`**：全程单次 Qwen3-VL 32B 编码 + 1-Slot 内容哈希缓存；二采 0ms 复用 `positive`，改 Seed 不重跑 32B。
- **`H3StudioPass1Sampler`**：一采采样引擎 + 1-Slot 内容哈希缓存（感知 LoRA/优化链指纹），严格输出槽位 1 `denoised_output`。
- **`H3StudioAVLatentUpscaler`**：直接调用 ComfyUI 原生 `MiniMaxH3LatentUpscaler` + `CoreModelPatcher` + `load_models_gpu(..., memory_required=...)`，音画 AV `NestedTensor` 直进直出，跑完仅卸载自身不踢出 DiT，根治 Windows WDDM 共享显存换页暴慢。
- **`H3StudioPass2Refiner`**：0.1ms 原地高分模板 + 0ms 复用 32B + 内部 Reconcile 校验 + 四柱抗漂移算法（① Flow-Matched 渐进置信度声明加噪、② 二采尾步动态解除约束、③ `22→5` 偏移 17 极速接续且音频零错位、④ 首段原色锚定 JoinMatch + 场景切换保护 + `OverlapRestore` 严格顺序），锁定槽位 0 `output` 与双流音画对齐裁切。
- **`H3StudioDualClipSaver`**：100% 兼容旧库与 `MiniMaxClipBinLongBuilder`，不重复落盘一采 MP4，写入 `latent_stats` / `root_clip_id` / `chain_depth` 并通知前端自动刷新。
- **`H3StudioBundleUnpack`**：兼容拆包逃生舱。

### 🧱 Legacy Inheritance & Localization
- 无损迁入全部 **28 个旧节点**（13 LVM + 15 ClipStream）与 **36 条旧 HTTP 路由** + 新增 2 条 Studio 路由，总计 **37 节点 / 38 路由 / 0 冲突**。
- 内置 `fork_guard.py` 冲突自动检测，保护旧目录磁盘数据零迁移直连。
- 新增 `web/studio/studio_labels.js`，全套 Studio 节点参数列表统一显示**中英双语**（不改变底层序列化字段名）。
