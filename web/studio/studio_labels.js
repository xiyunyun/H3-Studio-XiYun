// H3 Studio Bilingual Widget Labels Extension
// Displays clear Chinese + English labels on all H3 Studio nodes without changing
// underlying widget names, preserving 100% workflow JSON & API compatibility.

import { app } from "../../../scripts/app.js";

const NODE_WIDGET_LABELS = {
    H3StudioMasterConfig: {
        aspect_ratio: "画幅比例 (aspect_ratio)",
        megapixels: "一采目标像素 MP (megapixels)",
        multiple: "分辨率对齐倍数 (multiple)",
        duration_seconds: "目标视频时长/秒 (duration_seconds)",
        custom_prompt: "前置触发词 / LoRA词 (custom_prompt)",
        prompt: "主提示词 (prompt)",
        ref_image_size: "参考图尺寸模式 (ref_image_size)",
        seed: "随机种子 (seed)",
        control_after_generate: "生成后种子变化 (control_after_generate)",
        pass1_steps: "一采步数 (pass1_steps)",
        pass2_steps: "二采步数 (pass2_steps)",
        upscale_factor: "二采放大倍率 (upscale_factor)",
        scheduler: "调度器 (scheduler)",
        sampler_name: "采样器 (sampler_name)",
        sigma_refiner_extra_steps: "低噪精修额外步数 (sigma_refiner_extra_steps)",
        sigma_refiner_start: "低噪精修起始阈值 (sigma_refiner_start)",
        sigma_refiner_end: "低噪精修结束阈值 (sigma_refiner_end)",
        sigma_refiner_spacing: "低噪精修插值曲线 (sigma_refiner_spacing)",
    },
    H3StudioMediaHub: {
        target_megapixels: "参考图目标像素 MP (target_megapixels)",
        resampling_method: "缩放重采样算法 (resampling_method)",
        upscale_small_images: "小图是否放大 (upscale_small_images)",
    },
    H3StudioContinuityHub: {
        video_output_size: "参考视频尺寸上限 (video_output_size)",
        video_custom_max_side: "参考视频自定义最长边 (video_custom_max_side)",
    },
    H3StudioPass1Sampler: {
        enable_pass1_guide: "启用一采关键帧引导 (enable_pass1_guide)",
    },
    H3StudioAVLatentUpscaler: {
        model_name: "3D 放大模型 (model_name)",
        align: "尺寸对齐倍数 (align)",
        precision: "计算精度 (precision)",
        release_after: "用完释放放大器显存 (release_after)",
    },
    H3StudioPass2Refiner: {
        enable_pass2_guide: "启用二采关键帧引导 (enable_pass2_guide)",
        extra_trim_frames: "对比流额外多裁帧数 (extra_trim_frames)",
    },
    H3StudioDualClipSaver: {
        prompt: "归档提示词 (prompt)",
        shot_tag: "镜头编号标签 (shot_tag)",
        save_video: "保存归档视频 (save_video)",
    },
};

function applyBilingualLabels(node, map) {
    if (!map || !Array.isArray(node?.widgets)) return;
    for (const widget of node.widgets) {
        const label = map[widget.name];
        if (label) widget.label = label;
    }
}

app.registerExtension({
    name: "H3.Studio.BilingualLabels",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        const map = NODE_WIDGET_LABELS[nodeData.name];
        if (!map) return;

        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = origCreated ? origCreated.apply(this, arguments) : undefined;
            applyBilingualLabels(this, map);
            return r;
        };

        const origConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = origConfigure ? origConfigure.apply(this, arguments) : undefined;
            applyBilingualLabels(this, map);
            return r;
        };
    },

    // Runs after ComfyUI core extensions attach `control_after_generate` widgets
    nodeCreated(node) {
        const map = NODE_WIDGET_LABELS[node?.comfyClass || node?.type];
        if (map) applyBilingualLabels(node, map);
    },

    loadedGraphNode(node) {
        const map = NODE_WIDGET_LABELS[node?.comfyClass || node?.type];
        if (map) applyBilingualLabels(node, map);
    },
});
