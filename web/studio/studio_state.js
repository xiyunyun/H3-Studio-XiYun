// Pure state & formatting helpers for ComfyUI-H3-Studio-Xiyun frontend panels.
// Zero DOM / API dependencies so every rule is unit-tested directly via `node --test`.

export const MAX_STUDIO_IMAGES = 9;
export const MAX_STUDIO_AUDIOS = 3;

export const PRESET_BALANCED = "均衡护色 (推荐)";
export const PRESET_STRONG = "强力护色";
export const PRESET_FAST = "极速接续 (二采5帧)";
export const PRESET_OFF = "关闭 (原始模式)";
export const PRESET_CUSTOM = "自定义";

export const ANTI_DRIFT_PRESET_SUMMARY = {
    [PRESET_BALANCED]: {
        code: "balanced",
        shortLabel: "均衡护色",
        badge: "22/22帧 · 末1步解除约束 · 首段校色0.3",
        p1Ctx: "22",
        p2Ctx: "22",
        p1Start: 0.82,
        p1Seam: 0.99,
        p2Start: 0.85,
        p2Seam: 0.995,
        releaseSteps: 1,
        joinMatch: 1.0,
        rootAnchor: 0.30,
        overlapP1: false,
        overlapP2: true,
    },
    [PRESET_STRONG]: {
        code: "strong",
        shortLabel: "强力护色",
        badge: "22/22帧 · 末2步解除约束 · 首段校色0.45",
        p1Ctx: "22",
        p2Ctx: "22",
        p1Start: 0.74,
        p1Seam: 0.97,
        p2Start: 0.78,
        p2Seam: 0.98,
        releaseSteps: 2,
        joinMatch: 1.0,
        rootAnchor: 0.45,
        overlapP1: false,
        overlapP2: true,
    },
    [PRESET_FAST]: {
        code: "fast",
        shortLabel: "极速接续",
        badge: "二采5帧提速 · 末1步解除约束 · 首段校色0.3",
        p1Ctx: "22",
        p2Ctx: "5",
        p1Start: 0.82,
        p1Seam: 0.99,
        p2Start: 0.88,
        p2Seam: 0.995,
        releaseSteps: 1,
        joinMatch: 1.0,
        rootAnchor: 0.30,
        overlapP1: false,
        overlapP2: true,
    },
    [PRESET_OFF]: {
        code: "off",
        shortLabel: "原始模式",
        badge: "22/22帧 · 仅保留重叠帧恢复",
        p1Ctx: "22",
        p2Ctx: "22",
        p1Start: 1.0,
        p1Seam: 1.0,
        p2Start: 1.0,
        p2Seam: 1.0,
        releaseSteps: 0,
        joinMatch: 0.0,
        rootAnchor: 0.0,
        overlapP1: false,
        overlapP2: true,
    },
    [PRESET_CUSTOM]: {
        code: "custom",
        shortLabel: "自定义",
        badge: "按下方高级参数运行",
    },
};

/**
 * Strip directory prefix and final extension from a filename.
 * Example: "dir/露帕-四视图.png" -> "露帕-四视图", "莫宁1.mp3" -> "莫宁1"
 */
export function stripFilenameExt(filename) {
    const normalized = String(filename ?? "").replace(/\\/g, "/").trim();
    if (!normalized) return "";
    const base = normalized.split("/").pop() || "";
    const dotIdx = base.lastIndexOf(".");
    if (dotIdx <= 0) return base;
    return base.slice(0, dotIdx);
}

/**
 * Resolve card title: custom label if non-empty, otherwise extension-stripped filename.
 * Never mutates entry so clearing custom label cleanly falls back to the filename stem.
 */
export function displayCardLabel(entry) {
    const custom = String(entry?.label ?? "").trim();
    if (custom) return custom;
    return stripFilenameExt(entry?.filename ?? entry?.original_name ?? "");
}

/**
 * Return a new entries array with card at `index` renamed.
 * Clearing `newLabel` removes the custom `label` key so it falls back to the filename stem.
 */
export function applyCardRename(entries, index, newLabel) {
    const list = Array.isArray(entries) ? entries.map(item => ({ ...item })) : [];
    if (index < 0 || index >= list.length) return list;
    const cleaned = String(newLabel ?? "").trim().replace(/\s+/g, " ");
    const entry = { ...list[index] };
    if (!cleaned) {
        delete entry.label;
    } else {
        entry.label = cleaned;
    }
    list[index] = entry;
    return list;
}

export function isEntrySkipped(entry) {
    return entry?.skip === true || entry?.skip === "true";
}

/**
 * Build the persistent Active Dock items (Plan 1A) in exact H3 presentation order:
 *   <Picture 1..9> followed by <Audio 1..3>
 */
export function buildActiveDockItems(imageEntries, audioEntries) {
    const images = [];
    const audios = [];
    let imgSlot = 0;
    for (let i = 0; i < (imageEntries ?? []).length; i++) {
        const entry = imageEntries[i];
        if (isEntrySkipped(entry)) continue;
        if (imgSlot >= MAX_STUDIO_IMAGES) break;
        imgSlot++;
        images.push({
            kind: "image",
            slotNumber: imgSlot,
            tag: `<Picture ${imgSlot}>`,
            sourceIndex: i,
            label: displayCardLabel(entry),
            entry,
        });
    }
    let audSlot = 0;
    for (let i = 0; i < (audioEntries ?? []).length; i++) {
        const entry = audioEntries[i];
        if (isEntrySkipped(entry)) continue;
        if (audSlot >= MAX_STUDIO_AUDIOS) break;
        audSlot++;
        const start = Number(entry?.start || 0);
        const duration = Number(entry?.duration || 0);
        const trimBadge = (start !== 0 || duration > 0)
            ? `${start >= 0 ? start.toFixed(1) : start.toFixed(1)}s${duration > 0 ? `+${duration.toFixed(1)}s` : "~尾"}`
            : "";
        audios.push({
            kind: "audio",
            slotNumber: audSlot,
            tag: `<Audio ${audSlot}>`,
            sourceIndex: i,
            label: displayCardLabel(entry),
            trimBadge,
            entry,
        });
    }
    return {
        images,
        audios,
        totalActive: images.length + audios.length,
    };
}

/**
 * Park (skip) a card directly from the persistent Active Dock without switching tabs.
 */
export function skipCardAtIndex(entries, sourceIndex) {
    const list = Array.isArray(entries) ? entries.map(item => ({ ...item })) : [];
    if (sourceIndex < 0 || sourceIndex >= list.length) return list;
    list[sourceIndex] = { ...list[sourceIndex], skip: true };
    const active = list.filter(e => !isEntrySkipped(e));
    const parked = list.filter(e => isEntrySkipped(e));
    return [...active, ...parked];
}

/**
 * Restore a parked card into active slots if capacity permits.
 */
export function restoreCardAtIndex(entries, sourceIndex, maxSlots) {
    const list = Array.isArray(entries) ? entries.map(item => ({ ...item })) : [];
    if (sourceIndex < 0 || sourceIndex >= list.length) {
        return { entries: list, restored: false, reason: "invalid_index" };
    }
    const activeCount = list.filter(e => !isEntrySkipped(e)).length;
    if (activeCount >= maxSlots) {
        return { entries: list, restored: false, reason: "full" };
    }
    const target = { ...list[sourceIndex] };
    delete target.skip;
    list.splice(sourceIndex, 1);
    const active = list.filter(e => !isEntrySkipped(e));
    const parked = list.filter(e => isEntrySkipped(e));
    return { entries: [...active, target, ...parked], restored: true, reason: "ok" };
}

/**
 * Manual one-click "⏭ 切至下一段并接续最新" state transition.
 * Never forces auto-advance on its own; strictly triggered by user click or explicit opt-in.
 */
export function computeNextSegmentStep({
    segmentId = 1,
    totalSegments = 0,
    clipMode = "Auto (首段全新 / 后续自动接续)",
    clipModeOptions = [
        "Auto (首段全新 / 后续自动接续)",
        "Force Initial (强制首段，无上下文)",
        "Strict Chaining (严格接续，空库报错)",
    ],
}) {
    const cur = Math.max(1, parseInt(segmentId, 10) || 1);
    const maxSeg = Math.max(0, parseInt(totalSegments, 10) || 0);
    const nextSeg = maxSeg > 0 ? Math.min(cur + 1, maxSeg) : cur + 1;
    const atEnd = maxSeg > 0 && cur >= maxSeg;

    let nextMode = clipMode;
    if (String(clipMode || "").startsWith("Force Initial")) {
        const autoOpt = clipModeOptions.find(opt => String(opt).startsWith("Auto"));
        if (autoOpt) nextMode = autoOpt;
    }
    return {
        segmentId: nextSeg,
        clipSelection: "latest",
        clipMode: nextMode,
        atEnd,
    };
}

/**
 * Determine whether post-execution auto-advance should fire.
 * Strictly FALSE unless user explicitly enabled `autoAdvanceSegment` AND `enableVideoRef`.
 */
export function shouldAutoAdvanceAfterRun({ autoAdvanceSegment, enableVideoRef, hadError = false }) {
    return Boolean(autoAdvanceSegment) && Boolean(enableVideoRef) && !hadError;
}

/**
 * Build concise labels for Continuity Hub's tabs and persistent status summary bar.
 */
export function continuityStatusSummary({
    clipMode = "",
    clipSelection = "latest",
    selectedClip = null,
    clipCount = 0,
    enableVideoRef = false,
    useVideoDuration = false,
    videoSourceMode = "segments",
    segmentId = 1,
    selectedSegment = null,
    preset = PRESET_BALANCED,
}) {
    const presetMeta = ANTI_DRIFT_PRESET_SUMMARY[preset] || ANTI_DRIFT_PRESET_SUMMARY[PRESET_BALANCED];
    const modeStr = String(clipMode || "");
    let clipBadge = "首段";
    let clipSummary = "✨ 首段全新生成";
    if (modeStr.startsWith("Force Initial")) {
        clipBadge = "强制首段";
        clipSummary = "✨ 强制首段（忽略历史）";
    } else if (clipCount === 0 && !selectedClip) {
        clipBadge = modeStr.startsWith("Strict") ? "空库报错" : "首段(空库)";
        clipSummary = modeStr.startsWith("Strict") ? "⚠️ 严格接续：当前库为空" : "✨ 首段（当前库为空）";
    } else {
        const tag = selectedClip?.shot_tag
            || (String(clipSelection || "latest").toLowerCase() === "latest" ? "最新镜头" : clipSelection);
        const depth = Number.isFinite(Number(selectedClip?.chain_depth))
            ? ` · 第${Number(selectedClip.chain_depth) + 2}段`
            : "";
        clipBadge = `接续 ${tag}`;
        clipSummary = `🔗 接续: ${tag}${depth} · ${presetMeta.shortLabel}`;
    }

    let videoBadge = "⚪关闭";
    let videoSummary = "🎬 参考视频: 关";
    if (enableVideoRef) {
        const segLabel = videoSourceMode === "uploads"
            ? (selectedSegment ? displayCardLabel(selectedSegment) : `视频 #${segmentId}`)
            : `片段 #${segmentId}`;
        const frameNum = selectedSegment?.frames ?? selectedSegment?.frame_count;
        const framesTxt = frameNum ? ` · ${frameNum}f` : "";
        const durTxt = useVideoDuration ? " · 同步时长" : "";
        videoBadge = `🟢开 #${segmentId}`;
        videoSummary = `🎬 ${segLabel}${framesTxt}${durTxt}`;
    }

    return {
        clipBadge,
        clipSummary,
        videoBadge,
        videoSummary,
        presetBadge: presetMeta.badge,
    };
}
