// H3 Studio Continuity & Video Ref Hub (Plan X):
// - Tab 1: Dual Clip Bin (100% preserves manual card selection, delete/re-roll, 3 modes,
//          dual 一采/二采 badges, hover preview, modal playback, and project switching)
// - Tab 2: Segment / Uploaded Video Ref Picker (enable toggle, duration sync, opt-in
//          auto-advance default OFF, full preview & delete support)
// - Persistent status bar + one-click manual "⏭ 下一段并接续最新" button
// - Compact anti-drift preset selector + collapsible advanced parameters drawer

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { addPanel } from "../lvm/dom_panel.js";
import { createCardZoom } from "../lvm/card_zoom.js";
import { createProjectMenu as createLvmProjectMenu } from "../lvm/project_menu.js";
import { openPreview as openLvmPreview } from "../lvm/preview_overlay.js";
import { addDeleteButton as addLvmDeleteButton } from "../lvm/delete_button.js";
import { appendThumbnail } from "../lvm/thumb_fit.js";
import { canBrowserPreview, formatDuration, uploadInfoLines } from "../lvm/picker_mode_state.js";
import { createProjectMenu as createClipProjectMenu } from "../clipstream/project_menu.js";
import { addDeleteButton as addClipDeleteButton } from "../clipstream/delete_button.js";
import {
    ANTI_DRIFT_PRESET_SUMMARY,
    PRESET_BALANCED,
    PRESET_CUSTOM,
    PRESET_FAST,
    PRESET_OFF,
    PRESET_STRONG,
    computeNextSegmentStep,
    continuityStatusSummary,
    displayCardLabel,
    shouldAutoAdvanceAfterRun,
} from "./studio_state.js";

try {
    const sheets = [
        ["h3lvm-styles", "../lvm/h3lvm_picker.css"],
        ["minimax-clip-bin-styles", "../clipstream/clip_bin_picker.css"],
        ["h3st-cont-styles", "./studio_continuity_hub.css"],
    ];
    for (const [id, rel] of sheets) {
        if (document.getElementById(id)) continue;
        const link = document.createElement("link");
        link.id = id;
        link.rel = "stylesheet";
        link.type = "text/css";
        link.href = new URL(rel, import.meta.url).href;
        document.head.appendChild(link);
    }
} catch (err) {
    console.warn("[H3 Studio ContinuityHub] CSS injection failed:", err);
}

const NODE_ID = "H3StudioContinuityHub";
const PRESET_ORDER = [PRESET_BALANCED, PRESET_STRONG, PRESET_FAST, PRESET_OFF, PRESET_CUSTOM];

const PRESET_FIELD_MAP = {
    pass1_context_length: "p1Ctx",
    pass2_context_length: "p2Ctx",
    p1_taper_start: "p1Start",
    p1_taper_seam: "p1Seam",
    p2_taper_start: "p2Start",
    p2_taper_seam: "p2Seam",
    p2_release_last_steps: "releaseSteps",
    join_match_strength: "joinMatch",
    root_anchor_strength: "rootAnchor",
    overlap_restore_p1: "overlapP1",
    overlap_restore_p2: "overlapP2",
};

function dualVariantName(label, clip) {
    if (label === "一采") return "一采·原始";
    if (label === "二采") {
        const parent = String(clip?.parent_clip_id || "").trim();
        const initial = ["[INITIAL]", "[INITIAL_GENERATION]"].includes(parent.toUpperCase());
        if (initial) return "二采·原始";
        if (parent) return "二采·裁切后";
    }
    return label;
}

function hideWidget(widget) {
    if (!widget) return;
    widget.options = widget.options || {};
    widget.options.hidden = true;
    widget.computeSize = () => [0, -4];
    widget.type = "hidden";
}

function setWidgetValue(widget, value) {
    if (!widget) return;
    widget.value = value;
    widget.callback?.(value);
}

app.registerExtension({
    name: "H3.Studio.ContinuityHub",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_ID) return;

        nodeType.prototype.min_size = [520, 460];

        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = origCreated ? origCreated.apply(this, arguments) : undefined;
            this.imgs = null;
            try {
                installContinuityHub(this);
                this.setSize([Math.max(this.size[0], 740), Math.max(this.size[1], 660)]);
            } catch (err) {
                console.warn("[H3 Studio ContinuityHub] setup failed (node still works):", err);
            }
            return r;
        };

        const origConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = origConfigure ? origConfigure.apply(this, arguments) : undefined;
            this.imgs = null;
            this._h3StudioContRefresh?.();
            return r;
        };

        const origExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const r = origExecuted ? origExecuted.apply(this, arguments) : undefined;
            this.imgs = null;
            this._h3StudioContExecuted?.(message);
            return r;
        };

        const origRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            this._h3StudioContLifetime?.abort();
            return origRemoved ? origRemoved.apply(this, arguments) : undefined;
        };

        nodeType.prototype.setSizeForImage = function () {};
    },
});

function installContinuityHub(node) {
    const lifetime = node._h3StudioContLifetime ?? new AbortController();
    node._h3StudioContLifetime = lifetime;
    const signal = lifetime.signal;

    const w = (name) => node.widgets?.find(item => item.name === name);
    const clipProjW = w("clip_project");
    const clipModeW = w("clip_mode");
    const clipSelW = w("clip_selection");

    const enVidW = w("enable_video_ref");
    const useDurW = w("use_video_duration");
    const vidModeW = w("video_source_mode");
    const vidProjW = w("video_project");
    const segIdW = w("segment_id");
    const vidSizeW = w("video_output_size");
    const vidCustomW = w("video_custom_max_side");
    const autoAdvW = w("auto_advance_segment");

    const presetW = w("anti_drift_preset");
    const advWidgets = {
        pass1_context_length: w("pass1_context_length"),
        pass2_context_length: w("pass2_context_length"),
        audio_context_length: w("audio_context_length"),
        enable_audio_context: w("enable_audio_context"),
        p1_taper_start: w("p1_taper_start"),
        p1_taper_seam: w("p1_taper_seam"),
        p2_taper_start: w("p2_taper_start"),
        p2_taper_seam: w("p2_taper_seam"),
        p2_release_last_steps: w("p2_release_last_steps"),
        join_match_strength: w("join_match_strength"),
        root_anchor_strength: w("root_anchor_strength"),
        overlap_restore_p1: w("overlap_restore_p1"),
        overlap_restore_p2: w("overlap_restore_p2"),
    };

    for (const item of [
        clipProjW, clipModeW, clipSelW,
        enVidW, useDurW, vidModeW, vidProjW, segIdW, autoAdvW,
        presetW, ...Object.values(advWidgets),
    ]) {
        hideWidget(item);
    }
    if (vidSizeW) vidSizeW.label = "参考视频尺寸上限 (video_output_size)";
    if (vidCustomW) vidCustomW.label = "参考视频自定义最长边 (video_custom_max_side)";

    let activeTab = "clips"; // "clips" | "video"
    let drawerOpen = false;
    let cachedClips = [];
    let cachedSegments = [];
    let clipRequestId = 0;
    let vidRequestId = 0;
    const clipPreviewCleanups = new Set();

    function releaseClipPreviews() {
        for (const fn of clipPreviewCleanups) fn();
        clipPreviewCleanups.clear();
    }
    signal.addEventListener("abort", releaseClipPreviews, { once: true });

    // --- DOM Root ---
    const root = document.createElement("div");
    root.className = "h3st-cont-root";

    // 1. Top 2-Tab Bar
    const tabBar = document.createElement("div");
    tabBar.className = "h3st-cont-tabs";

    const clipTabBtn = document.createElement("button");
    clipTabBtn.type = "button";
    clipTabBtn.className = "h3st-cont-tab active";
    const clipTabTitle = document.createElement("span");
    clipTabTitle.textContent = "🔗 生成接续库";
    const clipTabBadge = document.createElement("span");
    clipTabBadge.className = "h3st-cont-tab-badge";
    clipTabBtn.append(clipTabTitle, clipTabBadge);

    const vidTabBtn = document.createElement("button");
    vidTabBtn.type = "button";
    vidTabBtn.className = "h3st-cont-tab";
    const vidTabTitle = document.createElement("span");
    vidTabTitle.textContent = "🎬 参考视频";
    const vidTabBadge = document.createElement("span");
    vidTabBadge.className = "h3st-cont-tab-badge";
    vidTabBtn.append(vidTabTitle, vidTabBadge);

    const quickVidToggle = document.createElement("button");
    quickVidToggle.type = "button";
    quickVidToggle.className = "h3st-cont-quick-toggle";
    quickVidToggle.title = "一键开启/关闭视频参考（原「1 开启/关闭视频参考」开关）";

    tabBar.append(clipTabBtn, vidTabBtn, quickVidToggle);
    root.appendChild(tabBar);

    // 2. Persistent Status Summary Bar (never hides!)
    const statusBar = document.createElement("div");
    statusBar.className = "h3st-cont-status-bar";

    const statusLeft = document.createElement("div");
    statusLeft.className = "h3st-cont-status-left";
    const statusThumb = document.createElement("img");
    statusThumb.className = "h3st-cont-status-thumb";
    statusThumb.style.display = "none";
    const clipStatusChip = document.createElement("span");
    clipStatusChip.className = "h3st-cont-status-chip";
    statusLeft.append(statusThumb, clipStatusChip);

    const statusRight = document.createElement("div");
    statusRight.className = "h3st-cont-status-right";
    const vidStatusChip = document.createElement("span");
    vidStatusChip.className = "h3st-cont-status-chip muted";
    const nextStepBtn = document.createElement("button");
    nextStepBtn.type = "button";
    nextStepBtn.className = "h3st-cont-next-btn";
    nextStepBtn.textContent = "⏭ 下一段并接续最新";
    nextStepBtn.title = "手动切到下一参考视频片段，并将接续源切回最新成片（删卡重抽时不会自动跳段）";
    statusRight.append(vidStatusChip, nextStepBtn);

    statusBar.append(statusLeft, statusRight);
    root.appendChild(statusBar);

    // 3. Anti-drift Preset Bar + Collapsible Advanced Drawer
    const adBar = document.createElement("div");
    adBar.className = "h3st-cont-ad-bar";
    const adRow = document.createElement("div");
    adRow.className = "h3st-cont-ad-row";

    const adPresetBtns = new Map();
    for (const pName of PRESET_ORDER) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "h3st-cont-ad-pill";
        btn.textContent = ANTI_DRIFT_PRESET_SUMMARY[pName].shortLabel;
        btn.title = `${pName}：${ANTI_DRIFT_PRESET_SUMMARY[pName].badge}`;
        btn.onclick = () => {
            setWidgetValue(presetW, pName);
            syncAntiDriftUI();
            syncStatusBar();
            node.setDirtyCanvas?.(true, true);
        };
        adPresetBtns.set(pName, btn);
        adRow.appendChild(btn);
    }

    const adBadge = document.createElement("span");
    adBadge.className = "h3st-cont-ad-badge";
    const adToggleBtn = document.createElement("button");
    adToggleBtn.type = "button";
    adToggleBtn.className = "h3st-cont-btn";
    adToggleBtn.textContent = "⚙ 高级参数";
    adToggleBtn.onclick = () => {
        drawerOpen = !drawerOpen;
        adDrawer.style.display = drawerOpen ? "grid" : "none";
    };
    adRow.append(adBadge, adToggleBtn);
    adBar.appendChild(adRow);

    const adDrawer = document.createElement("div");
    adDrawer.className = "h3st-cont-ad-drawer";
    adDrawer.style.display = "none";

    function copyCurrentPresetToCustomWidgets() {
        const curPreset = String(presetW?.value || PRESET_BALANCED);
        if (curPreset === PRESET_CUSTOM) return;
        const meta = ANTI_DRIFT_PRESET_SUMMARY[curPreset];
        if (!meta) return;
        for (const [wKey, metaKey] of Object.entries(PRESET_FIELD_MAP)) {
            if (meta[metaKey] !== undefined && advWidgets[wKey]) {
                advWidgets[wKey].value = meta[metaKey];
            }
        }
    }

    const advFieldDefs = [
        { key: "pass1_context_length", label: "一采参考帧数 (p1_ctx)", type: "select", opts: ["22", "5", "39", "56"] },
        { key: "pass2_context_length", label: "二采参考帧数 (p2_ctx)", type: "select", opts: ["22", "5", "39", "56"] },
        { key: "audio_context_length", label: "音频参考帧数 (audio_ctx)", type: "number", step: "1", min: "0", max: "240" },
        { key: "enable_audio_context", label: "启用音频接续 (audio_on)", type: "checkbox" },
        { key: "p1_taper_start", label: "一采旧帧权重 (p1_start)", type: "number", step: "0.01", min: "0.1", max: "1" },
        { key: "p1_taper_seam", label: "一采接缝权重 (p1_seam)", type: "number", step: "0.01", min: "0.1", max: "1" },
        { key: "p2_taper_start", label: "二采旧帧权重 (p2_start)", type: "number", step: "0.01", min: "0.1", max: "1" },
        { key: "p2_taper_seam", label: "二采接缝权重 (p2_seam)", type: "number", step: "0.005", min: "0.1", max: "1" },
        { key: "p2_release_last_steps", label: "二采末尾解除步数 (release)", type: "number", step: "1", min: "0", max: "3" },
        { key: "join_match_strength", label: "接缝色彩对齐 (join_match)", type: "number", step: "0.05", min: "0", max: "1.5" },
        { key: "root_anchor_strength", label: "首段原色拉回 (root_anchor)", type: "number", step: "0.05", min: "0", max: "0.8" },
        { key: "overlap_restore_p1", label: "一采重叠帧恢复 (ov_p1)", type: "checkbox" },
        { key: "overlap_restore_p2", label: "二采重叠帧恢复 (ov_p2)", type: "checkbox" },
    ];
    const advControls = new Map();
    for (const def of advFieldDefs) {
        const wrap = document.createElement("label");
        wrap.className = "h3st-cont-ad-field";
        const lbl = document.createElement("span");
        lbl.textContent = def.label;
        let ctrl;
        if (def.type === "select") {
            ctrl = document.createElement("select");
            for (const o of def.opts) {
                const opt = document.createElement("option");
                opt.value = o;
                opt.textContent = o;
                ctrl.appendChild(opt);
            }
            ctrl.onchange = () => {
                copyCurrentPresetToCustomWidgets();
                setWidgetValue(presetW, PRESET_CUSTOM);
                setWidgetValue(advWidgets[def.key], ctrl.value);
                syncAntiDriftUI();
                syncStatusBar();
            };
        } else if (def.type === "checkbox") {
            ctrl = document.createElement("input");
            ctrl.type = "checkbox";
            ctrl.onchange = () => {
                if (def.key !== "enable_audio_context") {
                    copyCurrentPresetToCustomWidgets();
                    setWidgetValue(presetW, PRESET_CUSTOM);
                }
                setWidgetValue(advWidgets[def.key], ctrl.checked);
                syncAntiDriftUI();
                syncStatusBar();
            };
        } else {
            ctrl = document.createElement("input");
            ctrl.type = "number";
            ctrl.step = def.step;
            ctrl.min = def.min;
            ctrl.max = def.max;
            ctrl.onchange = () => {
                if (def.key !== "audio_context_length") {
                    copyCurrentPresetToCustomWidgets();
                    setWidgetValue(presetW, PRESET_CUSTOM);
                }
                const val = def.step === "1" ? parseInt(ctrl.value, 10) : parseFloat(ctrl.value);
                if (Number.isFinite(val)) setWidgetValue(advWidgets[def.key], val);
                syncAntiDriftUI();
                syncStatusBar();
            };
        }
        advControls.set(def.key, { ctrl, def });
        wrap.append(lbl, ctrl);
        adDrawer.appendChild(wrap);
    }
    adBar.appendChild(adDrawer);
    root.appendChild(adBar);

    // 4. Tab 1 Pane: Dual Clip Bin
    const clipPane = document.createElement("div");
    clipPane.className = "h3st-cont-pane";

    const clipProjMenu = createClipProjectMenu({
        node,
        widget: clipProjW,
        signal,
        mode: "picker",
        assetKind: "dual",
    });
    clipPane.appendChild(clipProjMenu.element);

    const clipToolbar = document.createElement("div");
    clipToolbar.className = "h3st-cont-toolbar";

    const clipModeDefs = [
        { prefix: "Auto", label: "🔁 自动" },
        { prefix: "Force Initial", label: "🎬 强制首段" },
        { prefix: "Strict Chaining", label: "🔗 严格接续" },
    ];
    const clipModeBtns = new Map();
    for (const def of clipModeDefs) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "h3st-cont-btn";
        btn.textContent = def.label;
        btn.onclick = () => {
            const opts = Array.isArray(clipModeW?.options?.values) ? clipModeW.options.values : [];
            const match = opts.find(v => String(v).startsWith(def.prefix)) || def.prefix;
            setWidgetValue(clipModeW, match);
            syncClipModeButtons();
            syncStatusBar();
            node.setDirtyCanvas?.(true, true);
        };
        clipModeBtns.set(def.prefix, btn);
        clipToolbar.appendChild(btn);
    }

    const clipRefreshBtn = document.createElement("button");
    clipRefreshBtn.type = "button";
    clipRefreshBtn.className = "h3st-cont-btn";
    clipRefreshBtn.textContent = "🔄 刷新";
    clipRefreshBtn.onclick = () => void loadClips();

    const clipZoomMount = document.createElement("span");
    createCardZoom({ container: clipPane, node, baseCardMin: 150, baseThumbH: 90 }).mount(clipZoomMount);
    clipToolbar.append(clipRefreshBtn, clipZoomMount);
    clipPane.appendChild(clipToolbar);

    const clipDeck = document.createElement("div");
    clipDeck.className = "h3st-cont-deck";
    const clipGrid = document.createElement("div");
    clipGrid.className = "h3st-cont-grid";
    clipDeck.appendChild(clipGrid);
    clipPane.appendChild(clipDeck);
    root.appendChild(clipPane);

    // 5. Tab 2 Pane: Video Ref / Segment Picker
    const vidPane = document.createElement("div");
    vidPane.className = "h3st-cont-pane";
    vidPane.style.display = "none";

    const vidMenuSlot = document.createElement("div");
    vidPane.appendChild(vidMenuSlot);
    let vidMenu = null;
    let vidMenuAbort = null;

    const vidToolbar = document.createElement("div");
    vidToolbar.className = "h3st-cont-toolbar";

    const segModeBtn = document.createElement("button");
    segModeBtn.type = "button";
    segModeBtn.className = "h3st-cont-btn";
    segModeBtn.textContent = "🎞 裁切片段";

    const upModeBtn = document.createElement("button");
    upModeBtn.type = "button";
    upModeBtn.className = "h3st-cont-btn";
    upModeBtn.textContent = "📹 投放视频";

    const useDurLabel = document.createElement("label");
    useDurLabel.className = "h3st-cont-check";
    useDurLabel.title = "原「2 以参考视频时长为目标视频时长」开关：勾选后按参考视频长度覆盖手动时长，并自动 17n+5 补帧";
    const useDurCheck = document.createElement("input");
    useDurCheck.type = "checkbox";
    useDurLabel.append(useDurCheck, document.createTextNode("同步参考时长"));

    const autoAdvLabel = document.createElement("label");
    autoAdvLabel.className = "h3st-cont-check";
    autoAdvLabel.title = "默认关闭！勾选后仅在生成成功完成时自动 segment_id + 1";
    const autoAdvCheck = document.createElement("input");
    autoAdvCheck.type = "checkbox";
    autoAdvLabel.append(autoAdvCheck, document.createTextNode("成功后自动下一段"));

    const vidUploadBtn = document.createElement("button");
    vidUploadBtn.type = "button";
    vidUploadBtn.className = "h3st-cont-btn";
    vidUploadBtn.textContent = "＋ 载入视频";
    vidUploadBtn.style.display = "none";

    const vidFileInput = document.createElement("input");
    vidFileInput.type = "file";
    vidFileInput.multiple = true;
    vidFileInput.accept = "video/*,.mp4,.mov,.mkv,.webm,.avi";
    vidFileInput.style.display = "none";
    root.appendChild(vidFileInput);

    const vidRefreshBtn = document.createElement("button");
    vidRefreshBtn.type = "button";
    vidRefreshBtn.className = "h3st-cont-btn";
    vidRefreshBtn.textContent = "🔄 刷新";

    const vidZoomMount = document.createElement("span");
    createCardZoom({ container: vidPane, node, baseCardMin: 150, baseThumbH: 90 }).mount(vidZoomMount);

    vidToolbar.append(
        segModeBtn, upModeBtn, useDurLabel, autoAdvLabel, vidUploadBtn, vidRefreshBtn, vidZoomMount
    );
    vidPane.appendChild(vidToolbar);

    const vidDeck = document.createElement("div");
    vidDeck.className = "h3st-cont-deck";
    const vidGrid = document.createElement("div");
    vidGrid.className = "h3st-cont-grid";
    vidDeck.appendChild(vidGrid);
    vidPane.appendChild(vidDeck);
    root.appendChild(vidPane);

    // --- Sync Helpers ---
    function syncClipModeButtons() {
        const cur = String(clipModeW?.value || "");
        for (const [prefix, btn] of clipModeBtns.entries()) {
            btn.classList.toggle("active", cur.startsWith(prefix));
        }
    }

    function syncAntiDriftUI() {
        const curPreset = String(presetW?.value || PRESET_BALANCED);
        const meta = ANTI_DRIFT_PRESET_SUMMARY[curPreset] || ANTI_DRIFT_PRESET_SUMMARY[PRESET_BALANCED];
        for (const [pName, btn] of adPresetBtns.entries()) {
            btn.classList.toggle("active", pName === curPreset);
        }
        adBadge.textContent = meta.badge;
        const isCustom = curPreset === PRESET_CUSTOM;
        adDrawer.classList.toggle("locked", !isCustom);
        for (const [key, { ctrl, def }] of advControls.entries()) {
            const presetKey = PRESET_FIELD_MAP[key];
            const val = (!isCustom && presetKey && meta[presetKey] !== undefined)
                ? meta[presetKey]
                : advWidgets[key]?.value;
            if (def.type === "checkbox") {
                ctrl.checked = Boolean(val);
            } else {
                ctrl.value = String(val ?? "");
            }
        }
    }

    function syncVideoControls() {
        const en = Boolean(enVidW?.value);
        quickVidToggle.classList.toggle("on", en);
        quickVidToggle.textContent = en ? "🟢 参考视频: 开" : "⚪ 参考视频: 关";
        useDurCheck.checked = Boolean(useDurW?.value);
        autoAdvCheck.checked = Boolean(autoAdvW?.value);
        const mode = String(vidModeW?.value || "segments");
        segModeBtn.classList.toggle("active", mode === "segments");
        upModeBtn.classList.toggle("active", mode === "uploads");
        vidUploadBtn.style.display = mode === "uploads" ? "" : "none";
    }

    function syncStatusBar() {
        const selVal = String(clipSelW?.value || "latest").trim();
        const isAutoSel = !selVal || selVal.toLowerCase() === "latest" || selVal.toLowerCase() === "auto";
        const selectedClip = isAutoSel
            ? (cachedClips[0] || null)
            : (cachedClips.find(c => c.clip_id === selVal) || null);
        const segId = parseInt(segIdW?.value, 10) || 1;
        const selectedSeg = cachedSegments.find(s => Number(s.segment_id) === segId) || null;

        const summary = continuityStatusSummary({
            clipMode: clipModeW?.value,
            clipSelection: selVal,
            selectedClip,
            clipCount: cachedClips.length,
            enableVideoRef: Boolean(enVidW?.value),
            useVideoDuration: Boolean(useDurW?.value),
            videoSourceMode: String(vidModeW?.value || "segments"),
            segmentId: segId,
            selectedSegment: selectedSeg,
            preset: String(presetW?.value || PRESET_BALANCED),
        });

        clipTabBadge.textContent = summary.clipBadge;
        vidTabBadge.textContent = summary.videoBadge;
        clipStatusChip.textContent = summary.clipSummary;
        vidStatusChip.textContent = summary.videoSummary;

        if (selectedClip?.thumbnail_url && !String(clipModeW?.value || "").startsWith("Force Initial")) {
            statusThumb.src = selectedClip.thumbnail_url;
            statusThumb.style.display = "";
        } else {
            statusThumb.style.display = "none";
        }
    }

    function switchTab(tab) {
        activeTab = tab;
        clipTabBtn.classList.toggle("active", tab === "clips");
        vidTabBtn.classList.toggle("active", tab === "video");
        clipPane.style.display = tab === "clips" ? "flex" : "none";
        vidPane.style.display = tab === "video" ? "flex" : "none";
        if (tab === "clips") void loadClips();
        else void loadVideoCards();
    }

    // --- Tab 1: Load Dual Clips ---
    function openClipVideoModal(clip) {
        if (!clip?.video_url) return;
        const existing = document.getElementById("h3st-clip-modal");
        if (existing) existing._h3Close?.();
        const overlay = document.createElement("div");
        overlay.id = "h3st-clip-modal";
        overlay.className = "minimax-video-modal-overlay";
        const modal = document.createElement("div");
        modal.className = "minimax-video-modal";
        const video = document.createElement("video");
        video.className = "minimax-modal-video";
        video.src = clip.video_url;
        video.controls = true;
        video.autoplay = true;
        video.playsInline = true;
        const bar = document.createElement("div");
        bar.style.cssText = "display:flex;justify-content:space-between;align-items:center;padding:8px 12px;";
        const titleEl = document.createElement("strong");
        titleEl.textContent = `🎬 ${clip.shot_tag || clip.clip_id}`;
        const closeBtn = document.createElement("button");
        closeBtn.type = "button";
        closeBtn.className = "h3st-cont-btn";
        closeBtn.textContent = "✕ 关闭";

        function closeModal() {
            video.pause();
            video.removeAttribute("src");
            video.load();
            overlay.remove();
            window.removeEventListener("keydown", onKeyDown);
            signal.removeEventListener("abort", closeModal);
        }
        function onKeyDown(e) { if (e.key === "Escape") closeModal(); }
        overlay._h3Close = closeModal;
        window.addEventListener("keydown", onKeyDown);
        signal.addEventListener("abort", closeModal, { once: true });
        closeBtn.onclick = closeModal;
        overlay.onclick = (e) => { if (e.target === overlay) closeModal(); };

        bar.append(titleEl, closeBtn);
        modal.append(bar, video);
        overlay.appendChild(modal);
        document.body.appendChild(overlay);
    }

    async function loadClips(deletedId = null) {
        if (signal.aborted) return;
        const req = ++clipRequestId;
        syncClipModeButtons();
        await clipProjMenu.refresh();
        if (signal.aborted || req !== clipRequestId) return;
        const project = clipProjW?.value || "Default_Project";
        if (deletedId && clipSelW?.value === deletedId) {
            setWidgetValue(clipSelW, "latest");
        }
        const curSel = String(clipSelW?.value || "latest").trim();

        try {
            const res = await api.fetchApi(
                `/minimax/clip_bin/list?project=${encodeURIComponent(project)}&asset_kind=dual`,
                { cache: "no-store" },
            );
            if (signal.aborted || req !== clipRequestId) return;
            const data = res.ok ? await res.json().catch(() => ({ clips: [] })) : { clips: [] };
            cachedClips = Array.isArray(data.clips) ? data.clips : [];
        } catch (_) {
            if (signal.aborted || req !== clipRequestId) return;
            cachedClips = [];
        }

        releaseClipPreviews();
        clipGrid.innerHTML = "";

        const isAutoActive = !curSel || curSel.toLowerCase() === "latest" || curSel.toLowerCase() === "auto";
        const autoCard = document.createElement("div");
        autoCard.className = `h3st-cont-card auto-card${isAutoActive ? " active" : ""}`;
        autoCard.innerHTML = `
            <div class="h3st-cont-card-thumb" style="font-size:26px;">⚡${isAutoActive ? '<span class="h3st-cont-card-badge">当前</span>' : ""}</div>
            <div class="h3st-cont-card-body no-delete">
                <div class="h3st-cont-card-title">✨ Auto / 自动最新</div>
                <div class="h3st-cont-card-meta">空库时生成首段 · 有片时接续最新</div>
            </div>
        `;
        autoCard.onclick = () => {
            setWidgetValue(clipSelW, "latest");
            void loadClips();
        };
        clipGrid.appendChild(autoCard);

        for (const clip of cachedClips) {
            const isActive = curSel === clip.clip_id;
            const card = document.createElement("div");
            card.className = `h3st-cont-card${isActive ? " active" : ""}`;

            const thumb = document.createElement("div");
            thumb.className = "h3st-cont-card-thumb";
            if (clip.thumbnail_url) {
                appendThumbnail(thumb, { thumbnail_url: clip.thumbnail_url });
            } else {
                thumb.textContent = "🎬";
            }
            if (isActive) {
                const b = document.createElement("span");
                b.className = "h3st-cont-card-badge";
                b.textContent = "已选";
                thumb.appendChild(b);
            }
            if (clip.has_video && clip.video_url) {
                const playBtn = document.createElement("button");
                playBtn.type = "button";
                playBtn.className = "h3st-cont-play-btn";
                playBtn.textContent = "▶ 播放";
                playBtn.onclick = (e) => { e.stopPropagation(); openClipVideoModal(clip); };
                thumb.appendChild(playBtn);

                // Lightweight muted hover preview
                let hoverVideo = null;
                let hoverTimer = null;
                const stopHover = () => {
                    clearTimeout(hoverTimer);
                    hoverTimer = null;
                    if (hoverVideo) {
                        hoverVideo.pause();
                        hoverVideo.removeAttribute("src");
                        hoverVideo.load();
                        hoverVideo.remove();
                        hoverVideo = null;
                    }
                };
                clipPreviewCleanups.add(stopHover);
                card.addEventListener("mouseenter", () => {
                    if (signal.aborted) return;
                    stopHover();
                    hoverTimer = setTimeout(() => {
                        if (signal.aborted || !card.isConnected) return;
                        hoverVideo = document.createElement("video");
                        hoverVideo.className = "minimax-clip-hover-video";
                        hoverVideo.src = clip.video_url;
                        hoverVideo.muted = true;
                        hoverVideo.loop = true;
                        hoverVideo.playsInline = true;
                        hoverVideo.autoplay = true;
                        thumb.appendChild(hoverVideo);
                        hoverVideo.play().catch(() => {});
                        hoverVideo.style.opacity = "1";
                    }, 220);
                });
                card.addEventListener("mouseleave", stopHover);
                card.ondblclick = (e) => { e.stopPropagation(); openClipVideoModal(clip); };
            }

            const cBody = document.createElement("div");
            cBody.className = "h3st-cont-card-body";
            const t = document.createElement("div");
            t.className = "h3st-cont-card-title";
            t.textContent = clip.shot_tag || clip.clip_id;
            t.title = `${clip.shot_tag || ""} (${clip.clip_id})`;
            cBody.appendChild(t);

            if (clip.variants) {
                const variantBar = document.createElement("div");
                variantBar.className = "minimax-clip-variants";
                for (const [label, variant] of Object.entries(clip.variants)) {
                    if (!variant?.has_latent) continue;
                    const dur = variant.duration_seconds != null ? `${Number(variant.duration_seconds).toFixed(2)}s` : "?";
                    const badge = document.createElement("div");
                    badge.className = `variant-badge ${label === "二采" ? "variant-b" : "variant-a"}`;
                    const nameSpan = document.createElement("span");
                    nameSpan.className = "variant-name";
                    nameSpan.textContent = dualVariantName(label, clip);
                    const timeSpan = document.createElement("span");
                    timeSpan.className = "variant-duration";
                    timeSpan.textContent = dur;
                    badge.append(nameSpan, timeSpan);
                    variantBar.appendChild(badge);
                }
                cBody.appendChild(variantBar);
            } else {
                const m = document.createElement("div");
                m.className = "h3st-cont-card-meta";
                m.textContent = clip.created_at?.slice(5, 16) || "点击选作接续源";
                cBody.appendChild(m);
            }

            if (clip.parent_clip_id) {
                const lin = document.createElement("div");
                lin.className = "minimax-clip-lineage";
                lin.textContent = `↳ 衍生自: ${clip.parent_clip_id.slice(-8)}`;
                lin.title = `父镜头: ${clip.parent_clip_id}`;
                cBody.appendChild(lin);
            }

            addClipDeleteButton(cBody, {
                description: `项目「${project}」的 ${clip.shot_tag || clip.clip_id}（含一采、二采）`,
                signal,
                onDelete: async () => {
                    releaseClipPreviews();
                    document.getElementById("h3st-clip-modal")?._h3Close?.();
                    const r = await api.fetchApi("/minimax/clip_bin/delete", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ project, clip_id: clip.clip_id }),
                    });
                    if (!r.ok) {
                        const err = await r.json().catch(() => ({}));
                        throw new Error(err.error || `删除失败 (${r.status})`);
                    }
                    void loadClips(clip.clip_id);
                },
            });

            card.append(thumb, cBody);
            card.onclick = () => {
                setWidgetValue(clipSelW, clip.clip_id);
                void loadClips();
            };
            clipGrid.appendChild(card);
        }
        syncStatusBar();
    }

    // --- Tab 2: Load Segments / Uploaded Videos ---
    function mountVideoProjectMenu() {
        vidMenuAbort?.abort();
        vidMenuAbort = new AbortController();
        const mode = String(vidModeW?.value || "segments");
        vidMenu = createLvmProjectMenu({
            node,
            widget: vidProjW,
            signal: vidMenuAbort.signal,
            scope: mode,
            onChange: () => { void loadVideoCards(); },
        });
        vidMenuSlot.replaceChildren(vidMenu.element);
    }
    signal.addEventListener("abort", () => vidMenuAbort?.abort(), { once: true });

    async function loadVideoCards() {
        if (signal.aborted) return;
        const req = ++vidRequestId;
        syncVideoControls();
        if (!vidMenu) mountVideoProjectMenu();
        await vidMenu.refresh();
        if (signal.aborted || req !== vidRequestId) return;

        const mode = String(vidModeW?.value || "segments");
        const project = vidProjW?.value || "H3_LVM";
        const curSegId = parseInt(segIdW?.value, 10) || 1;
        const url = mode === "uploads"
            ? `/h3_lvm/uploads?project=${encodeURIComponent(project)}&output_size=${encodeURIComponent(vidSizeW?.value ?? "")}&custom_max_side=${encodeURIComponent(vidCustomW?.value ?? "")}`
            : `/h3_lvm/segments?project=${encodeURIComponent(project)}`;

        try {
            const res = await api.fetchApi(url, { cache: "no-store" });
            if (signal.aborted || req !== vidRequestId) return;
            const data = res.ok ? await res.json().catch(() => ({})) : {};
            cachedSegments = Array.isArray(data.segments)
                ? data.segments
                : (Array.isArray(data.videos) ? data.videos : []);
        } catch (_) {
            if (signal.aborted || req !== vidRequestId) return;
            cachedSegments = [];
        }

        vidGrid.innerHTML = "";
        if (!cachedSegments.length) {
            const empty = document.createElement("div");
            empty.style.cssText = "grid-column:1/-1;padding:24px;text-align:center;color:#64748b;";
            empty.textContent = mode === "uploads"
                ? "当前视频库为空，可点击「＋ 载入视频」导入或拖放视频文件。"
                : "当前切片库为空，请先使用 H3 Long Video Manager / Smart Split 切分视频。";
            vidGrid.appendChild(empty);
            syncStatusBar();
            return;
        }

        for (const item of cachedSegments) {
            const sid = Number(item.segment_id);
            const isActive = sid === curSegId;
            const card = document.createElement("div");
            card.className = `h3st-cont-card${isActive ? " active" : ""}`;

            const thumb = document.createElement("div");
            thumb.className = "h3st-cont-card-thumb";
            const thumbUrl = item.thumbnail_url || item.cover_url;
            if (thumbUrl) {
                appendThumbnail(thumb, {
                    thumbnail_url: thumbUrl,
                    width: item.width,
                    height: item.height,
                });
            } else {
                thumb.textContent = "🎞";
            }
            const badge = document.createElement("span");
            badge.className = "h3st-cont-card-badge";
            badge.textContent = isActive ? `✓ #${sid}` : `#${sid}`;
            thumb.appendChild(badge);

            if (mode === "segments" && item.has_mp4 && item.mp4_url) {
                const playBtn = document.createElement("button");
                playBtn.type = "button";
                playBtn.className = "h3st-cont-play-btn";
                playBtn.textContent = "▶ 预览";
                playBtn.onclick = (e) => {
                    e.stopPropagation();
                    openLvmPreview(item, signal);
                };
                thumb.appendChild(playBtn);
            } else if (mode === "uploads" && !item.missing_file && item.video_url) {
                const playBtn = document.createElement("button");
                playBtn.type = "button";
                playBtn.className = "h3st-cont-play-btn";
                playBtn.textContent = "▶ 预览";
                playBtn.onclick = async (e) => {
                    e.stopPropagation();
                    playBtn.disabled = true;
                    playBtn.textContent = "…";
                    try {
                        const r = await api.fetchApi("/h3_lvm/uploads/preview", {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ project, segment_id: sid }),
                        });
                        const data = await r.json().catch(() => ({}));
                        if (!r.ok) throw new Error(data.error || `预览读取失败 (${r.status})`);
                        if (signal.aborted) return;
                        const caption = `视频 ${sid} · ${displayCardLabel(item)} · ${formatDuration(item.duration_sec)} · ${item.width}×${item.height}`;
                        openLvmPreview({
                            ...item,
                            mp4_url: data.url,
                            proxy_url: data.proxy_url,
                            proxy_recommended: data.proxy_recommended || !canBrowserPreview(item),
                        }, signal, caption);
                    } catch (err) {
                        if (!signal.aborted) window.alert(`预览失败：${err.message}`);
                    } finally {
                        playBtn.disabled = false;
                        playBtn.textContent = "▶ 预览";
                    }
                };
                thumb.appendChild(playBtn);
            }

            const cBody = document.createElement("div");
            cBody.className = "h3st-cont-card-body";
            const title = document.createElement("div");
            title.className = "h3st-cont-card-title";
            title.textContent = mode === "uploads"
                ? (displayCardLabel(item) || `视频 #${sid}`)
                : `片段 #${sid}`;
            title.title = item.original_name || title.textContent;
            cBody.appendChild(title);

            if (mode === "uploads") {
                for (const line of uploadInfoLines(item)) {
                    const meta = document.createElement("div");
                    meta.className = "h3st-cont-card-meta";
                    meta.textContent = line;
                    cBody.appendChild(meta);
                }
            } else {
                const meta = document.createElement("div");
                meta.className = "h3st-cont-card-meta";
                const frames = item.frames ? `${item.frames}帧` : "";
                const dur = item.duration_sec != null ? `${Number(item.duration_sec).toFixed(1)}s` : "";
                const resTxt = (item.width && item.height) ? `${item.width}×${item.height}` : "";
                meta.textContent = [frames, dur, resTxt].filter(Boolean).join(" · ") || "点击选中";
                cBody.appendChild(meta);
            }

            addLvmDeleteButton(cBody, {
                description: mode === "uploads"
                    ? `投放视频库「${project}」的视频 #${sid}（${displayCardLabel(item)}）`
                    : `项目「${project}」的片段 #${sid}`,
                signal,
                onDelete: async () => {
                    document.getElementById("h3lvm-preview-overlay")?._h3Close?.();
                    const ep = mode === "uploads" ? "/h3_lvm/uploads/delete" : "/h3_lvm/delete";
                    const r = await api.fetchApi(ep, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ project, segment_id: sid }),
                    });
                    if (!r.ok) {
                        const err = await r.json().catch(() => ({}));
                        throw new Error(err.error || `删除失败 (${r.status})`);
                    }
                    void loadVideoCards();
                },
            });

            card.append(thumb, cBody);
            card.onclick = () => {
                setWidgetValue(segIdW, sid);
                node.setDirtyCanvas?.(true, true);
                void loadVideoCards();
            };
            vidGrid.appendChild(card);
        }
        syncStatusBar();
    }

    async function importUploadedVideos(files) {
        if (!files || !files.length) return;
        const project = vidProjW?.value || "H3_LVM";
        for (const file of files) {
            const fd = new FormData();
            // Order matters: backend reads "project" text field BEFORE the "file" part!
            fd.append("project", project);
            fd.append("file", file, file.name);
            await api.fetchApi("/h3_lvm/uploads/import", { method: "POST", body: fd });
        }
        void loadVideoCards();
    }

    // --- Event Bindings ---
    clipTabBtn.onclick = () => switchTab("clips");
    vidTabBtn.onclick = () => switchTab("video");

    quickVidToggle.onclick = (e) => {
        e.stopPropagation();
        setWidgetValue(enVidW, !Boolean(enVidW?.value));
        syncVideoControls();
        syncStatusBar();
        node.setDirtyCanvas?.(true, true);
    };

    useDurCheck.onchange = () => {
        setWidgetValue(useDurW, useDurCheck.checked);
        syncStatusBar();
        node.setDirtyCanvas?.(true, true);
    };
    autoAdvCheck.onchange = () => {
        setWidgetValue(autoAdvW, autoAdvCheck.checked);
        node.setDirtyCanvas?.(true, true);
    };

    segModeBtn.onclick = () => {
        setWidgetValue(vidModeW, "segments");
        mountVideoProjectMenu();
        void loadVideoCards();
    };
    upModeBtn.onclick = () => {
        setWidgetValue(vidModeW, "uploads");
        mountVideoProjectMenu();
        void loadVideoCards();
    };

    vidRefreshBtn.onclick = () => void loadVideoCards();
    vidUploadBtn.onclick = () => vidFileInput.click();
    vidFileInput.onchange = () => {
        const files = Array.from(vidFileInput.files || []);
        vidFileInput.value = "";
        void importUploadedVideos(files);
    };
    vidDeck.addEventListener("dragover", (e) => {
        if (String(vidModeW?.value) === "uploads") e.preventDefault();
    });
    vidDeck.addEventListener("drop", (e) => {
        if (String(vidModeW?.value) !== "uploads") return;
        e.preventDefault();
        void importUploadedVideos(Array.from(e.dataTransfer?.files || []));
    });

    // Hook ClipBin project & resize widget callbacks so menu switching reloads immediately!
    if (clipProjW) {
        const prev = clipProjW.callback;
        clipProjW.callback = function () {
            const r = prev ? prev.apply(this, arguments) : undefined;
            void loadClips();
            return r;
        };
    }
    for (const widget of [vidSizeW, vidCustomW]) {
        if (!widget) continue;
        const prev = widget.callback;
        widget.callback = function () {
            const r = prev ? prev.apply(this, arguments) : undefined;
            if (String(vidModeW?.value) === "uploads") void loadVideoCards();
            return r;
        };
    }

    // Manual one-click "⏭ 下一段并接续最新"
    nextStepBtn.onclick = () => {
        const step = computeNextSegmentStep({
            segmentId: segIdW?.value,
            totalSegments: cachedSegments.length,
            clipMode: clipModeW?.value,
            clipModeOptions: clipModeW?.options?.values || [],
        });
        setWidgetValue(segIdW, step.segmentId);
        setWidgetValue(clipSelW, step.clipSelection);
        setWidgetValue(clipModeW, step.clipMode);
        syncClipModeButtons();
        syncStatusBar();
        node.setDirtyCanvas?.(true, true);
        if (activeTab === "clips") void loadClips();
        else void loadVideoCards();
    };

    // Listen for backend refresh events & clean up all listeners on abort
    const onClipChanged = (ev) => {
        if (signal.aborted) return;
        const deletedId = ev?.detail?.project === (clipProjW?.value || "Default_Project")
            ? ev?.detail?.deleted_id : null;
        void loadClips(deletedId);
    };
    const onLvmChanged = () => {
        if (!signal.aborted) void loadVideoCards();
    };
    const onExecSuccess = () => {
        if (!signal.aborted) void loadClips();
    };
    api.addEventListener?.("h3_studio/clip_saved", onClipChanged);
    api.addEventListener?.("minimax/clip_bin/changed", onClipChanged);
    api.addEventListener?.("h3_lvm/changed", onLvmChanged);
    api.addEventListener?.("h3_lvm/uploads_changed", onLvmChanged);
    api.addEventListener?.("execution_success", onExecSuccess);
    signal.addEventListener("abort", () => {
        api.removeEventListener?.("h3_studio/clip_saved", onClipChanged);
        api.removeEventListener?.("minimax/clip_bin/changed", onClipChanged);
        api.removeEventListener?.("h3_lvm/changed", onLvmChanged);
        api.removeEventListener?.("h3_lvm/uploads_changed", onLvmChanged);
        api.removeEventListener?.("execution_success", onExecSuccess);
    }, { once: true });

    addPanel(node, "h3_studio_continuity_hub", root, 480);

    node._h3StudioContRefresh = () => {
        syncClipModeButtons();
        syncAntiDriftUI();
        syncVideoControls();
        mountVideoProjectMenu();
        void loadClips();
        void loadVideoCards();
    };

    node._h3StudioContExecuted = () => {
        void loadClips();
        if (shouldAutoAdvanceAfterRun({
            autoAdvanceSegment: Boolean(autoAdvW?.value),
            enableVideoRef: Boolean(enVidW?.value),
            hadError: false,
        })) {
            const step = computeNextSegmentStep({
                segmentId: segIdW?.value,
                totalSegments: cachedSegments.length,
                clipMode: clipModeW?.value,
                clipModeOptions: clipModeW?.options?.values || [],
            });
            setWidgetValue(segIdW, step.segmentId);
            setWidgetValue(clipSelW, step.clipSelection);
            syncStatusBar();
            void loadVideoCards();
        }
    };

    syncClipModeButtons();
    syncAntiDriftUI();
    syncVideoControls();
    mountVideoProjectMenu();
    void loadClips();
    void loadVideoCards();
}
