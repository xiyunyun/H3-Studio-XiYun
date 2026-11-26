// H3 Studio Media Hub (Plan 1A): 2-Tab Reference Image (0-9) & Audio (0-3) Workbench
// - Persistent Active Dock showing all enabled references across both tabs
// - Drag-and-drop card reordering (restored from ref_loader.js)
// - Clean actions: 忽略 / 替换 / 删除 (active), 启用 / 删除 (ignored)
// - Extension-stripped default filenames + custom inline rename (clearing reverts)
// - 100% compatible with /h3_lvm/ref_presets, /h3_lvm/image_metadata & /h3_lvm/ref_plan

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { addPanel } from "../lvm/dom_panel.js";
import { createCardZoom } from "../lvm/card_zoom.js";
import { createProjectMenu } from "../lvm/project_menu.js";
import { appendThumbnail } from "../lvm/thumb_fit.js";
import { imageMetadataPath, imageThumbnailUrl } from "../lvm/image_preview.js";
import {
    MAX_REF_AUDIOS,
    MAX_REF_IMAGES,
    addEntries,
    applyPreset,
    countActive,
    insertPosition,
    mediaKey,
    mediaKind,
    moveActive,
    parseMedia,
    pickPresetEntries,
    removeEntry,
    replaceEntry,
    restoreAll,
    serializeMedia,
    sizeLine,
    splitEntries,
} from "../lvm/ref_state.js";
import {
    applyCardRename,
    buildActiveDockItems,
    displayCardLabel,
    restoreCardAtIndex,
    skipCardAtIndex,
    stripFilenameExt,
} from "./studio_state.js";

try {
    const sheets = [
        ["h3lvm-styles", "../lvm/h3lvm_picker.css"],
        ["h3st-media-styles", "./studio_media_hub.css"],
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
    console.warn("[H3 Studio MediaHub] CSS injection failed:", err);
}

const NODE_ID = "H3StudioMediaHub";
const UPLOAD_ROOT = "h3-lvm-refs";
const CARD_MIME = "application/x-h3st-card";
const SCROLL_EDGE = 28;

function sanitizeProjectFolder(name) {
    const cleaned = String(name ?? "")
        .trim()
        .replace(/[\\/:*?"<>|\x00-\x1f]+/g, "_")
        .trim()
        .replace(/^\.+|\.+$/g, "");
    return cleaned || "H3_LVM";
}

function viewUrlFor(entry) {
    const params = new URLSearchParams({
        filename: entry.filename || "",
        type: entry.type || "input",
    });
    if (entry.subfolder) params.set("subfolder", entry.subfolder);
    const path = `/view?${params.toString()}`;
    if (typeof api.apiURL === "function") return api.apiURL(path);
    if (typeof api.getFileUrl === "function") return api.getFileUrl(path);
    return path;
}

function hideWidget(widget) {
    if (!widget) return;
    widget.options = widget.options || {};
    widget.options.hidden = true;
    widget.computeSize = () => [0, -4];
    widget.type = "hidden";
}

app.registerExtension({
    name: "H3.Studio.MediaHub",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_ID) return;

        nodeType.prototype.min_size = [480, 420];

        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = origCreated ? origCreated.apply(this, arguments) : undefined;
            try {
                installMediaHub(this);
                this.setSize([Math.max(this.size[0], 720), Math.max(this.size[1], 620)]);
            } catch (err) {
                console.warn("[H3 Studio MediaHub] setup failed (node still works):", err);
            }
            return r;
        };

        const origConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = origConfigure ? origConfigure.apply(this, arguments) : undefined;
            this._h3StudioMediaRefresh?.();
            return r;
        };

        const origRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            this._h3StudioMediaLifetime?.abort();
            return origRemoved ? origRemoved.apply(this, arguments) : undefined;
        };
    },
});

function installMediaHub(node) {
    const lifetime = node._h3StudioMediaLifetime ?? new AbortController();
    node._h3StudioMediaLifetime = lifetime;
    const signal = lifetime.signal;

    const w = (name) => node.widgets?.find(item => item.name === name);
    const imgFilesW = w("image_files");
    const imgProjW = w("image_project");
    const mpW = w("target_megapixels");
    const resampleW = w("resampling_method");
    const upscaleW = w("upscale_small_images");
    const audFilesW = w("audio_files");
    const audProjW = w("audio_project");

    if (mpW) mpW.label = "参考图目标像素 MP (target_megapixels)";
    if (resampleW) resampleW.label = "缩放重采样算法 (resampling_method)";
    if (upscaleW) upscaleW.label = "小图是否放大 (upscale_small_images)";
    hideWidget(imgFilesW);
    hideWidget(audFilesW);
    hideWidget(imgProjW);
    hideWidget(audProjW);

    let activeTab = "image"; // "image" | "audio"
    let imgEntries = [];
    let audEntries = [];
    const sourceCache = new Map();
    const planCache = new Map();
    let hoverPop = null;
    let renderGen = 0;

    // Auto-save timers per kind (matches ref_loader.js behavior)
    const activeProjectByKind = { image: "", audio: "" };
    const saveTimerByKind = { image: 0, audio: 0 };
    const pendingSaveByKind = { image: false, audio: false };
    const switchTokenByKind = { image: 0, audio: 0 };
    let suppressAutoSave = false;

    function getEntries(kind) {
        return kind === "image" ? imgEntries : audEntries;
    }

    function maxFor(kind) {
        return kind === "image" ? MAX_REF_IMAGES : MAX_REF_AUDIOS;
    }

    function menuFor(kind) {
        return kind === "image" ? imgMenu : audMenu;
    }

    function readListsFromWidgets() {
        try {
            imgEntries = parseMedia(imgFilesW?.value || "[]", MAX_REF_IMAGES).entries;
        } catch (_) {
            imgEntries = [];
        }
        try {
            audEntries = parseMedia(audFilesW?.value || "[]", MAX_REF_AUDIOS).entries;
        } catch (_) {
            audEntries = [];
        }
    }

    function schedulePresetSave(kind) {
        if (suppressAutoSave) return;
        pendingSaveByKind[kind] = true;
        window.clearTimeout(saveTimerByKind[kind]);
        saveTimerByKind[kind] = window.setTimeout(() => {
            saveTimerByKind[kind] = 0;
            void flushPresetSave(kind, activeProjectByKind[kind] || menuFor(kind).name());
        }, 500);
    }

    async function flushPresetSave(kind, project) {
        window.clearTimeout(saveTimerByKind[kind]);
        saveTimerByKind[kind] = 0;
        if (!pendingSaveByKind[kind]) return;
        pendingSaveByKind[kind] = false;
        await writePreset(kind, project, getEntries(kind), { quiet: true });
    }

    function writeList(kind, entries, { autoSave = true } = {}) {
        if (kind === "image") {
            imgEntries = entries;
            if (imgFilesW) {
                imgFilesW.value = serializeMedia(entries);
                imgFilesW.callback?.(imgFilesW.value);
            }
        } else {
            audEntries = entries;
            if (audFilesW) {
                audFilesW.value = serializeMedia(entries);
                audFilesW.callback?.(audFilesW.value);
            }
        }
        node.setDirtyCanvas?.(true, true);
        node.graph?.setChanged?.();
        if (autoSave) schedulePresetSave(kind);
        renderAll();
    }

    function clearHoverPop() {
        if (hoverPop) {
            hoverPop.remove();
            hoverPop = null;
        }
    }

    // --- DOM Root ---
    const root = document.createElement("div");
    root.className = "h3st-media-root";

    // 1. Top 2-Tab Bar + Snapshot button
    const tabBar = document.createElement("div");
    tabBar.className = "h3st-media-tabs";

    const imgTabBtn = document.createElement("button");
    imgTabBtn.type = "button";
    imgTabBtn.className = "h3st-media-tab active";
    const imgTabText = document.createElement("span");
    imgTabText.textContent = "🖼 参考图";
    const imgTabBadge = document.createElement("span");
    imgTabBadge.className = "h3st-media-tab-badge";
    imgTabBtn.append(imgTabText, imgTabBadge);

    const audTabBtn = document.createElement("button");
    audTabBtn.type = "button";
    audTabBtn.className = "h3st-media-tab";
    const audTabText = document.createElement("span");
    audTabText.textContent = "🎧 参考音频";
    const audTabBadge = document.createElement("span");
    audTabBadge.className = "h3st-media-tab-badge";
    audTabBtn.append(audTabText, audTabBadge);

    const spacer = document.createElement("span");
    spacer.className = "h3st-media-spacer";

    const snapBtn = document.createElement("button");
    snapBtn.type = "button";
    snapBtn.className = "h3st-media-snap-btn";
    snapBtn.textContent = "📸 组合快照";
    snapBtn.title = "一键保存或恢复当前「参考图 + 参考音频」整套搭配";

    tabBar.append(imgTabBtn, audTabBtn, spacer, snapBtn);
    root.appendChild(tabBar);

    // 2. Persistent Active Dock (Plan 1A: never hides!)
    const dock = document.createElement("div");
    dock.className = "h3st-media-dock";
    root.appendChild(dock);

    // 3. Tab Content Body
    const body = document.createElement("div");
    body.className = "h3st-media-body";
    root.appendChild(body);

    // Project Menus (one per tab, preserving independent libraries)
    const imgMenuSlot = document.createElement("div");
    const audMenuSlot = document.createElement("div");
    const imgMenu = createProjectMenu({
        node,
        widget: imgProjW,
        signal,
        scope: "image",
        onChange: (name, source) => { void switchProject("image", name, source); },
    });
    const audMenu = createProjectMenu({
        node,
        widget: audProjW,
        signal,
        scope: "audio",
        onChange: (name, source) => { void switchProject("audio", name, source); },
    });
    activeProjectByKind.image = imgMenu.name();
    activeProjectByKind.audio = audMenu.name();
    imgMenuSlot.appendChild(imgMenu.element);
    audMenuSlot.appendChild(audMenu.element);
    body.append(imgMenuSlot, audMenuSlot);

    // Toolbar
    const toolbar = document.createElement("div");
    toolbar.className = "h3st-media-toolbar";

    const addBtn = document.createElement("button");
    addBtn.type = "button";
    addBtn.className = "h3st-media-btn primary";

    const savePresetBtn = document.createElement("button");
    savePresetBtn.type = "button";
    savePresetBtn.className = "h3st-media-btn";
    savePresetBtn.textContent = "💾 存预设";
    savePresetBtn.title = "将当前卡片顺序、名称与启用/忽略状态保存到当前项目预设";

    const loadPresetBtn = document.createElement("button");
    loadPresetBtn.type = "button";
    loadPresetBtn.className = "h3st-media-btn";
    loadPresetBtn.textContent = "📂 读预设";
    loadPresetBtn.title = "从当前项目读取已保存的预设（恢复当时的启用与忽略状态）";

    const refreshBtn = document.createElement("button");
    refreshBtn.type = "button";
    refreshBtn.className = "h3st-media-btn";
    refreshBtn.textContent = "🔄 刷新";

    const zoomMount = document.createElement("span");
    createCardZoom({ container: body, node, baseCardMin: 150, baseThumbH: 100 }).mount(zoomMount);

    const statusLine = document.createElement("span");
    statusLine.style.cssText = "margin-left:auto;font-size:11px;color:#94a3b8;";

    toolbar.append(addBtn, savePresetBtn, loadPresetBtn, refreshBtn, zoomMount, statusLine);
    body.appendChild(toolbar);

    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.multiple = true;
    fileInput.style.display = "none";
    root.appendChild(fileInput);

    // Reusable hidden file picker for single-card "替换"
    let replaceTarget = null;
    const replaceInput = document.createElement("input");
    replaceInput.type = "file";
    replaceInput.multiple = false;
    replaceInput.style.display = "none";
    root.appendChild(replaceInput);

    // Card Deck
    const deck = document.createElement("div");
    deck.className = "h3st-media-deck";
    body.appendChild(deck);

    // --- Backend Preset & Upload Actions ---
    async function projectExists(kind, name) {
        try {
            const res = await api.fetchApi(
                `/h3_lvm/projects?scope=${encodeURIComponent(kind)}`,
                { cache: "no-store" },
            );
            if (!res.ok) return true;
            const data = await res.json().catch(() => ({}));
            return !Array.isArray(data.projects) || data.projects.some(bin => bin.name === name);
        } catch (_) {
            return true;
        }
    }

    async function fetchPresetEntries(kind, projectName) {
        const res = await api.fetchApi(
            `/h3_lvm/ref_presets?project=${encodeURIComponent(projectName)}&kind=${encodeURIComponent(kind)}`,
            { cache: "no-store" },
        );
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json().catch(() => ({}));
        return pickPresetEntries(data);
    }

    async function switchProject(kind, name, source = "select") {
        if (!name) return;
        if (source === "rename") {
            activeProjectByKind[kind] = name;
            return;
        }
        const previous = activeProjectByKind[kind];
        if (name === previous) return;
        const token = ++switchTokenByKind[kind];
        if (pendingSaveByKind[kind] && previous && await projectExists(kind, previous)) {
            await flushPresetSave(kind, previous);
        } else {
            window.clearTimeout(saveTimerByKind[kind]);
            saveTimerByKind[kind] = 0;
            pendingSaveByKind[kind] = false;
        }
        activeProjectByKind[kind] = name;
        try {
            const incoming = await fetchPresetEntries(kind, name);
            if (token !== switchTokenByKind[kind] || signal.aborted) return;
            const applied = applyPreset([], incoming, maxFor(kind));
            suppressAutoSave = true;
            writeList(kind, applied.entries, { autoSave: false });
            suppressAutoSave = false;
            statusLine.textContent = applied.entries.length
                ? `已切至「${name}」并载入 ${applied.entries.length} 项`
                : `已切至「${name}」（该库暂无预设）`;
        } catch (err) {
            if (!signal.aborted) {
                console.warn("[H3 Studio MediaHub] switchProject failed:", err);
                statusLine.textContent = `切换项目「${name}」读取失败`;
            }
        }
    }

    async function loadPreset(kind, projectName) {
        const proj = projectName || menuFor(kind).name();
        try {
            const incoming = await fetchPresetEntries(kind, proj);
            if (signal.aborted) return;
            if (!incoming.length) {
                statusLine.textContent = `项目「${proj}」暂无已保存预设`;
                return;
            }
            const applied = applyPreset(getEntries(kind), incoming, maxFor(kind));
            suppressAutoSave = true;
            writeList(kind, applied.entries, { autoSave: false });
            suppressAutoSave = false;
            const activeCnt = applied.entries.length - applied.parked;
            statusLine.textContent = `已载入「${proj}」：已启用 ${activeCnt} · 已忽略 ${applied.parked}`;
        } catch (err) {
            console.warn("[H3 Studio MediaHub] loadPreset failed:", err);
            statusLine.textContent = `读取项目「${proj}」预设失败`;
        }
    }

    async function writePreset(kind, proj, entries, { quiet = false } = {}) {
        try {
            const serialized = serializeMedia(entries);
            const res = await api.fetchApi("/h3_lvm/ref_presets", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ kind, project: proj, entries: serialized }),
            });
            const data = await res.json().catch(() => ({}));
            if (!res.ok) {
                if (!quiet) statusLine.textContent = data.error || `保存失败 (HTTP ${res.status})`;
                return false;
            }
            menuFor(kind).updateBin({
                name: data.name || proj,
                references: Number(data.saved) || 0,
                cover_url: data.cover_url || "",
                updated_at: data.updated_at || "",
            });
            if (!quiet) statusLine.textContent = `已存至项目「${proj}」(${Number(data.saved) || entries.length}项)`;
            return true;
        } catch (err) {
            if (!signal.aborted) console.warn("[H3 Studio MediaHub] savePreset failed:", err);
            if (!quiet) statusLine.textContent = "保存失败：无法连接后台服务";
            return false;
        }
    }

    async function savePreset(kind) {
        const proj = menuFor(kind).name();
        const entries = getEntries(kind);
        pendingSaveByKind[kind] = false;
        window.clearTimeout(saveTimerByKind[kind]);
        saveTimerByKind[kind] = 0;
        await writePreset(kind, proj, entries, { quiet: false });
    }

    async function uploadSingleFile(kind, file) {
        const proj = sanitizeProjectFolder(menuFor(kind).name());
        const subfolder = `${UPLOAD_ROOT}/${kind}/${proj}`;
        const fd = new FormData();
        fd.append("image", file);
        fd.append("type", "input");
        fd.append("subfolder", subfolder);
        const res = await api.fetchApi("/upload/image", { method: "POST", body: fd });
        if (!res.ok) throw new Error(`上传失败 (HTTP ${res.status})`);
        const data = await res.json().catch(() => ({}));
        if (!data.name) throw new Error("上传失败：服务器未返回文件名");
        return {
            filename: data.name,
            subfolder: typeof data.subfolder === "string" && data.subfolder ? data.subfolder : subfolder,
            type: data.type || "input",
        };
    }

    async function uploadFiles(kind, files) {
        if (!files || files.length === 0) return;
        const usable = Array.from(files).filter(f => mediaKind(f.name) === kind);
        if (!usable.length) {
            statusLine.textContent = `只能添加${kind === "image" ? "图片" : "音频"}文件`;
            return;
        }
        const uploaded = [];
        let failed = 0;
        for (let i = 0; i < usable.length; i++) {
            statusLine.textContent = `上传中 ${i + 1}/${usable.length}...`;
            try {
                uploaded.push(await uploadSingleFile(kind, usable[i]));
            } catch (_) {
                failed++;
            }
        }
        if (uploaded.length) {
            const res = addEntries(getEntries(kind), uploaded, maxFor(kind));
            writeList(kind, res.entries);
            const notes = [`已添加 ${res.added} 项`];
            if (res.parked) notes.push(`${res.parked} 项放入已忽略（槽位已满）`);
            if (failed) notes.push(`${failed} 个失败`);
            statusLine.textContent = notes.join(" · ");
        } else if (failed) {
            statusLine.textContent = `上传失败 ${failed} 个`;
        }
    }

    function pickReplacement(kind, sourceIndex) {
        replaceTarget = { kind, sourceIndex };
        replaceInput.accept = kind === "image"
            ? "image/png,image/jpeg,image/webp,image/bmp,image/gif"
            : "audio/*,.wav,.mp3,.m4a,.flac,.ogg,.opus";
        replaceInput.click();
    }

    replaceInput.onchange = async () => {
        const file = replaceInput.files?.[0];
        replaceInput.value = "";
        const target = replaceTarget;
        replaceTarget = null;
        if (!file || !target) return;
        if (mediaKind(file.name) !== target.kind) {
            statusLine.textContent = `只能选择${target.kind === "image" ? "图片" : "音频"}文件进行替换`;
            return;
        }
        statusLine.textContent = "正在替换素材...";
        try {
            const uploaded = await uploadSingleFile(target.kind, file);
            const next = replaceEntry(getEntries(target.kind), target.sourceIndex, uploaded);
            writeList(target.kind, next);
            statusLine.textContent = "已替换素材（槽位顺序与名称已保留）";
        } catch (err) {
            statusLine.textContent = err?.message || "替换失败";
        }
    };

    async function loadImageSource(entry) {
        const key = mediaKey(entry);
        if (sourceCache.has(key)) return sourceCache.get(key);
        let result;
        try {
            const res = await api.fetchApi(imageMetadataPath(entry), { cache: "no-store" });
            const data = await res.json().catch(() => ({}));
            const width = Number(data.width);
            const height = Number(data.height);
            result = res.ok && width > 0 && height > 0
                ? { width, height }
                : { error: data.error || "图片读取失败" };
        } catch (_) {
            result = { error: "图片读取失败" };
        }
        sourceCache.set(key, result);
        return result;
    }

    async function fetchImageSizeLine(entry) {
        const source = await loadImageSource(entry);
        if (source.error) return { text: source.error, source: null };
        const mp = Number(mpW?.value ?? 1.25);
        const up = Boolean(upscaleW?.value);
        const key = `${source.width}x${source.height}|${mp}|${up ? 1 : 0}`;
        let plan = planCache.get(key);
        if (!plan) {
            try {
                const res = await api.fetchApi(
                    `/h3_lvm/ref_plan?w=${encodeURIComponent(source.width)}&h=${encodeURIComponent(source.height)}&mp=${encodeURIComponent(mp)}&upscale=${up ? 1 : 0}`,
                    { cache: "no-store" },
                );
                plan = res.ok ? await res.json().catch(() => null) : null;
                if (plan) planCache.set(key, plan);
            } catch (_) {
                plan = null;
            }
        }
        return { text: sizeLine({ source, plan }), full: sizeLine({ source, plan, full: true }), source };
    }

    // --- Snapshot Modal ---
    async function openSnapshotModal() {
        const overlay = document.createElement("div");
        overlay.className = "h3st-snap-overlay";
        const modal = document.createElement("div");
        modal.className = "h3st-snap-modal";

        const header = document.createElement("div");
        header.style.cssText = "display:flex;justify-content:space-between;align-items:center;font-weight:700;";
        const titleSpan = document.createElement("span");
        titleSpan.textContent = "📸 参考图 + 音频 组合快照";
        const closeBtn = document.createElement("button");
        closeBtn.type = "button";
        closeBtn.className = "h3st-media-btn";
        closeBtn.textContent = "✕ 关闭";
        closeBtn.onclick = () => overlay.remove();
        header.append(titleSpan, closeBtn);

        const saveRow = document.createElement("div");
        saveRow.className = "h3st-snap-row";
        const nameInput = document.createElement("input");
        nameInput.placeholder = "输入快照名称（如：露帕+莫宁）";
        const doSaveBtn = document.createElement("button");
        doSaveBtn.type = "button";
        doSaveBtn.className = "h3st-media-btn primary";
        doSaveBtn.textContent = "💾 保存当前搭配";
        saveRow.append(nameInput, doSaveBtn);

        const listBox = document.createElement("div");
        listBox.className = "h3st-snap-list";

        async function refreshSnaps() {
            listBox.innerHTML = `<div style="color:#94a3b8;font-size:11px;">读取中...</div>`;
            const res = await api.fetchApi("/h3_studio/media_snapshots", { cache: "no-store" });
            const data = res.ok ? await res.json().catch(() => ({ snapshots: [] })) : { snapshots: [] };
            const snaps = data.snapshots || [];
            listBox.innerHTML = "";
            if (!snaps.length) {
                listBox.innerHTML = `<div style="color:#64748b;font-size:11px;">暂无已保存的组合快照</div>`;
                return;
            }
            for (const s of snaps) {
                const row = document.createElement("div");
                row.className = "h3st-snap-item";
                const info = document.createElement("div");
                const nameDiv = document.createElement("div");
                nameDiv.style.fontWeight = "600";
                nameDiv.textContent = s.name;
                const subDiv = document.createElement("div");
                subDiv.style.cssText = "font-size:10px;color:#94a3b8;";
                subDiv.textContent = `图 ${s.image_count} · 音 ${s.audio_count} · ${s.saved_at || ""}`;
                info.append(nameDiv, subDiv);

                const btns = document.createElement("div");
                btns.style.cssText = "display:flex;gap:4px;";
                const applyBtn = document.createElement("button");
                applyBtn.type = "button";
                applyBtn.className = "h3st-media-btn primary";
                applyBtn.textContent = "载入";
                applyBtn.onclick = () => {
                    if (s.image_project && imgProjW) {
                        imgProjW.value = s.image_project;
                        activeProjectByKind.image = s.image_project;
                    }
                    if (s.audio_project && audProjW) {
                        audProjW.value = s.audio_project;
                        activeProjectByKind.audio = s.audio_project;
                    }
                    suppressAutoSave = true;
                    writeList("image", parseMedia(s.image_files || [], MAX_REF_IMAGES).entries, { autoSave: false });
                    writeList("audio", parseMedia(s.audio_files || [], MAX_REF_AUDIOS).entries, { autoSave: false });
                    suppressAutoSave = false;
                    void imgMenu.refresh();
                    void audMenu.refresh();
                    overlay.remove();
                };
                const delBtn = document.createElement("button");
                delBtn.type = "button";
                delBtn.className = "h3st-media-act danger";
                delBtn.textContent = "删除";
                delBtn.onclick = async () => {
                    await api.fetchApi("/h3_studio/media_snapshots", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ action: "delete", name: s.name }),
                    });
                    void refreshSnaps();
                };
                btns.append(applyBtn, delBtn);
                row.append(info, btns);
                listBox.appendChild(row);
            }
        }

        doSaveBtn.onclick = async () => {
            const name = nameInput.value.trim();
            if (!name) return;
            await api.fetchApi("/h3_studio/media_snapshots", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    action: "save",
                    name,
                    image_project: imgMenu.name(),
                    audio_project: audMenu.name(),
                    image_files: imgEntries,
                    audio_files: audEntries,
                }),
            });
            nameInput.value = "";
            void refreshSnaps();
        };

        modal.append(header, saveRow, listBox);
        overlay.appendChild(modal);
        overlay.onclick = (e) => { if (e.target === overlay) overlay.remove(); };
        document.body.appendChild(overlay);
        void refreshSnaps();
    }

    // --- Rendering ---
    function switchTab(tab, flashIndex = null) {
        activeTab = tab;
        imgTabBtn.classList.toggle("active", tab === "image");
        audTabBtn.classList.toggle("active", tab === "audio");
        imgMenuSlot.style.display = tab === "image" ? "" : "none";
        audMenuSlot.style.display = tab === "audio" ? "" : "none";
        addBtn.textContent = tab === "image" ? "＋ 添加参考图" : "＋ 添加参考音频";
        fileInput.accept = tab === "image"
            ? "image/png,image/jpeg,image/webp,image/bmp,image/gif"
            : "audio/*,.wav,.mp3,.m4a,.flac,.ogg,.opus";
        renderDeck(flashIndex);
    }

    function renderDock() {
        const dockData = buildActiveDockItems(imgEntries, audEntries);
        imgTabBadge.textContent = `${dockData.images.length}/${MAX_REF_IMAGES}`;
        audTabBadge.textContent = `${dockData.audios.length}/${MAX_REF_AUDIOS}`;

        dock.innerHTML = "";
        if (dockData.totalActive === 0) {
            const empty = document.createElement("span");
            empty.className = "h3st-media-dock-empty";
            empty.textContent = "当前未启用参考素材（纯文生视频）";
            dock.appendChild(empty);
            return;
        }

        const makePill = (item) => {
            const pill = document.createElement("div");
            pill.className = `h3st-media-pill ${item.kind}`;
            if (item.kind === "image") {
                const img = document.createElement("img");
                img.className = "h3st-media-pill-thumb";
                img.src = imageThumbnailUrl(api, item.entry);
                pill.appendChild(img);
            }
            const tag = document.createElement("span");
            tag.className = "h3st-media-pill-tag";
            tag.textContent = item.tag;

            const name = document.createElement("span");
            name.className = "h3st-media-pill-name";
            name.textContent = item.trimBadge ? `${item.label} ${item.trimBadge}` : item.label;

            const xBtn = document.createElement("button");
            xBtn.type = "button";
            xBtn.className = "h3st-media-pill-x";
            xBtn.textContent = "×";
            xBtn.title = "点击忽略（取消启用，不删除文件）";
            xBtn.onclick = (e) => {
                e.stopPropagation();
                clearHoverPop();
                writeList(item.kind, skipCardAtIndex(getEntries(item.kind), item.sourceIndex));
            };

            pill.onclick = () => {
                clearHoverPop();
                switchTab(item.kind, item.sourceIndex);
            };
            pill.addEventListener("mouseenter", () => {
                clearHoverPop();
                const pop = document.createElement("div");
                pop.className = "h3st-media-hover-pop";
                if (item.kind === "image") {
                    const big = document.createElement("img");
                    big.src = imageThumbnailUrl(api, item.entry);
                    pop.appendChild(big);
                }
                const cap = document.createElement("div");
                cap.className = "h3st-media-hover-pop-cap";
                cap.textContent = `${item.tag} ${item.label}${item.trimBadge ? ` (${item.trimBadge})` : ""}`;
                pop.appendChild(cap);
                document.body.appendChild(pop);
                const rect = pill.getBoundingClientRect();
                pop.style.left = `${Math.min(window.innerWidth - 250, Math.max(8, rect.left))}px`;
                pop.style.top = `${Math.max(8, rect.bottom + 6)}px`;
                hoverPop = pop;
            });
            pill.addEventListener("mouseleave", clearHoverPop);

            pill.append(tag, name, xBtn);
            return pill;
        };

        for (const item of dockData.images) dock.appendChild(makePill(item));
        if (dockData.images.length && dockData.audios.length) {
            const sep = document.createElement("span");
            sep.className = "h3st-media-dock-sep";
            dock.appendChild(sep);
        }
        for (const item of dockData.audios) dock.appendChild(makePill(item));
    }

    function renderCard(kind, entry, sourceIndex, slotIdx, isSkippedCard, flash, gen) {
        const max = maxFor(kind);
        const card = document.createElement("div");
        card.className = `h3st-media-card ${kind}${isSkippedCard ? " skipped" : ""}${flash ? " flash" : ""}`;
        card.dataset.index = String(sourceIndex);
        card.dataset.activeIndex = String(isSkippedCard ? -1 : slotIdx);
        // Active cards can be dragged to reorder their slot index (<Picture 1..9> / <Audio 1..3>)
        card.draggable = !isSkippedCard;

        const head = document.createElement("div");
        head.className = "h3st-media-card-head";
        const slotEl = document.createElement("span");
        slotEl.className = "h3st-media-slot-tag";
        slotEl.textContent = isSkippedCard
            ? "已忽略"
            : (kind === "image" ? `<Picture ${slotIdx + 1}>` : `<Audio ${slotIdx + 1}>`);
        head.appendChild(slotEl);
        if (!isSkippedCard) {
            const dragHint = document.createElement("span");
            dragHint.style.cssText = "color:#64748b;font-size:10px;";
            dragHint.textContent = "⋮⋮";
            dragHint.title = "按住卡片可拖拽排序";
            head.appendChild(dragHint);
        }
        card.appendChild(head);

        if (kind === "image") {
            const thumb = document.createElement("div");
            thumb.className = "h3st-media-card-thumb";
            const known = sourceCache.get(mediaKey(entry));
            appendThumbnail(thumb, {
                thumbnail_url: imageThumbnailUrl(api, entry),
                width: known?.width,
                height: known?.height,
            });
            card.appendChild(thumb);
        }

        const cBody = document.createElement("div");
        cBody.className = "h3st-media-card-body";

        // Title + inline rename (default = extension-stripped filename, clearing reverts!)
        const titleRow = document.createElement("div");
        titleRow.className = "h3st-media-title-row";
        const titleEl = document.createElement("div");
        titleEl.className = "h3st-media-title";
        const shownLabel = displayCardLabel(entry);
        titleEl.textContent = shownLabel;
        titleEl.title = `${shownLabel}\n原文件: ${entry.filename}\n双击改名（清空后自动恢复原文件名）`;

        const renameBtn = document.createElement("button");
        renameBtn.type = "button";
        renameBtn.className = "h3st-media-act";
        renameBtn.style.flex = "0 0 auto";
        renameBtn.style.padding = "2px 6px";
        renameBtn.textContent = "✎";
        renameBtn.title = "重命名（清空后自动恢复原文件名）";
        renameBtn.draggable = false;

        const startRename = () => {
            const input = document.createElement("input");
            input.className = "h3st-media-title-input";
            input.value = entry.label || "";
            input.placeholder = stripFilenameExt(entry.filename);
            input.draggable = false;
            input.addEventListener("dragstart", (e) => { e.preventDefault(); e.stopPropagation(); });
            input.addEventListener("mousedown", (e) => e.stopPropagation());
            const commit = () => {
                writeList(kind, applyCardRename(getEntries(kind), sourceIndex, input.value));
            };
            input.onblur = commit;
            input.onkeydown = (e) => {
                e.stopPropagation();
                if (e.key === "Enter") { e.preventDefault(); input.blur(); }
                if (e.key === "Escape") { renderDeck(); }
            };
            titleRow.replaceChildren(input);
            input.focus();
            input.select();
        };
        titleEl.ondblclick = startRename;
        renameBtn.onclick = (e) => { e.stopPropagation(); startRename(); };
        titleRow.append(titleEl, renameBtn);
        cBody.appendChild(titleRow);

        const metaEl = document.createElement("div");
        metaEl.className = "h3st-media-meta";
        metaEl.textContent = kind === "image" ? "读取尺寸中…" : entry.filename;
        metaEl.title = entry.filename;
        cBody.appendChild(metaEl);

        if (kind === "image") {
            void fetchImageSizeLine(entry).then((info) => {
                if (gen !== renderGen || !metaEl.isConnected) return;
                metaEl.textContent = info.text || entry.filename;
                metaEl.title = `${entry.filename}\n${info.full || info.text || ""}`;
            });
        } else {
            const audioEl = document.createElement("audio");
            audioEl.className = "h3st-media-audio-player";
            audioEl.controls = true;
            audioEl.preload = "none";
            audioEl.src = viewUrlFor(entry);
            audioEl.draggable = false;
            audioEl.addEventListener("dragstart", (e) => { e.preventDefault(); e.stopPropagation(); });
            cBody.appendChild(audioEl);

            const trimRow = document.createElement("div");
            trimRow.className = "h3st-media-trim-row";
            const startLbl = document.createElement("span");
            startLbl.textContent = "起始(秒)";
            const startIn = document.createElement("input");
            startIn.type = "number";
            startIn.step = "0.1";
            startIn.value = String(entry.start ?? 0);
            startIn.draggable = false;
            startIn.addEventListener("dragstart", (e) => { e.preventDefault(); e.stopPropagation(); });
            startIn.addEventListener("mousedown", (e) => e.stopPropagation());
            const durLbl = document.createElement("span");
            durLbl.textContent = "时长(秒)";
            const durIn = document.createElement("input");
            durIn.type = "number";
            durIn.min = "0";
            durIn.step = "0.1";
            durIn.value = String(entry.duration ?? 0);
            durIn.draggable = false;
            durIn.addEventListener("dragstart", (e) => { e.preventDefault(); e.stopPropagation(); });
            durIn.addEventListener("mousedown", (e) => e.stopPropagation());
            const commitTrim = () => {
                const copy = getEntries(kind).map((e, idx) => {
                    if (idx !== sourceIndex) return e;
                    const next = { ...e };
                    const st = parseFloat(startIn.value) || 0;
                    const du = Math.max(0, parseFloat(durIn.value) || 0);
                    if (st) next.start = st; else delete next.start;
                    if (du) next.duration = du; else delete next.duration;
                    return next;
                });
                writeList(kind, copy);
            };
            startIn.onchange = commitTrim;
            durIn.onchange = commitTrim;
            trimRow.append(startLbl, startIn, durLbl, durIn);
            cBody.appendChild(trimRow);
        }

        // Action row:
        // Active card  -> [忽略] [替换] [删除]  (drag card to reorder!)
        // Ignored card -> [启用] [删除]
        const acts = document.createElement("div");
        acts.className = "h3st-media-actions";
        if (!isSkippedCard) {
            const ignoreBtn = document.createElement("button");
            ignoreBtn.type = "button";
            ignoreBtn.className = "h3st-media-act";
            ignoreBtn.textContent = "忽略";
            ignoreBtn.title = "暂不启用（不占槽位，移至下方已忽略区）";
            ignoreBtn.onclick = (e) => {
                e.stopPropagation();
                writeList(kind, skipCardAtIndex(getEntries(kind), sourceIndex));
            };

            const replaceBtn = document.createElement("button");
            replaceBtn.type = "button";
            replaceBtn.className = "h3st-media-act";
            replaceBtn.textContent = "替换";
            replaceBtn.title = "替换为新文件（保持当前槽位和名称不变）";
            replaceBtn.onclick = (e) => {
                e.stopPropagation();
                pickReplacement(kind, sourceIndex);
            };

            acts.append(ignoreBtn, replaceBtn);
        } else {
            const enableBtn = document.createElement("button");
            enableBtn.type = "button";
            enableBtn.className = "h3st-media-act enable-btn";
            enableBtn.textContent = "启用";
            enableBtn.title = "启用这张素材（加入上方槽位）";
            enableBtn.onclick = (e) => {
                e.stopPropagation();
                const res = restoreCardAtIndex(getEntries(kind), sourceIndex, max);
                if (res.restored) writeList(kind, res.entries);
                else statusLine.textContent = `已达槽位上限 (${max})，请先忽略一张再启用`;
            };
            acts.appendChild(enableBtn);
        }

        const delBtn = document.createElement("button");
        delBtn.type = "button";
        delBtn.className = "h3st-media-act danger";
        delBtn.textContent = "删除";
        delBtn.title = "从列表中移除这张素材（不删除本地原文件）";
        delBtn.onclick = (e) => {
            e.stopPropagation();
            writeList(kind, removeEntry(getEntries(kind), sourceIndex));
        };
        acts.appendChild(delBtn);

        cBody.appendChild(acts);
        card.appendChild(cBody);
        return card;
    }

    function renderDeck(flashIndex = null) {
        const gen = ++renderGen;
        const kind = activeTab;
        const entries = getEntries(kind);
        const max = maxFor(kind);
        const { active, parked } = splitEntries(entries);
        statusLine.textContent = `已启用 ${active.length}/${max} · 已忽略 ${parked.length}`;

        deck.innerHTML = "";
        if (entries.length === 0) {
            const empty = document.createElement("div");
            empty.style.cssText = "padding:24px;text-align:center;color:#64748b;";
            empty.textContent = kind === "image"
                ? "暂无参考图，可点「📂 读预设」恢复当前项目，或点击「＋ 添加参考图」/拖放图片到此处。"
                : "暂无参考音频，可点「📂 读预设」恢复当前项目，或点击「＋ 添加参考音频」/拖放音频到此处。";
            deck.appendChild(empty);
            return;
        }

        let flashEl = null;
        if (active.length) {
            const grid = document.createElement("div");
            grid.className = "h3st-media-grid";
            let slot = 0;
            entries.forEach((entry, idx) => {
                if (entry.skip) return;
                const isFlash = flashIndex === idx;
                const card = renderCard(kind, entry, idx, slot++, false, isFlash, gen);
                if (isFlash) flashEl = card;
                grid.appendChild(card);
            });
            deck.appendChild(grid);
        }

        if (parked.length) {
            const sec = document.createElement("div");
            sec.className = "h3st-media-section-title";
            const secLabel = document.createElement("span");
            secLabel.textContent = `已忽略（不占槽位 · ${parked.length}）`;
            const enableAllBtn = document.createElement("button");
            enableAllBtn.type = "button";
            enableAllBtn.className = "h3st-media-btn";
            enableAllBtn.textContent = "全部启用";
            enableAllBtn.onclick = () => writeList(kind, restoreAll(getEntries(kind), max).entries);
            sec.append(secLabel, enableAllBtn);
            deck.appendChild(sec);

            const pGrid = document.createElement("div");
            pGrid.className = "h3st-media-grid";
            entries.forEach((entry, idx) => {
                if (!entry.skip) return;
                const card = renderCard(kind, entry, idx, -1, true, false, gen);
                pGrid.appendChild(card);
            });
            deck.appendChild(pGrid);
        }

        if (flashEl) {
            flashEl.scrollIntoView({ behavior: "smooth", block: "nearest" });
            setTimeout(() => flashEl?.classList.remove("flash"), 1400);
        }
    }

    function renderAll() {
        renderDock();
        renderDeck();
    }

    // --- Drag-and-Drop Card Reordering + External File Drop (matches ref_loader.js) ---
    let dragIndex = -1;
    let heldIndex = -1;
    let pointerY = 0;
    let scrollFrame = 0;

    function dragTypes(event) {
        return Array.from(event.dataTransfer?.types ?? []);
    }

    function isFileDrag(event) {
        return dragTypes(event).includes("Files");
    }

    function insideDeck(event) {
        const rect = deck.getBoundingClientRect();
        return event.clientX >= rect.left && event.clientX <= rect.right
            && event.clientY >= rect.top && event.clientY <= rect.bottom;
    }

    function clearDropHints() {
        deck.classList.remove("drag-over");
        for (const card of deck.querySelectorAll(".drop-before, .drop-after")) {
            card.classList.remove("drop-before", "drop-after");
        }
    }

    function cardRects() {
        const rects = [];
        for (const card of deck.querySelectorAll(".h3st-media-card:not(.skipped)")) {
            const rect = card.getBoundingClientRect();
            rects.push({
                index: Number(card.dataset.activeIndex),
                left: rect.left,
                top: rect.top,
                right: rect.right,
                bottom: rect.bottom,
            });
        }
        return rects;
    }

    function showInsertHint(at) {
        clearDropHints();
        if (at < 0) return;
        const last = countActive(getEntries(activeTab)) - 1;
        if (at > last) {
            deck.querySelector(`[data-active-index="${last}"]`)?.classList.add("drop-after");
            return;
        }
        deck.querySelector(`[data-active-index="${at}"]`)?.classList.add("drop-before");
    }

    function startAutoScroll() {
        if (scrollFrame) return;
        const step = () => {
            scrollFrame = requestAnimationFrame(step);
            if (dragIndex < 0 || !pointerY) return;
            const rect = deck.getBoundingClientRect();
            if (pointerY < rect.top + SCROLL_EDGE) deck.scrollTop -= 12;
            else if (pointerY > rect.bottom - SCROLL_EDGE) deck.scrollTop += 12;
        };
        scrollFrame = requestAnimationFrame(step);
    }

    function stopAutoScroll() {
        if (!scrollFrame) return;
        cancelAnimationFrame(scrollFrame);
        scrollFrame = 0;
        pointerY = 0;
    }

    signal.addEventListener("abort", () => {
        clearHoverPop();
        stopAutoScroll();
        window.clearTimeout(saveTimerByKind.image);
        window.clearTimeout(saveTimerByKind.audio);
    }, { once: true });

    body.addEventListener("pointerdown", (event) => {
        const card = event.target?.closest?.(".h3st-media-card");
        heldIndex = card ? Number(card.dataset.activeIndex) : -1;
    }, { signal });

    body.addEventListener("dragstart", (event) => {
        if (event.target?.closest?.("input, button, audio")) return;
        const card = event.target?.closest?.(".h3st-media-card");
        if (!card || !deck.contains(card)) return;
        dragIndex = Number(card.dataset.activeIndex);
        if (dragIndex < 0) return; // Ignored cards are not draggable
        heldIndex = dragIndex;
        event.dataTransfer.setData(CARD_MIME, String(dragIndex));
        event.dataTransfer.setData("text/plain", String(dragIndex));
        event.dataTransfer.effectAllowed = "move";
        card.classList.add("dragging");
        startAutoScroll();
    }, { signal });

    body.addEventListener("dragover", (event) => {
        if (isFileDrag(event)) {
            if (!insideDeck(event)) return;
            event.preventDefault();
            event.stopPropagation();
            event.dataTransfer.dropEffect = "copy";
            deck.classList.add("drag-over");
            return;
        }
        if (dragIndex < 0 && dragTypes(event).includes(CARD_MIME)) dragIndex = heldIndex;
        if (dragIndex < 0) return;
        event.preventDefault();
        event.stopPropagation();
        event.dataTransfer.dropEffect = "move";
        pointerY = event.clientY;
        const at = insertPosition(cardRects(), event.clientX, event.clientY, dragIndex);
        showInsertHint(at === dragIndex || at === dragIndex + 1 ? -1 : at);
    }, { signal });

    body.addEventListener("dragleave", (event) => {
        if (body.contains(event.relatedTarget)) return;
        clearDropHints();
    }, { signal });

    body.addEventListener("drop", (event) => {
        if (isFileDrag(event) && !insideDeck(event)) return;
        event.preventDefault();
        event.stopPropagation();
        clearDropHints();
        if (dragIndex < 0 && dragTypes(event).includes(CARD_MIME)) dragIndex = heldIndex;
        const from = dragIndex;
        dragIndex = -1;
        stopAutoScroll();
        if (from < 0) {
            void uploadFiles(activeTab, Array.from(event.dataTransfer?.files || []));
            return;
        }
        const at = insertPosition(cardRects(), event.clientX, event.clientY, from);
        if (at < 0) return;
        const to = at > from ? at - 1 : at;
        if (to === from) return;
        writeList(activeTab, moveActive(getEntries(activeTab), from, to - from));
    }, { signal });

    body.addEventListener("dragend", () => {
        dragIndex = -1;
        heldIndex = -1;
        stopAutoScroll();
        for (const card of deck.querySelectorAll(".dragging")) {
            card.classList.remove("dragging");
        }
        clearDropHints();
    }, { signal });

    // --- Toolbar Event bindings ---
    imgTabBtn.onclick = () => switchTab("image");
    audTabBtn.onclick = () => switchTab("audio");
    snapBtn.onclick = () => void openSnapshotModal();
    addBtn.onclick = () => fileInput.click();
    fileInput.onchange = () => {
        const files = Array.from(fileInput.files || []);
        fileInput.value = "";
        void uploadFiles(activeTab, files);
    };
    savePresetBtn.onclick = () => void savePreset(activeTab);
    loadPresetBtn.onclick = () => void loadPreset(activeTab);
    refreshBtn.onclick = () => {
        sourceCache.clear();
        planCache.clear();
        readListsFromWidgets();
        void imgMenu.refresh();
        void audMenu.refresh();
        renderAll();
    };

    // Recompute image plans when MP/upscale widgets change
    for (const widget of [mpW, upscaleW]) {
        if (!widget) continue;
        const prevCb = widget.callback;
        widget.callback = function () {
            planCache.clear();
            const r = prevCb ? prevCb.apply(this, arguments) : undefined;
            if (activeTab === "image") renderDeck();
            return r;
        };
    }

    addPanel(node, "h3_studio_media_hub", root, 440);
    node._h3StudioMediaRefresh = () => {
        window.clearTimeout(saveTimerByKind.image);
        window.clearTimeout(saveTimerByKind.audio);
        pendingSaveByKind.image = false;
        pendingSaveByKind.audio = false;
        activeProjectByKind.image = imgMenu.name();
        activeProjectByKind.audio = audMenu.name();
        readListsFromWidgets();
        void imgMenu.refresh();
        void audMenu.refresh();
        switchTab(activeTab);
        renderDock();
    };
    readListsFromWidgets();
    switchTab("image");
    renderDock();
}
