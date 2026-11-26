import assert from "node:assert/strict";
import test from "node:test";

import {
    MAX_STUDIO_AUDIOS,
    MAX_STUDIO_IMAGES,
    PRESET_BALANCED,
    PRESET_FAST,
    applyCardRename,
    buildActiveDockItems,
    computeNextSegmentStep,
    continuityStatusSummary,
    displayCardLabel,
    restoreCardAtIndex,
    shouldAutoAdvanceAfterRun,
    skipCardAtIndex,
    stripFilenameExt,
} from "../../web/studio/studio_state.js";

test("stripFilenameExt removes path and final extension cleanly", () => {
    assert.equal(stripFilenameExt("露帕-四视图.png"), "露帕-四视图");
    assert.equal(stripFilenameExt("h3-lvm-refs/audio/莫宁1.mp3"), "莫宁1");
    assert.equal(stripFilenameExt("char.v2.final.webp"), "char.v2.final");
    assert.equal(stripFilenameExt("no_extension"), "no_extension");
});

test("displayCardLabel defaults to filename stem and falls back when custom label is cleared", () => {
    const initial = [
        { filename: "露帕-四视图.png", subfolder: "x", type: "input" },
        { filename: "莫宁1.mp3", label: "主唱声线", subfolder: "y", type: "input" },
    ];
    assert.equal(displayCardLabel(initial[0]), "露帕-四视图");
    assert.equal(displayCardLabel(initial[1]), "主唱声线");

    // Custom rename works
    const renamed = applyCardRename(initial, 0, "  露帕正面  ");
    assert.equal(renamed[0].label, "露帕正面");
    assert.equal(displayCardLabel(renamed[0]), "露帕正面");
    // Original list untouched (immutable)
    assert.equal(initial[0].label, undefined);

    // Clearing custom label deletes `label` key and reverts to extension-stripped filename
    const cleared = applyCardRename(renamed, 0, "   ");
    assert.equal(Object.prototype.hasOwnProperty.call(cleared[0], "label"), false);
    assert.equal(displayCardLabel(cleared[0]), "露帕-四视图");
});

test("Plan 1A Active Dock orders active images & audios, skips parked cards, and supports 1-click skip/restore", () => {
    const images = [
        { filename: "露帕-四视图.png" },
        { filename: "废图.png", skip: true },
        { filename: "爱弥斯.webp", label: "女主" },
    ];
    const audios = [
        { filename: "莫宁1.mp3", start: 0, duration: 3.2 },
        { filename: "环境音.wav", skip: true },
    ];

    const dock = buildActiveDockItems(images, audios);
    assert.equal(dock.totalActive, 3);
    assert.deepEqual(
        dock.images.map(i => [i.tag, i.label, i.sourceIndex]),
        [
            ["<Picture 1>", "露帕-四视图", 0],
            ["<Picture 2>", "女主", 2],
        ],
    );
    assert.deepEqual(
        dock.audios.map(a => [a.tag, a.label, a.trimBadge, a.sourceIndex]),
        [["<Audio 1>", "莫宁1", "0.0s+3.2s", 0]],
    );

    // Clicking × on <Picture 1> skips index 0 without touching audio
    const afterSkip = skipCardAtIndex(images, 0);
    const dock2 = buildActiveDockItems(afterSkip, audios);
    assert.deepEqual(dock2.images.map(i => [i.tag, i.label]), [["<Picture 1>", "女主"]]);
    assert.equal(afterSkip.filter(e => e.skip).length, 2);

    // Restoring parked card brings it back into active slots
    const parkedIdx = afterSkip.findIndex(e => e.filename === "露帕-四视图.png");
    const restored = restoreCardAtIndex(afterSkip, parkedIdx, MAX_STUDIO_IMAGES);
    assert.equal(restored.restored, true);
    assert.equal(buildActiveDockItems(restored.entries, audios).images.length, 2);

    // Capacity guard when active audio slots are full (3/3)
    const fullAudios = [
        { filename: "a1.mp3" },
        { filename: "a2.mp3" },
        { filename: "a3.mp3" },
        { filename: "a4.mp3", skip: true },
    ];
    const refused = restoreCardAtIndex(fullAudios, 3, MAX_STUDIO_AUDIOS);
    assert.equal(refused.restored, false);
    assert.equal(refused.reason, "full");
});

test("Continuity Hub manual next-segment button and opt-in auto-advance rules (Pitfall #9)", () => {
    // Auto-advance MUST stay off unless user explicitly checked it AND video ref is enabled
    assert.equal(shouldAutoAdvanceAfterRun({ autoAdvanceSegment: false, enableVideoRef: true }), false);
    assert.equal(shouldAutoAdvanceAfterRun({ autoAdvanceSegment: true, enableVideoRef: false }), false);
    assert.equal(shouldAutoAdvanceAfterRun({ autoAdvanceSegment: true, enableVideoRef: true, hadError: true }), false);
    assert.equal(shouldAutoAdvanceAfterRun({ autoAdvanceSegment: true, enableVideoRef: true, hadError: false }), true);

    // Manual "⏭ 下一段并接续最新" increments segment_id, sets clip_selection='latest',
    // and exits Force Initial into Auto so Shot #2 chains from Shot #1
    const step = computeNextSegmentStep({
        segmentId: 1,
        totalSegments: 5,
        clipMode: "Force Initial (强制首段，无上下文)",
    });
    assert.equal(step.segmentId, 2);
    assert.equal(step.clipSelection, "latest");
    assert.equal(step.clipMode, "Auto (首段全新 / 后续自动接续)");
    assert.equal(step.atEnd, false);

    // Clamps cleanly at the final segment
    const endStep = computeNextSegmentStep({
        segmentId: 5,
        totalSegments: 5,
        clipMode: "Auto (首段全新 / 后续自动接续)",
    });
    assert.equal(endStep.segmentId, 5);
    assert.equal(endStep.atEnd, true);
});

test("continuityStatusSummary formats badges accurately across modes and presets", () => {
    const s1 = continuityStatusSummary({
        clipMode: "Force Initial (强制首段，无上下文)",
        enableVideoRef: false,
        preset: PRESET_BALANCED,
    });
    assert.equal(s1.clipBadge, "强制首段");
    assert.equal(s1.videoBadge, "⚪关闭");

    const s2 = continuityStatusSummary({
        clipMode: "Auto (首段全新 / 后续自动接续)",
        clipSelection: "latest",
        selectedClip: { clip_id: "clip_1", shot_tag: "Shot 54", chain_depth: 1 },
        clipCount: 3,
        enableVideoRef: true,
        useVideoDuration: true,
        videoSourceMode: "segments",
        segmentId: 2,
        selectedSegment: { segment_id: 2, frame_count: 124 },
        preset: PRESET_FAST,
    });
    assert.equal(s2.clipBadge, "接续 Shot 54");
    assert.ok(s2.clipSummary.includes("第3段"));
    assert.ok(s2.clipSummary.includes("极速接续"));
    assert.equal(s2.videoBadge, "🟢开 #2");
    assert.ok(s2.videoSummary.includes("124f"));
    assert.ok(s2.videoSummary.includes("同步时长"));

    // Real /h3_lvm/segments payload uses `frames` (not `frame_count`)
    const s3 = continuityStatusSummary({
        clipMode: "Auto (首段全新 / 后续自动接续)",
        enableVideoRef: true,
        useVideoDuration: false,
        videoSourceMode: "segments",
        segmentId: 3,
        selectedSegment: { segment_id: 3, frames: 90 },
        preset: PRESET_BALANCED,
    });
    assert.ok(s3.videoSummary.includes("90f"));
});
