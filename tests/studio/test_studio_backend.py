"""Comprehensive Stage 2 backend tests for ComfyUI-H3-Studio-Xiyun."""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
import uuid
from unittest import mock

import torch

STUDIO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMFY_ROOT = os.path.dirname(os.path.dirname(STUDIO_ROOT))

for path in (COMFY_ROOT, STUDIO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)


def _load_studio_package():
    pkg_name = "h3_studio_under_test"
    if pkg_name in sys.modules:
        return sys.modules[pkg_name]
    spec = importlib.util.spec_from_file_location(
        pkg_name,
        os.path.join(STUDIO_ROOT, "__init__.py"),
        submodule_search_locations=[STUDIO_ROOT],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = mod
    spec.loader.exec_module(mod)
    return mod


PKG = _load_studio_package()
studio_types = importlib.import_module("h3_studio_under_test.studio.types")
studio_cache = importlib.import_module("h3_studio_under_test.studio.cache")
studio_sigma = importlib.import_module("h3_studio_under_test.studio.sigma_plan")
studio_ad = importlib.import_module("h3_studio_under_test.studio.anti_drift")
studio_cfg = importlib.import_module("h3_studio_under_test.studio.config_hub")
studio_enc = importlib.import_module("h3_studio_under_test.studio.unified_encoder")
studio_p1 = importlib.import_module("h3_studio_under_test.studio.pass1_sampler")
studio_up = importlib.import_module("h3_studio_under_test.studio.av_upscaler")
studio_p2 = importlib.import_module("h3_studio_under_test.studio.pass2_refiner")
studio_saver = importlib.import_module("h3_studio_under_test.studio.dual_saver")
studio_cont = importlib.import_module("h3_studio_under_test.studio.continuity_hub")
studio_media = importlib.import_module("h3_studio_under_test.studio.media_hub")
studio_api = importlib.import_module("h3_studio_under_test.studio.studio_api")
strength_patch = importlib.import_module("h3_studio_under_test.lvm.comfyui.h3_strength_patch")
guide_release_mod = importlib.import_module("h3_studio_under_test.lvm.comfyui.guide_release")


def _make_temp_dir(prefix: str) -> str:
    parent = tempfile.gettempdir()
    for _ in range(30):
        candidate = os.path.join(parent, f"{prefix}_{uuid.uuid4().hex[:8]}")
        try:
            os.mkdir(candidate)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError("failed to create temp dir")


def _make_av_latent(t_steps=37, h=8, w=6, fill=None, seed=42, dtype=torch.float32):
    import comfy.nested_tensor
    from comfy_extras.nodes_minimax_h3 import FRAME_PER_TOKEN

    frames = sum(FRAME_PER_TOKEN[k % 5] for k in range(t_steps))
    audio_t = round(frames / 24.0 * 40.0)
    gen = torch.Generator("cpu").manual_seed(seed)
    if fill is None:
        video = torch.randn((1, 24, t_steps, h, w), generator=gen, dtype=dtype)
        audio = torch.randn((1, 32, 2, audio_t), generator=gen, dtype=dtype)
    else:
        video = torch.full((1, 24, t_steps, h, w), float(fill), dtype=dtype)
        audio = torch.full((1, 32, 2, audio_t), float(fill), dtype=dtype)
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}


class _FakeClip:
    def __init__(self):
        self.tokenize_calls = 0
        self.encode_calls = 0
        self.last_ref_items = None

    def tokenize(self, prompt, minimax_ref_items=None, **kwargs):
        self.tokenize_calls += 1
        self.last_ref_items = minimax_ref_items
        return {"prompt": prompt, "n_refs": len(minimax_ref_items or [])}

    def encode_from_tokens_scheduled(self, tokens):
        self.encode_calls += 1
        return [[torch.zeros(1, 4, 8), {"prompt_marker": tokens["prompt"], "encode_call": self.encode_calls}]]


class _FakeVideoVAE:
    def __init__(self):
        self.encode_calls = 0
        self.decode_calls = 0

    def encode(self, frames):
        self.encode_calls += 1
        f, h, w, _c = frames.shape
        t = 1 if f == 1 else (2 if f <= 5 else ((f - 5) // 17) * 5 + 2)
        return torch.ones((1, 24, t, h // 16, w // 16), dtype=torch.float32) * 0.25

    def decode(self, video_lat):
        self.decode_calls += 1
        from comfy_extras.nodes_minimax_h3 import FRAME_PER_TOKEN
        t = int(video_lat.shape[2])
        frames = sum(FRAME_PER_TOKEN[k % 5] for k in range(t))
        h = int(video_lat.shape[3]) * 16
        w = int(video_lat.shape[4]) * 16
        return torch.zeros((frames, h, w, 3), dtype=torch.float32) + 0.5


class _FakeAudioVAE:
    audio_sample_rate = 32000

    def __init__(self):
        self.encode_calls = 0
        self.decode_calls = 0

    def encode(self, waveform_blc):
        self.encode_calls += 1
        samples = int(waveform_blc.shape[1])
        t = max(1, round(samples / 32000.0 * 40.0))
        return torch.ones((1, 32, 2, t), dtype=torch.float32) * 0.1

    def decode(self, audio_lat):
        self.decode_calls += 1
        t = int(audio_lat.shape[-1])
        n_samples = max(1, round(t / 40.0 * 32000.0))
        return torch.zeros((1, n_samples, 2), dtype=torch.float32)


class _FakeModel:
    def __init__(self, patches_uuid="u1"):
        import comfy.model_sampling
        self.model = object()
        self.patches = {}
        self.object_patches = {}
        self.patches_uuid = patches_uuid
        self.model_options = {"transformer_options": {"minimax_h3_sigma_shift_video": 12.0, "minimax_h3_sigma_shift_audio": 3.0}, "wrappers": {}}

        class _Cfg:
            sampling_settings = {}
        ms = comfy.model_sampling.ModelSamplingAV(_Cfg())
        ms.set_parameters(shift=12.0, audio_shift=3.0)
        self._ms = ms
        self._window_fn = lambda payload, window: payload

    def get_model_object(self, name):
        if name == "model_sampling":
            return self._ms
        if name == "_window_payload":
            return self.object_patches.get("_window_payload", self._window_fn)
        raise KeyError(name)

    def clone(self):
        c = _FakeModel(self.patches_uuid)
        c.model = self.model
        c.patches = dict(self.patches)
        c.object_patches = dict(self.object_patches)
        c.model_options = {
            "transformer_options": dict(self.model_options["transformer_options"]),
            "wrappers": {k: dict(v) for k, v in self.model_options["wrappers"].items()},
        }
        return c

    def remove_wrappers_with_key(self, wtype, key):
        self.model_options["wrappers"].setdefault(wtype, {}).pop(key, None)

    def add_wrapper_with_key(self, wtype, key, fn):
        self.model_options["wrappers"].setdefault(wtype, {})[key] = fn

    def add_object_patch(self, name, obj):
        self.object_patches[name] = obj


class TestRegistryAndSchemas(unittest.TestCase):
    def test_all_37_nodes_registered_without_duplicates(self):
        node_ids = [cls.define_schema().node_id for cls in PKG.NODE_LIST]
        self.assertEqual(len(node_ids), 37, f"Expected 28 legacy + 9 studio = 37 nodes, got {len(node_ids)}: {node_ids}")
        self.assertEqual(len(set(node_ids)), 37, "Duplicate node_id detected!")
        for expected_studio in (
            "H3StudioMediaHub",
            "H3StudioContinuityHub",
            "H3StudioMasterConfig",
            "H3StudioUnifiedEncoder",
            "H3StudioPass1Sampler",
            "H3StudioAVLatentUpscaler",
            "H3StudioPass2Refiner",
            "H3StudioDualClipSaver",
            "H3StudioBundleUnpack",
            "MiniMaxClipBinLongBuilder",
            "H3 Guide Timeline v2",
            "H3 Long Video Manager",
        ):
            self.assertIn(expected_studio, node_ids)

    def test_continuity_hub_auto_advance_defaults_false(self):
        schema = studio_cont.H3StudioContinuityHub.define_schema()
        inp = next(i for i in schema.inputs if i.id == "auto_advance_segment")
        self.assertFalse(inp.default)


class TestTypesAndZeroWastePadding(unittest.TestCase):
    def test_strong_bus_validation(self):
        with self.assertRaises(ValueError):
            studio_types.validate_media_bundle({"schema_version": 2})
        with self.assertRaises(ValueError):
            studio_types.validate_cond_pack({"schema_version": 1})
        valid = {"schema_version": 1, "images": [], "audios": [], "media_fingerprint": "abc"}
        self.assertIs(studio_types.validate_media_bundle(valid), valid)

    def test_17n_plus_5_zero_waste_padding_pitfall_12(self):
        # Exact grid counts must NEVER duplicate a single frame!
        for exact in (5, 22, 39, 56, 124, 141):
            vid = torch.arange(exact * 4 * 4 * 3, dtype=torch.float32).reshape(exact, 4, 4, 3)
            out, raw, aligned = studio_types.pad_video_to_17n_plus_5(vid)
            self.assertEqual(raw, exact)
            self.assertEqual(aligned, exact)
            self.assertEqual(int(out.shape[0]), exact)
            self.assertIs(out, vid)

        # Off-grid count pads by repeating ONLY the last frame
        vid10 = torch.zeros((10, 4, 4, 3))
        vid10[-1] = 0.77
        out, raw, aligned = studio_types.pad_video_to_17n_plus_5(vid10)
        self.assertEqual((raw, aligned, int(out.shape[0])), (10, 22, 22))
        self.assertTrue(torch.allclose(out[10:], torch.full((12, 4, 4, 3), 0.77)))

    def test_default_card_label_strips_extension_and_respects_custom_label(self):
        self.assertEqual(studio_types.display_card_label({"filename": "露帕-四视图.png"}), "露帕-四视图")
        self.assertEqual(studio_types.display_card_label({"filename": "a/b/莫宁1.mp3", "label": ""}), "莫宁1")
        self.assertEqual(studio_types.display_card_label({"filename": "露帕-四视图.png", "label": "战斗姿"}), "战斗姿")


class TestMasterConfigAndSigmaPlan(unittest.TestCase):
    def test_resolution_matches_stock_resolution_selector(self):
        from comfy_extras.nodes_resolution import ResolutionSelector
        for ratio in studio_cfg.ASPECT_RATIO_OPTIONS:
            stock_w, stock_h = ResolutionSelector.execute(ratio, 0.6, 32)
            w, h = studio_cfg.compute_resolution(ratio, 0.6, 32)
            self.assertEqual((w, h), (stock_w, stock_h), ratio)

    def test_duration_alignment(self):
        self.assertEqual(studio_cfg.compute_aligned_frames(5.0), 124)
        self.assertEqual(studio_cfg.compute_aligned_frames(6.0), 158)

    def test_sigma_refiner_parity_and_split(self):
        model = _FakeModel()
        plan = {
            "pass1_steps": 6,
            "pass2_steps": 4,
            "scheduler": "simple",
            "sigma_refiner_extra_steps": 1,
            "sigma_refiner_start": 0.7,
            "sigma_refiner_end": 0.0,
            "sigma_refiner_spacing": "cosine",
        }
        full, high, low = studio_sigma.build_studio_sigmas(model, plan)
        self.assertEqual(len(high), 7)  # 6 steps -> 7 boundary sigmas
        self.assertGreater(float(high[-1]), 0.0)
        self.assertAlmostEqual(float(low[0]), float(high[-1]), places=6)
        self.assertAlmostEqual(float(low[-1]), 0.0, places=6)
        # Verify monotonic descent
        self.assertTrue(bool(torch.all(full[:-1] > full[1:])))
        # Verify extra_steps=0 no-op
        sig = torch.tensor([1.0, 0.8, 0.5, 0.2, 0.0])
        self.assertTrue(torch.equal(studio_sigma.refine_sigmas_h3(sig, extra_steps=0), sig))
        # Verify linear & exponential spacing produce exact expected length
        lin = studio_sigma.refine_sigmas_h3(sig, extra_steps=2, start_at_sigma=0.6, spacing="linear")
        exp = studio_sigma.refine_sigmas_h3(sig, extra_steps=2, start_at_sigma=0.6, spacing="exponential")
        self.assertEqual(len(lin), len(sig) + 2)
        self.assertEqual(len(exp), len(sig) + 2)


class TestCachingPitfall1And2(unittest.TestCase):
    def setUp(self):
        studio_cache.ENCODE_CACHE.clear()
        studio_cache.PASS1_CACHE.clear()

    def test_unified_encoder_cache_hit_and_eviction(self):
        clip = _FakeClip()
        vvae = _FakeVideoVAE()
        avae = _FakeAudioVAE()
        media = {
            "schema_version": 1,
            "images": [{"tensor": torch.zeros(1, 64, 64, 3), "label": "char", "filename": "c.png", "plan": {}, "fingerprint": "img1"}],
            "audios": [],
            "media_fingerprint": "m1",
        }
        env1 = {
            "schema_version": 1, "clip": clip, "video_vae": vvae, "audio_vae": avae,
            "width": 576, "height": 1024, "manual_aligned_frames": 124,
            "prompt": "<Picture 1> dancing", "ref_image_size": "max",
        }
        pack1, _lat1, _pos1 = studio_enc.H3StudioUnifiedEncoder.execute(env1, media_bundle=media)
        self.assertEqual(clip.encode_calls, 1)

        # Mutate returned conditioning — cached object MUST remain unpolluted!
        pack1["positive"][0][1]["minimax_keyframes"] = [{"resolved_frame_index": 0}]

        # Changing canvas size in 'max' mode MUST STILL HIT CACHE (0 CLIP calls!)
        env2 = dict(env1, width=768, height=1344)
        pack2, _lat2, _pos2 = studio_enc.H3StudioUnifiedEncoder.execute(env2, media_bundle=media)
        self.assertEqual(clip.encode_calls, 1)
        self.assertNotIn("minimax_keyframes", pack2["positive"][0][1])

        # Changing prompt MUST MISS CACHE and replace the single slot
        env3 = dict(env1, prompt="<Picture 1> jumping")
        pack3, _, _ = studio_enc.H3StudioUnifiedEncoder.execute(env3, media_bundle=media)
        self.assertEqual(clip.encode_calls, 2)
        self.assertEqual(studio_cache.ENCODE_CACHE.current_key, pack3["encode_fingerprint"])

    def test_pass1_cache_skips_when_only_pass2_params_change_and_takes_slot1(self):
        model = _FakeModel()
        cond_pack = {
            "schema_version": 1,
            "positive": [[torch.zeros(1, 4, 8), {}]],
            "latent_pass1": _make_av_latent(t_steps=37, h=4, w=4, fill=0.0),
            "width_p1": 64, "height_p1": 64, "frame_count": 124, "frame_cap": 0,
            "ref_image_size": "max", "raw_ref_images": [],
            "video_vae": _FakeVideoVAE(), "audio_vae": _FakeAudioVAE(),
            "encode_fingerprint": "enc_fp_1",
        }
        plan1 = {
            "schema_version": 1, "seed": 123, "pass1_steps": 6, "pass2_steps": 4,
            "upscale_factor": 1.5, "scheduler": "simple", "sampler_name": "euler",
            "sigma_refiner_extra_steps": 1, "sigma_refiner_start": 0.7,
            "sigma_refiner_end": 0.0, "sigma_refiner_spacing": "cosine",
        }
        noisy_sentinel = _make_av_latent(t_steps=37, h=4, w=4, fill=9.9)
        denoised_sentinel = _make_av_latent(t_steps=37, h=4, w=4, fill=1.25)

        calls = {"count": 0}
        def fake_sampler(*args, **kwargs):
            calls["count"] += 1
            return noisy_sentinel, denoised_sentinel

        with mock.patch.object(studio_p1, "run_stock_sampler", side_effect=fake_sampler):
            p1_res1, lat_p1_1, _ = studio_p1.H3StudioPass1Sampler.execute(model, cond_pack, plan1)
            self.assertEqual(calls["count"], 1)
            # Pitfall #5: MUST select slot 1 (denoised_sentinel), NEVER slot 0 (noisy_sentinel)!
            self.assertIs(lat_p1_1, denoised_sentinel)
            self.assertIs(p1_res1["latent_p1"], denoised_sentinel)

            # Changing upscale_factor only MUST hit Pass 1 cache!
            plan2 = dict(plan1, upscale_factor=2.0)
            p1_res2, lat_p1_2, _ = studio_p1.H3StudioPass1Sampler.execute(model, cond_pack, plan2)
            self.assertEqual(calls["count"], 1)
            self.assertIs(lat_p1_2, denoised_sentinel)

            # Changing seed MUST miss Pass 1 cache!
            plan3 = dict(plan1, seed=456)
            studio_p1.H3StudioPass1Sampler.execute(model, cond_pack, plan3)
            self.assertEqual(calls["count"], 2)


class TestFourPillarAntiDriftAlgorithms(unittest.TestCase):
    def test_algorithm_1_progressive_taper_schedule_and_timestep_relabeling(self):
        sched = studio_ad.compute_taper_schedule(7, 0.82, 0.99, ramp_steps=2)
        self.assertEqual(sched, [0.82, 0.82, 0.82, 0.82, 0.82, 0.905, 0.99])

        orig_blocks = [torch.ones((1, 24, 1, 4, 4), dtype=torch.float32) * (k + 1) for k in range(7)]
        kfs = [{"resolved_frame_index": k, "latent": b.clone()} for k, b in enumerate(orig_blocks)]

        # Test against real h3_strength_patch installation (or mocked when model unavailable)
        coeffs = studio_ad.apply_progressive_taper(kfs, 0.82, 0.99, ramp_steps=2, install_patch=True)
        self.assertEqual(len(coeffs), 7)
        for k, (kf, orig, coeff) in enumerate(zip(kfs, orig_blocks, coeffs)):
            self.assertAlmostEqual(kf["noise_aug"], coeff, places=6)
            self.assertFalse(torch.equal(kf["latent"], orig))

        # Verify shared h3_strength_patch row_timestep_labels aligns cond segments!
        layout = types.SimpleNamespace(
            segments=[(0, 10, "video"), *[(10 + i, 11 + i, "cond") for i in range(7)], (20, 25, "ref_img")]
        )
        payload = {
            "keyframes": kfs,
            "refs": [{"latent": torch.zeros(1, 24, 1, 4, 4)}],
            "cond_video_latents": [kf["latent"] for kf in kfs] + [torch.zeros(1, 24, 1, 4, 4)],
        }
        seg_t = {"video": 0.5, "cond": 0.999, "ref_img": 0.999}
        labels = strength_patch.row_timestep_labels(payload, layout, seg_t, t_video=0.5)
        self.assertEqual(labels[0], 0.5)
        self.assertAlmostEqual(labels[1], coeffs[0], places=6)
        self.assertAlmostEqual(labels[7], coeffs[6], places=6)
        # Reference image keeps untouched 0.999!
        self.assertEqual(labels[8], 0.999)

    def test_algorithm_2_mc_release_threshold_and_pitfall_7_isolation(self):
        low_sigmas = torch.tensor([0.80, 0.63, 0.35, 0.15, 0.0])
        self.assertAlmostEqual(studio_ad.compute_mc_release_threshold(low_sigmas, 1), 0.25, places=6)
        self.assertAlmostEqual(studio_ad.compute_mc_release_threshold(low_sigmas, 2), 0.49, places=6)
        self.assertIsNone(studio_ad.compute_mc_release_threshold(low_sigmas, 0))

        mc_kf = {"resolved_frame_index": 0, "latent": torch.ones(1, 24, 1, 4, 4), studio_ad.MC_RELEASE_KEY: 0.25}
        guide_kf = {"resolved_frame_index": 40, "latent": torch.ones(1, 24, 1, 4, 4) * 2.0}
        audio_kf = {"resolved_frame_index": -5.0, "audio_latent": torch.ones(1, 32, 2, 30)}
        ref_img = {"kind": "image", "latent": torch.ones(1, 24, 1, 8, 8) * 3.0}
        payload = {
            "keyframes": [guide_kf, mc_kf, audio_kf],
            "refs": [ref_img],
            "layout": "cached_layout",
            "cond_video_latents": [guide_kf["latent"], mc_kf["latent"], ref_img["latent"]],
            "cond_audio_latents": [audio_kf["audio_latent"]],
        }

        # Above threshold: untouched
        kept_above = studio_ad.release_mc_payload(payload, sigma=0.35)
        self.assertIs(kept_above, payload)

        # Below threshold: ONLY mc_kf dropped; guide_kf, audio_kf, ref_img preserved; layout=None!
        filtered = studio_ad.release_mc_payload(payload, sigma=0.15)
        self.assertIsNone(filtered["layout"])
        self.assertEqual(len(filtered["keyframes"]), 2)
        self.assertIn(guide_kf, filtered["keyframes"])
        self.assertIn(audio_kf, filtered["keyframes"])
        self.assertEqual(len(filtered["refs"]), 1)
        self.assertEqual(len(filtered["cond_video_latents"]), 2)  # guide + ref_img
        self.assertEqual(len(filtered["cond_audio_latents"]), 1)  # audio preserved!

        # Verify _window_payload metadata preservation across slicing
        def fake_native_window(pl, win):
            # Simulate core reconstructing keyframe dicts without custom keys
            rebuilt = []
            for k in pl.get("keyframes") or []:
                base = {"resolved_frame_index": k["resolved_frame_index"]}
                if "latent" in k:
                    base["latent"] = k["latent"]
                if "audio_latent" in k:
                    base["audio_latent"] = k["audio_latent"]
                rebuilt.append(base)
            return dict(pl, keyframes=rebuilt)

        wrapped_win = studio_ad.preserve_mc_window_metadata(fake_native_window)
        mc_kf["noise_aug"] = 0.85
        guide_kf[studio_ad.GUIDE_RELEASE_KEY] = 0.60
        sliced = wrapped_win(payload, (0, 100))
        mc_after = [k for k in sliced["keyframes"] if k["resolved_frame_index"] == 0][0]
        guide_after = [k for k in sliced["keyframes"] if k["resolved_frame_index"] == 40][0]
        self.assertEqual(mc_after.get(studio_ad.MC_RELEASE_KEY), 0.25)
        self.assertEqual(mc_after.get("noise_aug"), 0.85)
        self.assertEqual(guide_after.get(studio_ad.GUIDE_RELEASE_KEY), 0.60)

    def test_algorithm_3_asymmetric_22_to_5_alignment_pitfall_6(self):
        target = _make_av_latent(t_steps=37, h=6, w=6, fill=0.0)   # 124 frames
        parent = _make_av_latent(t_steps=37, h=6, w=6, seed=7)     # 124 frames
        base_cond = [[torch.zeros(1, 4, 8), {"minimax_keyframes": [{"resolved_frame_index": 80, "latent": torch.zeros(1, 24, 1, 6, 6)}]}]]

        cond_p2, trim_p1, info = studio_ad.inject_motion_context_studio(
            base_cond,
            target,
            parent,
            context_length=5,
            trim_frames_p1=22,
            audio_context_length=18,
            enable_audio_context=True,
            taper_enabled=True,
            taper_start=0.88,
            taper_seam=0.995,
            release_threshold=0.25,
        )
        self.assertEqual(trim_p1, 22)
        self.assertEqual(info["frame_offset"], 17)
        # 5 frames = 2 steps with local offsets [0, 1] -> anchored at [17, 18] (NEVER [0, 1]!)
        self.assertEqual(info["video_indices"], [17, 18])
        # Audio end_frame MUST stay anchored near 22 (NOT 5!)
        self.assertAlmostEqual(info["audio_end_frame"], 22.0, delta=0.5)

        kfs = cond_p2[0][1]["minimax_keyframes"]
        # Upstream guide at frame 80 preserved without MC release tag!
        guide_kfs = [k for k in kfs if k["resolved_frame_index"] == 80]
        self.assertEqual(len(guide_kfs), 1)
        self.assertNotIn(studio_ad.MC_RELEASE_KEY, guide_kfs[0])
        # Audio keyframe has NO MC release tag and NO noise_aug!
        audio_kfs = [k for k in kfs if "audio_latent" in k]
        self.assertEqual(len(audio_kfs), 1)
        self.assertNotIn(studio_ad.MC_RELEASE_KEY, audio_kfs[0])
        self.assertNotIn("noise_aug", audio_kfs[0])

    def test_algorithm_4_order_root_anchor_and_scene_shift_guard_pitfall_8(self):
        # Static shot: parent has constant pattern across steps; child drifts +0.15 in mean
        parent = _make_av_latent(t_steps=37, h=6, w=6, seed=101)
        pv, _pa = parent["samples"].unbind()
        # Make a smooth static sequence so shot_motion is low (<= 0.48)
        base_frame = pv[:, :, :5].clone()
        pv_static = base_frame.repeat(1, 1, 8, 1, 1)[:, :, :37].clone()
        import comfy.nested_tensor
        parent_static = {"samples": comfy.nested_tensor.NestedTensor((pv_static, _pa.clone()))}
        root_stats = studio_ad.compute_latent_stats(parent_static, trim_frames=0)

        # Child starts from parent_static, drifted slightly (+0.12 mean shift across new frames)
        cv = pv_static.clone()
        cv[:, :, 7:] += 0.12
        # Corrupt head slightly so we prove OverlapRestore restores exact parent tail AFTER JoinMatch
        cv[:, :, :7] += 0.50
        child_audio = torch.randn_like(_pa)
        child = {"samples": comfy.nested_tensor.NestedTensor((cv, child_audio.clone()))}

        out, diag = studio_ad.apply_pass2_anti_drift_postprocess(
            child,
            context_latent=parent_static,
            trim_frames_p1=22,
            root_stats=root_stats,
            chain_depth=2,
            join_match_enabled=True,
            join_match_strength=1.0,
            root_anchor_strength=0.30,
            overlap_restore_p2=True,
        )
        # Strict Pitfall #8 order verified!
        self.assertEqual(diag["order"], ["join_match", "root_anchor", "overlap_restore"])
        out_v, out_a = out["samples"].unbind()
        # Audio bit-for-bit unchanged!
        self.assertTrue(torch.equal(out_a, child_audio))
        # Head 7 steps (22 frames) MUST equal parent's exact last 7 steps!
        self.assertTrue(torch.equal(out_v[:, :, :7], pv_static[:, :, -7:]))

        # Scene-Shift Guard test: huge deliberate brightness shift (rel_shift > 0.60) -> eff_w == 0!
        cv_night = pv_static.clone()
        cv_night[:, :, 7:] -= 5.0
        child_night = {"samples": comfy.nested_tensor.NestedTensor((cv_night, child_audio.clone()))}
        out_night, ra_diag = studio_ad.apply_root_anchor_pull(
            child_night, root_stats, trim_frames=22, chain_depth=3, root_anchor_strength=0.45
        )
        self.assertFalse(ra_diag["applied"])
        self.assertEqual(ra_diag["reason"], "scene_shift_guard")
        self.assertEqual(ra_diag["eff_weight"], 0.0)
        self.assertIs(out_night, child_night)


class TestAVUpscalerAndPass2AndDualSaver(unittest.TestCase):
    def test_upscaler_sd_normalization_and_prefix_stripping(self):
        raw_sd = {
            "model": {
                "upscaler.conv_in.weight": torch.zeros(512, 24, 3, 3, 3, dtype=torch.float16),
                "upscaler.in_blocks.0.in_layers.0.weight": torch.zeros(512, dtype=torch.float16),
                "upscaler.in_blocks.1.dwconv.weight": torch.zeros(512, 1, 5, 1, 1, dtype=torch.float16),
                "upscaler.out_blocks.0.in_layers.0.weight": torch.zeros(512, dtype=torch.float16),
            }
        }
        if hasattr(torch, "float8_e4m3fn"):
            raw_sd["model"]["upscaler.conv_out.weight"] = torch.zeros(24, 512, 3, 3, 3).to(torch.float8_e4m3fn)
        clean, cfg = studio_up.normalize_upscaler_state_dict(raw_sd)
        self.assertIn("conv_in.weight", clean)
        self.assertNotIn("upscaler.conv_in.weight", clean)
        if hasattr(torch, "float8_e4m3fn"):
            self.assertEqual(clean["conv_out.weight"].dtype, torch.float16)
        self.assertEqual(cfg["in_channels"], 24)
        self.assertEqual(cfg["channels"], 512)
        self.assertEqual(cfg["in_blocks"], 1)
        self.assertEqual(cfg["out_blocks"], 1)
        self.assertEqual(cfg["temporal_kernel"], 5)

    def test_upscaler_memory_managed_av_passthrough_and_self_unload_only(self):
        p1_lat = _make_av_latent(t_steps=7, h=4, w=4, seed=5)
        _v, orig_audio = p1_lat["samples"].unbind()
        p1_res = {
            "schema_version": 1, "latent_p1": p1_lat,
            "low_sigmas": torch.tensor([0.5, 0.0]), "high_sigmas": torch.tensor([1.0, 0.5]),
            "trim_frames_p1": 0, "seed": 1, "pass1_fingerprint": "fp",
        }
        plan = {
            "schema_version": 1, "seed": 1, "pass1_steps": 6, "pass2_steps": 4,
            "upscale_factor": 1.5, "scheduler": "simple", "sampler_name": "euler",
            "sigma_refiner_extra_steps": 0, "sigma_refiner_start": 0.7,
            "sigma_refiner_end": 0.0, "sigma_refiner_spacing": "cosine",
        }

        class _FakeCoreUpscaler:
            def memory_required(self, shape, size):
                return 123456
            def get_dtype(self):
                return torch.float32
            def __call__(self, x, size):
                return torch.zeros((x.shape[0], x.shape[1], size[0], size[1], size[2]), dtype=x.dtype, device=x.device)

        fake_patcher = types.SimpleNamespace(
            model=_FakeCoreUpscaler(),
            load_device=torch.device("cpu"),
            model_dtype=lambda: torch.float32,
        )

        import comfy.model_management as mm
        with mock.patch.object(studio_up, "_load_or_get_upscaler_patcher", return_value=fake_patcher), \
             mock.patch.object(mm, "load_models_gpu") as m_load, \
             mock.patch.object(mm, "unload_model_and_clones") as m_unload, \
             mock.patch.object(mm, "unload_all_models") as m_unload_all:
            up_lat, w_px, h_px = studio_up.H3StudioAVLatentUpscaler.execute(
                p1_res, plan, model_name="fake.safetensors", align=32, precision="fp32", release_after=True
            )
            m_load.assert_called_once_with([fake_patcher], memory_required=123456)
            m_unload.assert_called_once_with(fake_patcher)
            m_unload_all.assert_not_called()
            self.assertEqual((w_px, h_px), (96, 96))
            up_v, up_a = up_lat["samples"].unbind()
            self.assertEqual(tuple(up_v.shape), (1, 24, 7, 6, 6))
            self.assertTrue(torch.equal(up_a, orig_audio))

    def test_pass2_reuses_32b_takes_slot0_and_trims_cleanly(self):
        model = _FakeModel()
        vvae = _FakeVideoVAE()
        avae = _FakeAudioVAE()
        clip = _FakeClip()
        up_lat = _make_av_latent(t_steps=12, h=6, w=6, seed=9)  # 39 frames
        p1_res = {
            "schema_version": 1, "latent_p1": _make_av_latent(t_steps=12, h=4, w=4, seed=8),
            "low_sigmas": torch.tensor([0.8, 0.4, 0.0]), "high_sigmas": torch.tensor([1.0, 0.8]),
            "trim_frames_p1": 0, "seed": 7, "pass1_fingerprint": "p1fp",
        }
        cond_pack = {
            "schema_version": 1, "positive": [[torch.zeros(1, 4, 8), {}]],
            "latent_pass1": p1_res["latent_p1"], "width_p1": 64, "height_p1": 64,
            "frame_count": 39, "frame_cap": 30, "ref_image_size": "max", "raw_ref_images": [],
            "video_vae": vvae, "audio_vae": avae, "encode_fingerprint": "efp",
        }
        plan = {
            "schema_version": 1, "seed": 7, "pass1_steps": 6, "pass2_steps": 4,
            "upscale_factor": 1.5, "scheduler": "simple", "sampler_name": "euler",
            "sigma_refiner_extra_steps": 0, "sigma_refiner_start": 0.7,
            "sigma_refiner_end": 0.0, "sigma_refiner_spacing": "cosine",
        }
        slot0_sentinel = _make_av_latent(t_steps=12, h=6, w=6, fill=0.42)
        slot1_sentinel = _make_av_latent(t_steps=12, h=6, w=6, fill=9.99)

        with mock.patch.object(studio_p2, "run_stock_sampler", return_value=(slot0_sentinel, slot1_sentinel)):
            imgs, aud, lat_p2, ex_imgs, ex_aud, report = studio_p2.H3StudioPass2Refiner.execute(
                model, up_lat, p1_res, cond_pack, plan, extra_trim_frames=2
            )
        # Pitfall #1: ZERO CLIP calls!
        self.assertEqual(clip.encode_calls, 0)
        # Pitfall #5: Pass 2 MUST take slot 0 (output)!
        self.assertIs(lat_p2, slot0_sentinel)
        # frame_cap=30 applied cleanly
        self.assertEqual(int(imgs.shape[0]), 30)
        self.assertEqual(int(ex_imgs.shape[0]), 28)
        rep = json.loads(report)
        self.assertTrue(rep["skipped_32b_reload"])
        self.assertEqual(rep["highres_size"], [96, 96])

    def test_dual_saver_and_continuity_hub_root_lineage_roundtrip(self):
        tmp_out = _make_temp_dir("h3_studio_bin")
        try:
            import folder_paths
            with mock.patch.object(folder_paths, "get_output_directory", return_value=tmp_out):
                lat_p1 = _make_av_latent(t_steps=12, h=4, w=4, seed=11)
                lat_p2 = _make_av_latent(t_steps=12, h=6, w=6, seed=12)
                imgs = torch.zeros((10, 32, 32, 3), dtype=torch.float32) + 0.3
                aud = {"waveform": torch.zeros((1, 2, 16000)), "sample_rate": 32000}

                # Shot 1 (initial)
                pipe1, _vref1 = studio_cont.H3StudioContinuityHub.execute(
                    clip_project="StudioTest", clip_mode=studio_cont.CLIP_MODES[0], clip_selection="latest"
                )
                self.assertTrue(pipe1["is_initial"])
                self.assertEqual(pipe1["chain_depth"], 0)

                p1_res1 = {
                    "schema_version": 1, "latent_p1": lat_p1,
                    "low_sigmas": torch.tensor([0.5, 0.0]), "high_sigmas": torch.tensor([1.0, 0.5]),
                    "trim_frames_p1": 0, "seed": 1, "pass1_fingerprint": "a",
                }
                clip1_id, _prev1, proj1 = studio_saver.H3StudioDualClipSaver.execute(
                    p1_res1, lat_p2, imgs, aud, pipe1, prompt="shot1", save_video=False
                )
                self.assertEqual(proj1, "StudioTest")

                # Shot 2 loads Shot 1 as parent & root!
                pipe2, _vref2 = studio_cont.H3StudioContinuityHub.execute(
                    clip_project="StudioTest", clip_mode=studio_cont.CLIP_MODES[0], clip_selection="latest"
                )
                self.assertFalse(pipe2["is_initial"])
                self.assertEqual(pipe2["parent_clip_id"], clip1_id)
                self.assertEqual(pipe2["root_clip_id"], clip1_id)
                self.assertEqual(pipe2["chain_depth"], 1)
                self.assertIsNotNone(pipe2["root_stats"])
                self.assertEqual(len(pipe2["root_stats"]["mean"]), 24)

                # Save Shot 2, then Shot 3 traces back to Shot 1 with chain_depth=2!
                clip2_id, _prev2, _ = studio_saver.H3StudioDualClipSaver.execute(
                    p1_res1, lat_p2, imgs, aud, pipe2, prompt="shot2", save_video=False
                )
                pipe3, _ = studio_cont.H3StudioContinuityHub.execute(
                    clip_project="StudioTest", clip_mode=studio_cont.CLIP_MODES[0], clip_selection="latest"
                )
                self.assertEqual(pipe3["parent_clip_id"], clip2_id)
                self.assertEqual(pipe3["root_clip_id"], clip1_id)
                self.assertEqual(pipe3["chain_depth"], 2)

                # Snapshot API roundtrip
                saved = studio_api.save_media_snapshot({
                    "name": "露帕+莫宁",
                    "image_project": "鸣潮3D",
                    "audio_project": "鸣潮音频库",
                    "image_files": [{"filename": "露帕.png"}],
                    "audio_files": [{"filename": "莫宁.mp3"}],
                })
                self.assertEqual(saved["name"], "露帕+莫宁")
                listed = studio_api.list_media_snapshots()
                self.assertEqual(len(listed), 1)
                self.assertEqual(listed[0]["image_count"], 1)
                self.assertTrue(studio_api.delete_media_snapshot("露帕+莫宁"))
                self.assertEqual(studio_api.list_media_snapshots(), [])
        finally:
            shutil.rmtree(tmp_out, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
