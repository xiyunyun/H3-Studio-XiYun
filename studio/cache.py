"""1-slot content-hash memory caches and safe conditioning cloning for H3 Studio.

Pitfall #2 guard:
Because H3StudioMasterConfig places encoding params, seed, and pass-2 params in one
convenient node, downstream nodes MUST maintain content-keyed 1-slot caches so:
- Changing seed or pass-2 params NEVER re-runs Qwen3-VL 32B encoding.
- Changing pass-2 steps/upscale/anti-drift params NEVER re-runs Pass 1 sampling.
- Replacing a cached entry actively releases the previous tensor reference so RAM
  never balloons across long sessions.
"""
from __future__ import annotations

import hashlib
import json
import threading
from typing import Any, Optional, Tuple


class SingleSlotCache:
    """Thread-safe 1-slot cache keyed by an immutable hashable fingerprint."""

    def __init__(self, name: str):
        self.name = name
        self._lock = threading.RLock()
        self._key: Optional[str] = None
        self._value: Any = None
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Tuple[bool, Any]:
        with self._lock:
            if self._key == key and self._value is not None:
                self.hits += 1
                return True, self._value
            self.misses += 1
            return False, None

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            # Explicitly drop old reference before storing new payload.
            self._key = None
            self._value = None
            self._key = key
            self._value = value

    def clear(self) -> None:
        with self._lock:
            self._key = None
            self._value = None

    @property
    def current_key(self) -> Optional[str]:
        with self._lock:
            return self._key


ENCODE_CACHE = SingleSlotCache("h3_studio_encode")
PASS1_CACHE = SingleSlotCache("h3_studio_pass1")


def stable_hash(payload: Any) -> str:
    """Compute a deterministic SHA-256 hex digest for JSON-serializable payloads."""
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def clone_conditioning_safe(conditioning: Any) -> list:
    """Shallow-copy conditioning entries and copy minimax_keyframes / minimax_refs shells.

    Tensor buffers stay shared (zero VRAM/RAM copy cost), while dict/list containers
    are duplicated so Pass 1 / Pass 2 / Guide / Motion Context mutations can never
    contaminate the cached conditioning.
    """
    if conditioning is None:
        return []
    cloned = []
    for item in conditioning:
        emb = item[0]
        extra = dict(item[1]) if isinstance(item[1], dict) else {}
        if "minimax_keyframes" in extra and extra["minimax_keyframes"] is not None:
            extra["minimax_keyframes"] = [
                dict(kf) if isinstance(kf, dict) else kf
                for kf in extra["minimax_keyframes"]
            ]
        if "minimax_refs" in extra and extra["minimax_refs"] is not None:
            extra["minimax_refs"] = [
                dict(ref) if isinstance(ref, dict) else ref
                for ref in extra["minimax_refs"]
            ]
        cloned.append([emb, extra])
    return cloned


def model_patch_fingerprint(model: Any) -> str:
    """Best-effort content fingerprint of a ComfyUI ModelPatcher (LoRAs, patches, options).

    Changing any LoRA weight/toggle, SigmaShift, attention patch, or wrapper alters
    this digest so the Pass 1 cache invalidates reliably.
    """
    inner = getattr(model, "model", None)
    patches = getattr(model, "patches", None) or {}
    object_patches = getattr(model, "object_patches", None) or {}
    model_options = getattr(model, "model_options", None) or {}
    to = model_options.get("transformer_options") or {}

    patch_summary = []
    for key in sorted(patches.keys()):
        entries = patches[key]
        if isinstance(entries, (list, tuple)):
            sig = []
            for ent in entries:
                if isinstance(ent, tuple) and ent:
                    # (strength_patch, patch_data, strength_model, offset, function)
                    strength_p = ent[0] if len(ent) > 0 else None
                    data_id = id(ent[1]) if len(ent) > 1 else None
                    strength_m = ent[2] if len(ent) > 2 else None
                    sig.append((strength_p, data_id, strength_m))
                else:
                    sig.append(id(ent))
            patch_summary.append((key, len(entries), sig))
        else:
            patch_summary.append((key, id(entries)))

    obj_summary = [(k, id(v)) for k, v in sorted(object_patches.items())]

    wrappers = model_options.get("wrappers") or {}
    wrapper_summary = []
    if isinstance(wrappers, dict):
        for wtype, wmap in sorted(wrappers.items(), key=lambda kv: str(kv[0])):
            if isinstance(wmap, dict):
                wrapper_summary.append((str(wtype), sorted(str(k) for k in wmap.keys())))

    # Shifts and scalar transformer options
    scalar_to = {}
    for k, v in sorted(to.items()):
        if isinstance(v, (int, float, str, bool)) or v is None:
            scalar_to[k] = v
        else:
            scalar_to[k] = f"obj:{id(v)}"

    return stable_hash({
        "model_id": id(inner) if inner is not None else id(model),
        "patches_uuid": str(getattr(model, "patches_uuid", "")),
        "patches": patch_summary,
        "object_patches": obj_summary,
        "wrappers": wrapper_summary,
        "transformer_options": scalar_to,
        "patcher_id": id(model) if not patch_summary and not getattr(model, "patches_uuid", None) else 0,
    })
