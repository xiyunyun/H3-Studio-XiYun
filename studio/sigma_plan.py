"""Unified sigma scheduler + self-contained H3SigmaRefiner + SplitSigmas for H3 Studio.

Matches BasicScheduler -> H3SigmaRefiner -> SplitSigmas(step=pass1_steps) bit-for-bit
with zero external dependency on ComfyUI-YCNodes-MiniMax-H3.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import torch

SCHEDULER_OPTIONS = ("simple", "normal", "karras", "beta", "sgm_uniform")
SIGMA_REFINER_SPACINGS = ("cosine", "linear", "exponential")


def refine_sigmas_h3(
    sigmas: torch.Tensor,
    extra_steps: int = 1,
    start_at_sigma: float = 0.7,
    end_at_sigma: float = 0.0,
    spacing: str = "cosine",
) -> torch.Tensor:
    """Self-contained port of H3SigmaRefiner (densifies low-sigma tail smoothly)."""
    extra_steps = int(extra_steps)
    if extra_steps <= 0:
        return sigmas

    sigmas_cpu = sigmas.detach().cpu()
    idx = -1
    for i, s in enumerate(sigmas_cpu):
        if float(s) <= float(start_at_sigma):
            idx = i
            break

    if idx == -1 or idx >= len(sigmas_cpu) - 1:
        return sigmas

    unmodified_head = sigmas_cpu[:idx]
    a_val = float(sigmas_cpu[idx].item())
    b_val = max(float(end_at_sigma), float(sigmas_cpu[-1].item()))

    original_tail_len = len(sigmas_cpu) - idx
    new_tail_len = original_tail_len + extra_steps

    t = torch.linspace(0.0, 1.0, steps=new_tail_len, dtype=torch.float32)
    if spacing == "cosine":
        factor = (1.0 - torch.cos(t * math.pi)) / 2.0
    elif spacing == "exponential":
        alpha = 3.0
        factor = (torch.exp(t * alpha) - 1.0) / (math.exp(alpha) - 1.0)
    else:  # linear
        factor = t

    new_tail = a_val + (b_val - a_val) * factor
    if float(sigmas_cpu[-1].item()) == 0.0 and b_val > 0.0:
        new_tail = torch.cat([new_tail, torch.tensor([0.0], dtype=torch.float32)])

    new_sigmas = torch.cat([unmodified_head.to(torch.float32), new_tail])
    return new_sigmas.to(device=sigmas.device, dtype=sigmas.dtype)


def split_sigmas_at(sigmas: torch.Tensor, step: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact ComfyUI SplitSigmas semantics: high = sigmas[:step+1], low = sigmas[step:]."""
    step = int(step)
    if step < 0 or step >= len(sigmas):
        raise ValueError(
            f"H3 Studio: split step {step} out of range for sigmas length {len(sigmas)}"
        )
    return sigmas[: step + 1].clone(), sigmas[step:].clone()


def build_studio_sigmas(
    model: Any,
    sampling_plan: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (full_sigmas, high_sigmas, low_sigmas) from model_sampling and sampling_plan."""
    import comfy.samplers

    pass1_steps = max(1, int(sampling_plan["pass1_steps"]))
    pass2_steps = max(1, int(sampling_plan["pass2_steps"]))
    total_steps = pass1_steps + pass2_steps
    scheduler = str(sampling_plan.get("scheduler") or "simple")

    model_sampling = model.get_model_object("model_sampling")
    base_sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, total_steps).cpu()

    refined = refine_sigmas_h3(
        base_sigmas,
        extra_steps=int(sampling_plan.get("sigma_refiner_extra_steps", 0)),
        start_at_sigma=float(sampling_plan.get("sigma_refiner_start", 0.7)),
        end_at_sigma=float(sampling_plan.get("sigma_refiner_end", 0.0)),
        spacing=str(sampling_plan.get("sigma_refiner_spacing") or "cosine"),
    )
    high_sigmas, low_sigmas = split_sigmas_at(refined, pass1_steps)
    return refined, high_sigmas, low_sigmas
