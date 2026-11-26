"""ComfyUI-H3-Studio-Xiyun: Core Studio Nodes & API Registration."""
from __future__ import annotations

from .av_upscaler import NODE_LIST as UPSCALER_NODES
from .config_hub import NODE_LIST as CONFIG_NODES
from .continuity_hub import NODE_LIST as CONTINUITY_NODES
from .dual_saver import NODE_LIST as SAVER_NODES
from .media_hub import NODE_LIST as MEDIA_NODES
from .pass1_sampler import NODE_LIST as PASS1_NODES
from .pass2_refiner import NODE_LIST as PASS2_NODES
from .studio_api import register_studio_routes
from .unified_encoder import NODE_LIST as ENCODER_NODES

NODE_LIST = [
    *MEDIA_NODES,
    *CONTINUITY_NODES,
    *CONFIG_NODES,
    *ENCODER_NODES,
    *PASS1_NODES,
    *UPSCALER_NODES,
    *PASS2_NODES,
    *SAVER_NODES,
]

__all__ = ["NODE_LIST", "register_studio_routes"]
