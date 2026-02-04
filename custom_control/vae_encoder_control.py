import os

from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb

import numpy as np
import torch
import torch.nn as nn
import json
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention_processor import Attention, AttentionProcessor
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm, LayerNorm
from diffusers.utils import logging
from diffusers.utils.accelerate_utils import apply_forward_hook
from einops import repeat
from torch_cluster import fps
from tqdm import tqdm


class VAEEncoderControl:
    def __init__(
        self,
        mask_3d,
    ):
        self.mask_3d = mask_3d
        self.token_mask = None

    def register(self, vae):

        def custom_encode(
            x: torch.Tensor, num_tokens: int = 2048, seed: Optional[int] = None
        ):
           
            position_channels = vae.config.in_channels
            positions, features = x[..., :position_channels], x[..., position_channels:]
            x_kv = torch.cat([vae.embedder(positions), features], dim=-1)
            sampled_x = vae._sample_features(x, num_tokens, seed)
            
            positions, features = (
                sampled_x[..., :position_channels],
                sampled_x[..., position_channels:],
            )
            
            # 方法1: 使用 contains 判断顶点是否在编辑区域内
            self.token_mask = self.mask_3d.contains(positions[0].cpu().numpy())
            x_q = torch.cat([vae.embedder(positions), features], dim=-1)

            x = vae.encoder(x_q, x_kv)


            x = vae.quant(x)

            return x
        vae._encode = custom_encode

