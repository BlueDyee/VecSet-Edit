import os

from typing import Callable, List, Optional, Tuple, Union, Dict, Any

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb

import numpy as np
import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention_processor import Attention, AttentionProcessor
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm, LayerNorm
from diffusers.utils import logging
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_version,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from einops import repeat
from torch_cluster import fps
from tqdm import tqdm
from triposg.models.transformers.modeling_outputs import Transformer1DModelOutput
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name





class CustomAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the TripoSG model. It applies a s normalization layer and rotary embedding on query and key vector.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # NOTE that pre-trained models split heads first then split qkv or kv, like .view(..., attn.heads, 3, dim)
        # instead of .view(..., 3, attn.heads, dim). So we need to re-split here.
        if not attn.is_cross_attention:
            qkv = torch.cat((query, key, value), dim=-1)
            split_size = qkv.shape[-1] // attn.heads // 3
            qkv = qkv.view(batch_size, -1, attn.heads, split_size * 3)
            query, key, value = torch.split(qkv, split_size, dim=-1)
        else:
            kv = torch.cat((key, value), dim=-1)
            split_size = kv.shape[-1] // attn.heads // 2
            kv = kv.view(batch_size, -1, attn.heads, split_size * 2)
            key, value = torch.split(kv, split_size, dim=-1)

        head_dim = key.shape[-1]

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            if not attn.is_cross_attention:
                key = apply_rotary_emb(key, image_rotary_emb)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        # ‼️手动计算 attention scores for Debugging
        #print("Query shape:", query.shape)
        #print("Key shape:", key.shape)
        custom_info={}
        query_dim = query.shape[-1]  
        scale = 1.0 / (query_dim ** 0.5)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        attention_probs = F.softmax(attention_scores, dim=-1)
        # 保存 attention_probs 和 attention
        custom_info["attn_probs"] = attention_probs
        custom_info["attn_vals"] = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        custom_info["value"] = value
        # ‼️
        
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)
        
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )
        
        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states, custom_info

class DiTBlockControl:
    def __init__(self,store_prob=True,store_attn=False, store_value=True, step_list=None, batch_index=1,self_attention=False):
        self.batch_index = batch_index # batch index 0=negative, 1=positive
        self.call_count = 0
        self.step_list=step_list
        self.prob_list=[]
        self.attn_list=[]
        self.value_list=[]
        self.store_prob=store_prob
        self.store_attn=store_attn
        self.store_value=store_value
        self.attn2_processor = CustomAttnProcessor2_0()
        self.self_attention=self_attention
        if self.self_attention:
            self.attn1_processor = CustomAttnProcessor2_0()
            self.prob_list_self=[]
            self.attn_list_self=[]
            self.value_list_self=[]
            self.original_attn1_processor = None
        
        # Store original methods for unregistering
        self.original_forward = None
        self.original_attn2_processor = None
        self.registered_block = None
    
    def register(self, block):
        # Store references for unregistering
        self.registered_block = block
        self.original_forward = block.forward
        self.original_attn2_processor = block.attn2.processor
        # 保存原始 forward 方法
        block.attn2.set_processor(self.attn2_processor)

        if self.self_attention:
            self.original_attn1_processor = block.attn1.processor
            block.attn1.set_processor(self.attn1_processor)
        @torch.no_grad()
        @apply_forward_hook
        def custom_forward(
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_hidden_states_2: Optional[torch.Tensor] = None,
            temb: Optional[torch.Tensor] = None,
            image_rotary_emb: Optional[torch.Tensor] = None,
            skip: Optional[torch.Tensor] = None,
            attention_kwargs= None,
            ) -> torch.Tensor:
            self.call_count+=1
            # Prepare attention kwargs
            attention_kwargs = attention_kwargs.copy() if attention_kwargs is not None else {}
            cross_attention_scale = attention_kwargs.pop("cross_attention_scale", 1.0)
            cross_attention_2_scale = attention_kwargs.pop("cross_attention_2_scale", 1.0)
            # 0. Long Skip Connection
            if block.skip_linear is not None:
                cat = torch.cat(
                    (
                        [skip, hidden_states]
                        if block.skip_concat_front
                        else [hidden_states, skip]
                    ),
                    dim=-1,
                )
                if block.skip_norm_last:
                    # don't do this
                    hidden_states = block.skip_linear(cat)
                    hidden_states = block.skip_norm(hidden_states)
                else:
                    cat = block.skip_norm(cat)
                    hidden_states = block.skip_linear(cat)

            # 1. Self-Attention
            if block.use_self_attention:
                norm_hidden_states = block.norm1(hidden_states)
                if self.self_attention:
                    attn_output, custom_info_self= block.attn1(
                        norm_hidden_states,
                        image_rotary_emb=image_rotary_emb,
                        **attention_kwargs,
                    )
                else:
                    attn_output = block.attn1(
                        norm_hidden_states,
                        image_rotary_emb=image_rotary_emb,
                        **attention_kwargs,
                    )
                hidden_states = hidden_states + attn_output

            # 2. Cross-Attention
            if block.use_cross_attention:
                if block.use_cross_attention_2:
                    hidden_states = (
                        hidden_states
                        + block.attn2(
                            block.norm2(hidden_states),
                            encoder_hidden_states=encoder_hidden_states,
                            image_rotary_emb=image_rotary_emb,
                            **attention_kwargs,
                        ) * cross_attention_scale
                        + block.attn2_2(
                            block.norm2_2(hidden_states),
                            encoder_hidden_states=encoder_hidden_states_2,
                            image_rotary_emb=image_rotary_emb,
                            **attention_kwargs,
                        ) * cross_attention_2_scale
                    )
                else:
                    # This attn2 is overide by custom_attention processor
                    attn_output, custom_info= block.attn2(
                        block.norm2(hidden_states),
                        encoder_hidden_states=encoder_hidden_states,
                        image_rotary_emb=image_rotary_emb,
                        **attention_kwargs,
                    ) 
                    hidden_states = hidden_states + attn_output* cross_attention_scale
                    

            # FFN Layer ### TODO: switch norm2 and norm3 in the state dict
            mlp_inputs = block.norm3(hidden_states)
            hidden_states = hidden_states + block.ff(mlp_inputs)
            # 🗼Store Debug information from custom attention processor
            if  (self.step_list is None) or (self.call_count in self.step_list):
                
                if self.store_prob:
                    attn_probs = custom_info["attn_probs"][self.batch_index]
                    self.prob_list.append(attn_probs.mean(dim=0))
                    if self.self_attention:
                        attn_probs = custom_info_self["attn_probs"][self.batch_index]
                        self.prob_list_self.append(attn_probs.mean(dim=0))
                if self.store_attn:
                    self.attn_list.append(custom_info["attn_vals"][self.batch_index])
                    if self.self_attention:
                        self.attn_list_self.append(custom_info_self["attn_vals"][self.batch_index])
                if self.store_value:
                    H, N, C = custom_info["value"][self.batch_index].shape
                    values=custom_info["value"][self.batch_index]
                    # Concat Head dimension and Channel dimension
                    values = values.permute(1, 0, 2).contiguous().view(N,-1)
                    self.value_list.append(values)
                    if self.self_attention:
                        H, N, C = custom_info_self["value"][self.batch_index].shape
                        values=custom_info_self["value"][self.batch_index]
                        # Concat Head dimension and Channel dimension
                        values = values.permute(1, 0, 2).contiguous().view(N,-1)
                        self.value_list_self.append(values)
            # 🗼

            return hidden_states
        block.forward = custom_forward

    def unregister(self):
        """
        Unregister the DiTBlockControl and restore original methods
        """
        if self.registered_block is None:
            print("⚠️ DiTBlockControl was not properly registered. Nothing to unregister.")
            return
            
        try:
            # Restore original forward method
            if self.original_forward is not None:
                self.registered_block.forward = self.original_forward
                
            # Restore original attention processor
            if self.original_attn2_processor is not None:
                self.registered_block.attn2.set_processor(self.original_attn2_processor)
            if self.self_attention and (self.original_attn1_processor is not None):
                self.registered_block.attn1.set_processor(self.original_attn1_processor)
            
            # Clear stored references
            self.registered_block = None
            self.original_forward = None
            self.original_attn2_processor = None
            self.original_attn1_processor = None
            
            # Clear stored data
            self.prob_list = []
            self.attn_list = []
            self.value_list = []
            self.call_count = 0
            if self.self_attention:
                self.prob_list_self = []
                self.attn_list_self = []
                self.value_list_self = []
        
            
        except Exception as e:
            print(f"❌ Error during unregistering DiTBlockControl: {e}")
            raise
class TripoSGDiTControl:
    def __init__(self,block_list=None,
                 store_prob=True,
                 store_attn=False,
                 store_value=False,
                 step_list=None,
                 batch_index=1,
                 self_attention=False,
                 self_attention_device="GPU"):
        self.call_count = 0
        if block_list is None:
            print("🌞 Block_list=None: assign totall 20 blocks for visualization")
            block_list=[i for i in range(20)]
        self.idx2control = {}
        self.block_list=block_list
        
        # Store original methods for unregistering
        self.original_dit_forward = None
        self.registered_dit = None
        self.store_prob=store_prob
        self.store_attn=store_attn
        self.store_value=store_value
        self.self_attention=self_attention
        self.self_attention_device=self_attention_device
    def register(self,dit):
        # Store references for unregistering
        self.registered_dit = dit
        self.original_dit_forward = dit.forward
        
        for idx, block in enumerate(dit.blocks):
            if idx in self.block_list:
                control = DiTBlockControl(store_prob=self.store_prob,
                                        store_attn=self.store_attn,
                                        store_value=self.store_value,
                                        self_attention=self.self_attention,
                                        step_list=None,
                                        batch_index=1)
                control.register(block)
                self.idx2control[idx] = control
        # exit()
        # Use self->cross1 for every block // no cross2
        #block.use_self_attention=True,  block.use_cross_attention=True,  block.use_cross_attention_2=False
        @torch.no_grad()
        @apply_forward_hook
        def custom_dit_forward(
            hidden_states: Optional[torch.Tensor],
            timestep: Union[int, float, torch.LongTensor],
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_hidden_states_2: Optional[torch.Tensor] = None,
            image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            attention_kwargs: Optional[Dict[str, Any]] = None,
            return_dict: bool = True,
        ):
            """
            The [`HunyuanDiT2DModel`] forward method.

            Args:
            hidden_states (`torch.Tensor` of shape `(batch size, dim, height, width)`):
                The input tensor.
            timestep ( `torch.LongTensor`, *optional*):
                Used to indicate denoising step.
            encoder_hidden_states ( `torch.Tensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer.
            return_dict: bool
                Whether to return a dictionary.
            """
            if attention_kwargs is not None:
                attention_kwargs = attention_kwargs.copy()
                lora_scale = attention_kwargs.pop("scale", 1.0)
            else:
                lora_scale = 1.0

            if USE_PEFT_BACKEND:
                # weight the lora layers by setting `lora_scale` for each PEFT layer
                scale_lora_layers(dit, lora_scale)
            else:
                if (
                    attention_kwargs is not None
                    and attention_kwargs.get("scale", None) is not None
                ):
                    logger.warning(
                        "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                    )

            _, N, _ = hidden_states.shape

            temb = dit.time_embed(timestep).to(hidden_states.dtype)
            temb = dit.time_proj(temb)
            temb = temb.unsqueeze(dim=1)  # unsqueeze to concat with hidden_states

            hidden_states = dit.proj_in(hidden_states)

            # N + 1 token
            hidden_states = torch.cat([temb, hidden_states], dim=1)

            skips = []
            for layer, block in enumerate(dit.blocks):
                skip = None if layer <= dit.config.num_layers // 2 else skips.pop()

                if dit.training and dit.gradient_checkpointing:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = (
                        {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    )
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        encoder_hidden_states_2,
                        temb,
                        image_rotary_emb,
                        skip,
                        attention_kwargs,
                        **ckpt_kwargs,
                    )
                else:
                    hidden_states = block(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_hidden_states_2=encoder_hidden_states_2,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        skip=skip,
                        attention_kwargs=attention_kwargs,
                    )  # (N, L, D)

                if layer < dit.config.num_layers // 2:
                    skips.append(hidden_states)

            # final layer
            hidden_states = dit.norm_out(hidden_states)
            hidden_states = hidden_states[:, -N:]
            hidden_states = dit.proj_out(hidden_states)

            if USE_PEFT_BACKEND:
                # remove `lora_scale` from each PEFT layer
                unscale_lora_layers(dit, lora_scale)

            if not return_dict:
                return (hidden_states,)

            return Transformer1DModelOutput(sample=hidden_states)
        dit.forward = custom_dit_forward

    def unregister(self):
        """
        Unregister the TripoSGDiTControl and restore original methods
        """
        if self.registered_dit is None:
            print("⚠️ TripoSGDiTControl was not properly registered. Nothing to unregister.")
            return
            
        try:
            # Unregister all individual block controls
            for idx, control in self.idx2control.items():
                control.unregister()
            
            # Restore original dit forward method
            if self.original_dit_forward is not None:
                self.registered_dit.forward = self.original_dit_forward
            
            # Clear stored references
            self.registered_dit = None
            self.original_dit_forward = None
            self.idx2control = {}
            self.call_count = 0
            torch.cuda.empty_cache()
            print("✅ TripoSGDiTControl successfully unregistered.")
            
        except Exception as e:
            print(f"❌ Error during unregistering TripoSGDiTControl: {e}")
            raise

                
def save_grid_logits_as_pointcloud(grid_logits: torch.Tensor, 
                                   output_path: str = "grid_logits_pointcloud",
                                   format_type: str = "ply",
                                   threshold: float = -5.0,
                                   downsample_factor: int = 1,
                                   colormap: str = "coolwarm"):
    """
    將3D grid logits轉換為點雲格式並儲存
    
    適合的3D點雲格式：
    1. PLY (Polygon File Format) - 最常用，支援顏色
    2. PCD (Point Cloud Data) - PCL library格式
    3. XYZ - 簡單文字格式
    4. OFF - Object File Format
    
    Args:
        grid_logits: shape (H, W, D) 的3D tensor
        output_path: 輸出檔案路徑（不含副檔名）
        format_type: 格式類型 ("ply", "xyz", "pcd")
        threshold: 只保存大於此閾值的點（過濾背景）
        downsample_factor: 下採樣因子，減少點的數量
        colormap: 色彩映射類型 ("viridis", "plasma", "inferno", "magma", "hot", "coolwarm", "RdYlBu_r")
    """
    
    if isinstance(grid_logits, torch.Tensor):
        grid_logits = grid_logits.cpu().numpy()
    
    H, W, D = grid_logits.shape
    print(f"Original grid shape: {grid_logits.shape}")
    print(f"Value range: {grid_logits.min():.3f} to {grid_logits.max():.3f}")
    
    # 生成3D座標網格
    x_coords, y_coords, z_coords = np.meshgrid(
        np.arange(0, H, downsample_factor),
        np.arange(0, W, downsample_factor), 
        np.arange(0, D, downsample_factor),
        indexing='ij'
    )
    
    # 下採樣grid_logits
    downsampled_logits = grid_logits[::downsample_factor, ::downsample_factor, ::downsample_factor]
    
    # 扁平化所有數據
    points = np.stack([x_coords.flatten(), y_coords.flatten(), z_coords.flatten()], axis=1)
    values = downsampled_logits.flatten()
    
    # 根據閾值過濾點
    mask = values > threshold
    filtered_points = points[mask]
    filtered_values = values[mask]
    
    print(f"Filtered points: {len(filtered_points)} / {len(points)}")
    
    # 正規化到0-1範圍
    min_val, max_val = filtered_values.min(), filtered_values.max()
    normalized_values = (filtered_values - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(filtered_values)
    
    # 也可以選擇其他色彩映射: 'plasma', 'inferno', 'magma', 'cividis', 'hot', 'coolwarm'
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    # 選擇色彩映射 - 提供多種heatmap顏色選擇
    colormap_obj = cm.get_cmap(colormap)
    
    # 將正規化值映射到RGB顏色
    colors_rgba = colormap_obj(normalized_values)
    colors = (colors_rgba[:, :3] * 255).astype(np.uint8)  # 轉換為0-255範圍的RGB
    
    if format_type.lower() == "ply":
        output_file = f"{output_path}.ply"
        save_ply_pointcloud(filtered_points, colors, filtered_values, output_file)
    else:
        raise ValueError(f"Unsupported format: {format_type}")
    
    print(f"Point cloud saved to: {output_file}")
    return output_file

def save_ply_pointcloud(points, colors, values, output_path):
    """儲存為PLY格式（最推薦，支援多數3D軟體）"""
    with open(output_path, 'w') as f:
        # PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property float value\n")
        f.write("end_header\n")
        
        # 點數據
        for i in range(len(points)):
            f.write(f"{points[i,0]:.3f} {points[i,1]:.3f} {points[i,2]:.3f} "
                   f"{colors[i,0]} {colors[i,1]} {colors[i,2]} {values[i]:.6f}\n")