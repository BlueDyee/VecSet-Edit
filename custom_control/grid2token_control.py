import os

from typing import Callable, List, Optional, Tuple, Union

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
from einops import repeat
from torch_cluster import fps
from tqdm import tqdm


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
        query_dim = query.shape[-1]  
        scale = 1.0 / (query_dim ** 0.5)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        #print("attention_scores shape:", attention_scores.shape)

        attention_probs = F.softmax(attention_scores, dim=-1)
        self.attn_probs=attention_probs
        self.attention  = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
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
        
        return hidden_states

class dec_cross_forward_override:
    def __init__(self):
        self.attn2_processor = CustomAttnProcessor2_0()
        self.vis_xyz_prob=[]
        self.vis_xyz_attn=[]
        self.vis_xyz_token_indices=[]
        self.call_count=0
        self.token_attention_distribution=[]
        # Should be initialized by vae control
        self.queries_mask=[]
        # self.rescale_xyz_metrics = torch.linspace(-1.0049, 1.0049, int(256), dtype=torch.float16)

    def register(self, block):
        # overwrite block.attn2.preocessor
        block.attn2.set_processor(self.attn2_processor)
        # 使用閉包捕獲 self（override 實例)
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
                    attn_output= block.attn2(
                        block.norm2(hidden_states),
                        encoder_hidden_states=encoder_hidden_states,
                        image_rotary_emb=image_rotary_emb,
                        **attention_kwargs,
                    ) * cross_attention_scale
                    hidden_states = hidden_states + attn_output

                    
                    attn_probs_val=self.attn2_processor.attn_probs[0].mean(dim=0)
                    # Feature select grid points to perform sum
                    sum_attn_distribution = attn_probs_val.sum(dim=0)
                    assert type(self.token_attention_distribution)==torch.Tensor, \
                        f"The list of token_attention_distribution should be initialize by vae control"
                    self.token_attention_distribution += sum_attn_distribution

            # FFN Layer ### TODO: switch norm2 and norm3 in the state dict
            mlp_inputs = block.norm3(hidden_states)
            hidden_states = hidden_states + block.ff(mlp_inputs)

            return hidden_states
            
        # 直接賦值，不需要 __get__
        block.forward = custom_forward

class DecoderOverride:
    def __init__(self, model, *args, **kwargs):
        self.model = model
        self.call_count = 0

    def custom_forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None, attention_kwargs=None) -> torch.Tensor:
        self.call_count += 1
        return self.model.decoder(x, skip=skip, attention_kwargs=attention_kwargs)
    
    def register(self):
        self.model.decoder.override.custom_forward = self.custom_forward

class VAEgrid2tokenControl:
    def __init__(self):
        self.call_count = 0
        self.token_strength = None
    def register(self,vae):
        self.decoder_override = dec_cross_forward_override()
        self.decoder_override.register(vae.decoder.blocks[-1])
        
        @torch.no_grad()
        def custom_vae_decode(
            z: torch.Tensor,
            sampled_points: torch.Tensor,
            num_chunks: int = 50000,
            to_cpu: bool = False,
            return_dict: bool = True,
        ) -> Union[DecoderOutput, torch.Tensor]:
            self.call_count+=1
            xyz_samples = sampled_points
            z = vae.post_quant(z)        

            num_points = xyz_samples.shape[1]
            kv_cache = None
            dec = []
            
            print(f"Starting to calculate token attention distribution for with latents = {z.shape}.")
            self.decoder_override.token_attention_distribution = torch.zeros((z.shape[1]), dtype=torch.float16, device=z.device)

            for i in range(0, num_points, num_chunks):
                queries = xyz_samples[:, i : i + num_chunks, :].to(z.device, dtype=z.dtype)
                queries = vae.embedder(queries)
                z_, kv_cache = vae.decoder(z, queries, kv_cache)
                dec.append(z_ if not to_cpu else z_.cpu())

            z = torch.cat(dec, dim=1)

            self.token_strength = self.decoder_override.token_attention_distribution
            
            if not return_dict:
                return (z,)

            return DecoderOutput(sample=z)
        vae._decode=custom_vae_decode
            



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
    
    # 將數值映射到顏色（使用matplotlib heatmap顏色映射）
    # 正規化到0-1範圍
    min_val, max_val = filtered_values.min(), filtered_values.max()
    normalized_values = (filtered_values - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(filtered_values)
    
    # 使用matplotlib viridis色彩映射 (深藍->綠->黃->紅)
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
    elif format_type.lower() == "xyz":
        output_file = f"{output_path}.xyz"
        save_xyz_pointcloud(filtered_points, colors, filtered_values, output_file)
    elif format_type.lower() == "pcd":
        output_file = f"{output_path}.pcd"
        save_pcd_pointcloud(filtered_points, colors, filtered_values, output_file)
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

def save_xyz_pointcloud(points, colors, values, output_path):
    """儲存為XYZ格式（簡單文字格式）"""
    with open(output_path, 'w') as f:
        for i in range(len(points)):
            f.write(f"{points[i,0]:.3f} {points[i,1]:.3f} {points[i,2]:.3f} "
                   f"{colors[i,0]} {colors[i,1]} {colors[i,2]} {values[i]:.6f}\n")

def save_pcd_pointcloud(points, colors, values, output_path):
    """儲存為PCD格式（PCL library格式）"""
    with open(output_path, 'w') as f:
        # PCD header
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z rgb value\n")
        f.write("SIZE 4 4 4 4 4\n")
        f.write("TYPE F F F U F\n")
        f.write("COUNT 1 1 1 1 1\n")
        f.write(f"WIDTH {len(points)}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {len(points)}\n")
        f.write("DATA ascii\n")
        
        # 點數據
        for i in range(len(points)):
            # 將RGB打包成單一整數
            rgb = (int(colors[i,0]) << 16) | (int(colors[i,1]) << 8) | int(colors[i,2])
            f.write(f"{points[i,0]:.3f} {points[i,1]:.3f} {points[i,2]:.3f} {rgb} {values[i]:.6f}\n")


def otsu_threshold_1d(data):
    """
    Otsu's method for 1D data to find optimal threshold
    
    Args:
        data: 1D numpy array of values
        
    Returns:
        optimal_threshold: the threshold value that maximizes inter-class variance
    """
    data = data.flatten()
    
    # Create histogram
    hist, bin_edges = np.histogram(data, bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Normalize histogram
    hist = hist.astype(np.float32)
    hist = hist / np.sum(hist)
    
    # Compute cumulative sums
    cumsum = np.cumsum(hist)
    cumsum_mean = np.cumsum(hist * bin_centers)
    
    # Avoid division by zero
    cumsum = np.maximum(cumsum, 1e-10)
    
    # Compute total mean
    global_mean = np.sum(hist * bin_centers)
    
    # Compute between-class variance for each possible threshold
    between_class_variance = np.zeros_like(cumsum)
    
    for i in range(len(cumsum)):
        if cumsum[i] > 0 and cumsum[i] < 1:
            # Mean of class 1 (below threshold)
            mean1 = cumsum_mean[i] / cumsum[i]
            
            # Mean of class 2 (above threshold)  
            w2 = 1 - cumsum[i]
            if w2 > 0:
                mean2 = (global_mean - cumsum_mean[i]) / w2
                
                # Between-class variance
                between_class_variance[i] = cumsum[i] * w2 * (mean1 - mean2) ** 2
    
    # Find threshold that maximizes between-class variance
    optimal_idx = np.argmax(between_class_variance)
    optimal_threshold = bin_centers[optimal_idx]
    
    return optimal_threshold


# Make Histogram of top5 prob
def plot_top5_prob_histogram(top5_probs, num_bins=50, output_path="./output/prob_histogram"):
    """
    繪製 top5 概率的直方圖
    
    Args:
        top5_probs: 1D numpy array of top5 probabilities
        num_bins: Number of bins for the histogram
    """
    import matplotlib.pyplot as plt
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    
    for k in range(5):
        plt.figure(figsize=(10, 6))
        plt.hist(top5_probs[:, k], bins=num_bins, color='blue', alpha=0.7)
        plt.title(f'Top {k+1} Probabilities Histogram')
        plt.xlabel('Probability')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(os.path.join(output_path, f'rank_{k+1}.png'))


def plot_token_top5_counts(top5_indices, output_path="./output/prob_histogram"):
    """
    繪製 top5 token 的計數直方圖

    Args:
        top5_indices: 2D numpy array of top5 token indices
    """
    import matplotlib.pyplot as plt
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    for k in range(5):
        plt.figure(figsize=(10, 6))
        plt.hist(top5_indices[:, k],bins=2048)

        plt.title(f'Top {k+1} Tokens counts')
        plt.xlabel('Token Index')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(os.path.join(output_path, f'token_count_rank={k+1}.png'))
