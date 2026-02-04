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

DEBUG_PRINT = True
# # 使用絕對導入，避免相對導入問題
# from triposg.models.attention_processor import FusedTripoSGAttnProcessor2_0, TripoSGAttnProcessor2_0, FlashTripoSGAttnProcessor2_0
# from triposg.models.embeddings import FrequencyPositionalEmbedding
# from triposg.models.transformers.triposg_transformer import DiTBlock
# from triposg.models.autoencoders.vae import DiagonalGaussianDistribution


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
        self.attn_probs = None
        self.attention = None

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
        # print("Query shape:", query.shape)
        # print("Key shape:", key.shape)
        query_dim = query.shape[-1]
        scale = 1.0 / (query_dim**0.5)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        # print("attention_scores shape:", attention_scores.shape)

        attention_probs = F.softmax(attention_scores, dim=-1)
        self.attn_probs = attention_probs
        self.attention = F.scaled_dot_product_attention(
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
    def __init__(self, vis_token_list, vis_attn_dim_list, cross_register_mode="whole"):
        self.store_information = {"score_map": None, "attn_map": None}
        self.attn2_processor = CustomAttnProcessor2_0()
        self.grid_matrix = None
        self.xyz_queries = None
        self.vis_token_list = vis_token_list  # 用於可視化的 token，默認為 0
        self.vis_attn_dim_list = vis_attn_dim_list  # 用於可視化的 attention 維度
        self.vis_xyz_prob = []
        self.vis_xyz_attn = []
        self.vis_xyz_token_indices = []
        self.call_count = 0
        self.cross_register_mode = cross_register_mode
        # self.rescale_xyz_metrics = torch.linspace(-1.0049, 1.0049, int(256), dtype=torch.float16)

        # Store original methods for unregistering
        self.original_forward = None
        self.original_attn2_processor = None
        self.registered_block = None

    def register(self, block):
        # Store references for unregistering
        self.registered_block = block
        self.original_forward = block.forward
        self.original_attn2_processor = block.attn2.processor

        # overwrite block.attn2.preocessor
        block.attn2.set_processor(self.attn2_processor)

        # 使用閉包捕獲 self（override 實例)
        @torch.no_grad()
        def custom_forward(
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_hidden_states_2: Optional[torch.Tensor] = None,
            temb: Optional[torch.Tensor] = None,
            image_rotary_emb: Optional[torch.Tensor] = None,
            skip: Optional[torch.Tensor] = None,
            attention_kwargs=None,
        ) -> torch.Tensor:
            self.call_count += 1

            # Prepare attention kwargs
            attention_kwargs = (
                attention_kwargs.copy() if attention_kwargs is not None else {}
            )
            cross_attention_scale = attention_kwargs.pop("cross_attention_scale", 1.0)
            cross_attention_2_scale = attention_kwargs.pop(
                "cross_attention_2_scale", 1.0
            )
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
                        )
                        * cross_attention_scale
                        + block.attn2_2(
                            block.norm2_2(hidden_states),
                            encoder_hidden_states=encoder_hidden_states_2,
                            image_rotary_emb=image_rotary_emb,
                            **attention_kwargs,
                        )
                        * cross_attention_2_scale
                    )
                else:
                    # This attn2 is overide by custom_attention processor
                    attn_output = (
                        block.attn2(
                            block.norm2(hidden_states),
                            encoder_hidden_states=encoder_hidden_states,
                            image_rotary_emb=image_rotary_emb,
                            **attention_kwargs,
                        )
                        * cross_attention_scale
                    )
                    hidden_states = hidden_states + attn_output

                    attn_probs_val = self.attn2_processor.attn_probs[0].mean(dim=0)
                    attention_val = self.attn2_processor.attention[0].mean(dim=0)

                    if self.cross_register_mode == "whole":
                        self.vis_xyz_prob.append(
                            attn_probs_val[:, self.vis_token_list].cpu().mean(dim=-1)
                        )
                        self.vis_xyz_attn.append(
                            attention_val[:, self.vis_attn_dim_list].cpu().mean(dim=-1)
                        )
                    elif self.cross_register_mode == "top5":
                        # 只保留 top5 的 token
                        attn_probs_val = attn_probs_val[:, self.vis_token_list]
                        top5_values, top5_indices = torch.topk(
                            attn_probs_val, k=5, dim=-1
                        )
                        # print(f"top5_indices.shape={top5_indices.shape} top5_values.shape={top5_values.shape}")
                        # print(f"attnprob.shape={attn_probs_val.shape}")
                        # 直接使用topk返回的values
                        self.vis_xyz_token_indices.append(top5_indices.cpu())
                        self.vis_xyz_prob.append(top5_values.cpu())

                        # print(f"self.vis_xyz_prob[-1].shape={attn_probs_val[top5_indices].shape}")

                        self.vis_xyz_attn.append(
                            attention_val[:, self.vis_attn_dim_list].cpu()
                        )
                        # print(f"self.vis_xyz_attn[-1].shape={self.vis_xyz_attn[-1].shape}")
                        # exit()

            # FFN Layer ### TODO: switch norm2 and norm3 in the state dict
            mlp_inputs = block.norm3(hidden_states)
            hidden_states = hidden_states + block.ff(mlp_inputs)
            # attention dim=128, out dim=1024
            return hidden_states

        # 直接賦值，不需要 __get__
        block.forward = custom_forward

    def unregister(self):
        """
        Unregister the decoder cross forward override and restore original methods
        """
        if self.registered_block is None:
            print(
                "⚠️ dec_cross_forward_override was not properly registered. Nothing to unregister."
            )
            return

        try:
            # Restore original forward method
            if self.original_forward is not None:
                self.registered_block.forward = self.original_forward

            # Restore original attention processor
            if self.original_attn2_processor is not None:
                self.registered_block.attn2.set_processor(self.original_attn2_processor)

            # Clear stored references
            self.registered_block = None
            self.original_forward = None
            self.original_attn2_processor = None

            # Clear visualization data
            self.vis_xyz_prob = []
            self.vis_xyz_attn = []
            self.vis_xyz_token_indices = []
            self.call_count = 0

            # print("✅ dec_cross_forward_override successfully unregistered.")

        except Exception as e:
            print(f"❌ Error during unregistering dec_cross_forward_override: {e}")
            raise


class DecoderOverride:
    def __init__(self, model, *args, **kwargs):
        self.model = model
        self.call_count = 0

    def custom_forward(
        self,
        x: torch.Tensor,
        skip: Optional[torch.Tensor] = None,
        attention_kwargs=None,
    ) -> torch.Tensor:
        self.call_count += 1
        return self.model.decoder(x, skip=skip, attention_kwargs=attention_kwargs)

    def register(self):
        self.model.decoder.override.custom_forward = self.custom_forward


class VAEvisualizationControl:
    def __init__(
        self,
        vis_token_list=[0, 1],
        vis_attn_dim_list=[0, 1],
        save_dir="./output/attn_cloud",
        cross_register_mode="whole",
        save_top5_path=None,
        save_attn_cloud=True,
        attn_cloud_name="attn_cloud",
    ):
        self.vis_token_list = vis_token_list  # 用於可視化的 token
        self.vis_attn_dim_list = vis_attn_dim_list  # 用於可視化的 attention 維度
        self.call_count = 0
        self.save_dir = save_dir
        self.cross_register_mode = cross_register_mode
        self.save_attn_cloud = save_attn_cloud
        self.save_top5_path = save_top5_path
        self.attn_cloud_name = attn_cloud_name

        # Store original methods for unregistering
        self.original_vae_decode = None
        self.original_block_forward = None
        self.original_attn2_processor = None
        self.registered_vae = None
        self.registered_block = None

    def register(self, vae):
        # Store references for unregistering
        self.registered_vae = vae
        self.registered_block = vae.decoder.blocks[-1]
        self.original_vae_decode = vae._decode
        self.original_block_forward = self.registered_block.forward
        self.original_attn2_processor = self.registered_block.attn2.processor

        # Also overide the cross attn block of vae.decoder
        self.decoder_override = dec_cross_forward_override(
            self.vis_token_list, self.vis_attn_dim_list, self.cross_register_mode
        )
        self.decoder_override.register(vae.decoder.blocks[-1])

        def custom_vae_decode(
            z: torch.Tensor,
            sampled_points: torch.Tensor,
            num_chunks: int = 50000,
            to_cpu: bool = False,
            return_dict: bool = True,
        ) -> Union[DecoderOutput, torch.Tensor]:
            self.call_count += 1
            xyz_samples = sampled_points
            grid_dim = round(xyz_samples.shape[1] ** (1 / 3))
            z = vae.post_quant(z)
            # print("z.shape after post_quant:", z.shape)

            num_points = xyz_samples.shape[1]
            kv_cache = None
            dec = []
            print(
                f"🥷 Starting to visualize xyz attention map with {len(self.vis_token_list)} tokens"
            )

            for i in range(0, num_points, num_chunks):
                queries = xyz_samples[:, i : i + num_chunks, :].to(
                    z.device, dtype=z.dtype
                )
                queries = vae.embedder(queries)
                z_, kv_cache = vae.decoder(z, queries, kv_cache)
                dec.append(z_ if not to_cpu else z_.cpu())
            z = torch.cat(dec, dim=1)

            # print(f"concatenating  overide attn score")
            out_prob_3d = torch.cat(
                [d for d in self.decoder_override.vis_xyz_prob], dim=0
            )
            out_attn_3d = torch.cat(
                [d for d in self.decoder_override.vis_xyz_attn], dim=0
            )
            # Apply Otsu's thresholding to find optimal cutting point
            if self.save_attn_cloud:
                # 直接使用原始形狀，不強制reshape為grid
                out_prob_3d = out_prob_3d.to(torch.float16)  # shape: [N]
                out_attn_3d = out_attn_3d.to(torch.float16)  # shape: [N]
                xyz_samples = xyz_samples[0].to(torch.float16)  # shape: [N, 3]

                # 確保數據長度一致
                min_length = min(len(out_prob_3d), len(out_attn_3d), len(xyz_samples))
                out_prob_3d = out_prob_3d[:min_length]
                out_attn_3d = out_attn_3d[:min_length]
                xyz_samples = xyz_samples[:min_length]

                otsu_threshold, retention_rate, iterations = otsu_threshold_1d_adaptive(
                    out_prob_3d.cpu().numpy(), 
                    max_retention_rate=0.5,  # 最多保留50%
                    max_iterations=10
                )
                print(f"🎯 Adaptive threshold: {otsu_threshold:.6f}, retention: {retention_rate:.2%}, iterations: {iterations}")
                os.makedirs(self.save_dir, exist_ok=True)
                # print(f"🔍 Otsu optimal threshold for prob: {otsu_threshold:.6f}")
                save_grid_logits_as_pointcloud_flexible(
                    xyz_samples=xyz_samples,
                    grid_logits=out_prob_3d,
                    output_path=f"{self.save_dir}/{self.attn_cloud_name}_{self.call_count}",
                    format_type="ply",
                    threshold=otsu_threshold,
                    downsample_factor=2,  # 下採樣減少點數量
                    colormap="coolwarm",
                )
                # otsu_threshold = otsu_threshold_1d(out_attn_3d.cpu().numpy())
                # print(f"🔍 Otsu optimal threshold for attn: {otsu_threshold:.6f}")
                # save_grid_logits_as_pointcloud(
                #     xyz_samples=xyz_samples,
                #     grid_logits=out_attn_3d,
                #     output_path=f"{self.save_dir}/attn_{self.call_count}",
                #     format_type="ply",
                #     threshold=otsu_threshold,  # Use Otsu threshold instead of fixed value
                #     downsample_factor=2,  # 下採樣減少點數量
                #     colormap="coolwarm"
                # )

            self.decoder_override.vis_xyz_prob = []
            self.decoder_override.vis_xyz_attn = []
            self.decoder_override.vis_xyz_token_indices = []

            if not return_dict:
                return (z,)

            return DecoderOutput(sample=z)

        vae._decode = custom_vae_decode

    def unregister(self):
        """
        Unregister the visualization control and restore original methods
        """
        if self.registered_vae is None or self.registered_block is None:
            print(
                "⚠️ VAEvisualizationControl was not properly registered. Nothing to unregister."
            )
            return

        try:
            # Unregister decoder override first
            if hasattr(self, "decoder_override") and self.decoder_override is not None:
                self.decoder_override.unregister()

            # Restore original VAE decode method
            if self.original_vae_decode is not None:
                self.registered_vae._decode = self.original_vae_decode

            # Clear stored references
            self.registered_vae = None
            self.registered_block = None
            self.original_vae_decode = None
            self.original_block_forward = None
            self.original_attn2_processor = None
            self.decoder_override = None

            # print("✅ VAEvisualizationControl successfully unregistered.")

        except Exception as e:
            print(f"❌ Error during unregistering: {e}")
            raise


def save_grid_logits_as_pointcloud_flexible(
    xyz_samples: torch.Tensor,
    grid_logits: torch.Tensor,
    output_path: str = "grid_logits_pointcloud",
    format_type: str = "ply",
    threshold: float = -5.0,
    downsample_factor: int = 1,
    colormap: str = "coolwarm",
):
    """
    將1D點雲數據轉換為點雲格式並儲存，不假設grid結構

    Args:
        xyz_samples: shape (N, 3) 的tensor，包含每個點的實際xyz座標
        grid_logits: shape (N,) 的tensor，包含每個點的logit值
        output_path: 輸出檔案路徑（不含副檔名）
        format_type: 格式類型 ("ply", "xyz", "pcd")
        threshold: 只保存大於此閾值的點（過濾背景）
        downsample_factor: 下採樣因子，減少點的數量
        colormap: 色彩映射類型 ("viridis", "plasma", "inferno", "magma", "hot", "coolwarm", "RdYlBu_r")
    """

    if isinstance(grid_logits, torch.Tensor):
        grid_logits = grid_logits.cpu().numpy()

    if isinstance(xyz_samples, torch.Tensor):
        xyz_samples = xyz_samples.cpu().numpy()

    # 確保輸入是2D (N, 3) 和1D (N,)
    if xyz_samples.ndim != 2 or xyz_samples.shape[1] != 3:
        raise ValueError(f"xyz_samples should be shape (N, 3), got {xyz_samples.shape}")
    if grid_logits.ndim != 1:
        raise ValueError(f"grid_logits should be shape (N,), got {grid_logits.shape}")
    if xyz_samples.shape[0] != grid_logits.shape[0]:
        raise ValueError(
            f"Number of points mismatch: xyz_samples {xyz_samples.shape[0]} vs grid_logits {grid_logits.shape[0]}"
        )

    N = xyz_samples.shape[0]
    # print(f"Total points: {N}")
    # print(f"Value range: {grid_logits.min():.3f} to {grid_logits.max():.3f}")

    # 下採樣處理
    if downsample_factor > 1:
        indices = np.arange(0, N, downsample_factor)
        downsampled_xyz = xyz_samples[indices]
        downsampled_logits = grid_logits[indices]
    else:
        downsampled_xyz = xyz_samples
        downsampled_logits = grid_logits

    # 根據閾值過濾點
    mask = downsampled_logits > threshold
    filtered_points = downsampled_xyz[mask]
    filtered_values = downsampled_logits[mask]

    # print(f"Filtered points: {len(filtered_points)} / {len(downsampled_xyz)}")
    # if len(filtered_points) > 0:
    #     # print(
    #     #     f"Point coordinate range: X[{filtered_points[:, 0].min():.3f}, {filtered_points[:, 0].max():.3f}], "
    #     #     f"Y[{filtered_points[:, 1].min():.3f}, {filtered_points[:, 1].max():.3f}], "
    #     #     f"Z[{filtered_points[:, 2].min():.3f}, {filtered_points[:, 2].max():.3f}]"
    #     # )

    if len(filtered_points) == 0:
        print(
            "Warning: No points remain after filtering. Lowering threshold or check data."
        )
        return output_path + f".{format_type}"

    # 將數值映射到顏色
    min_val, max_val = filtered_values.min(), filtered_values.max()
    normalized_values = (
        (filtered_values - min_val) / (max_val - min_val)
        if max_val > min_val
        else np.zeros_like(filtered_values)
    )

    # 使用matplotlib色彩映射
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    colormap_obj = cm.get_cmap(colormap)
    colors_rgba = colormap_obj(normalized_values)
    colors = (colors_rgba[:, :3] * 255).astype(np.uint8)  # 轉換為0-255範圍的RGB

    if format_type.lower() == "ply":
        output_file = f"{output_path}.ply"
        save_ply_pointcloud(filtered_points, colors, normalized_values, output_file)
    elif format_type.lower() == "xyz":
        output_file = f"{output_path}.xyz"
        save_xyz_pointcloud(filtered_points, colors, normalized_values, output_file)
    elif format_type.lower() == "pcd":
        output_file = f"{output_path}.pcd"
        save_pcd_pointcloud(filtered_points, colors, normalized_values, output_file)
    else:
        raise ValueError(f"Unsupported format: {format_type}")

    print(f"Point cloud saved to: {output_file}")
    return output_file


def save_grid_logits_as_pointcloud(
    xyz_samples: torch.Tensor,
    grid_logits: torch.Tensor,
    output_path: str = "grid_logits_pointcloud",
    format_type: str = "ply",
    threshold: float = -5.0,
    downsample_factor: int = 1,
    colormap: str = "coolwarm",
):
    """
    將3D grid logits轉換為點雲格式並儲存，使用提供的xyz座標

    適合的3D點雲格式：
    1. PLY (Polygon File Format) - 最常用，支援顏色
    2. PCD (Point Cloud Data) - PCL library格式
    3. XYZ - 簡單文字格式
    4. OFF - Object File Format

    Args:
        xyz_samples: shape (H, W, D, 3) 的3D tensor，包含每個點的實際xyz座標
        grid_logits: shape (H, W, D) 的3D tensor，包含每個點的logit值
        output_path: 輸出檔案路徑（不含副檔名）
        format_type: 格式類型 ("ply", "xyz", "pcd")
        threshold: 只保存大於此閾值的點（過濾背景）
        downsample_factor: 下採樣因子，減少點的數量
        colormap: 色彩映射類型 ("viridis", "plasma", "inferno", "magma", "hot", "coolwarm", "RdYlBu_r")
    """

    if isinstance(grid_logits, torch.Tensor):
        grid_logits = grid_logits.cpu().numpy()

    if isinstance(xyz_samples, torch.Tensor):
        xyz_samples = xyz_samples.cpu().numpy()

    H, W, D = grid_logits.shape
    print(f"Original grid shape: {grid_logits.shape}")
    print(f"XYZ samples shape: {xyz_samples.shape}")
    print(f"Value range: {grid_logits.min():.3f} to {grid_logits.max():.3f}")

    # 確保xyz_samples和grid_logits的維度匹配
    if xyz_samples.shape[:3] != grid_logits.shape:
        raise ValueError(
            f"xyz_samples shape {xyz_samples.shape[:3]} doesn't match grid_logits shape {grid_logits.shape}"
        )

    # 下採樣處理
    downsampled_logits = grid_logits[
        ::downsample_factor, ::downsample_factor, ::downsample_factor
    ]
    downsampled_xyz = xyz_samples[
        ::downsample_factor, ::downsample_factor, ::downsample_factor
    ]

    # 扁平化所有數據
    points = downsampled_xyz.reshape(-1, 3)  # 使用實際的xyz座標
    values = downsampled_logits.flatten()

    # 根據閾值過濾點
    mask = values > threshold
    filtered_points = points[mask]
    filtered_values = values[mask]

    print(f"Filtered points: {len(filtered_points)} / {len(points)}")
    print(
        f"Point coordinate range: X[{filtered_points[:, 0].min():.3f}, {filtered_points[:, 0].max():.3f}], "
        f"Y[{filtered_points[:, 1].min():.3f}, {filtered_points[:, 1].max():.3f}], "
        f"Z[{filtered_points[:, 2].min():.3f}, {filtered_points[:, 2].max():.3f}]"
    )

    # 將數值映射到顏色（使用matplotlib heatmap顏色映射）
    # 正規化到0-1範圍
    min_val, max_val = filtered_values.min(), filtered_values.max()
    normalized_values = (
        (filtered_values - min_val) / (max_val - min_val)
        if max_val > min_val
        else np.zeros_like(filtered_values)
    )

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
        save_ply_pointcloud(filtered_points, colors, normalized_values, output_file)
    elif format_type.lower() == "xyz":
        output_file = f"{output_path}.xyz"
        save_xyz_pointcloud(filtered_points, colors, normalized_values, output_file)
    elif format_type.lower() == "pcd":
        output_file = f"{output_path}.pcd"
        save_pcd_pointcloud(filtered_points, colors, normalized_values, output_file)
    else:
        raise ValueError(f"Unsupported format: {format_type}")

    print(f"Point cloud saved to: {output_file}")
    return output_file


def save_ply_pointcloud(points, colors, values, output_path):
    """儲存為PLY格式（最推薦，支援多數3D軟體）"""
    with open(output_path, "w") as f:
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
        if DEBUG_PRINT:
            print("points max:", points.max(), "min:", points.min())
        for i in range(len(points)):
            f.write(
                f"{points[i,0]:.3f} {points[i,1]:.3f} {points[i,2]:.3f} "
                f"{colors[i,0]} {colors[i,1]} {colors[i,2]} {values[i]:.6f}\n"
            )
    npy_output_path = output_path.replace(".ply", "_values.npy")
    np.save(npy_output_path, values)


def save_xyz_pointcloud(points, colors, values, output_path):
    """儲存為XYZ格式（簡單文字格式）"""
    with open(output_path, "w") as f:
        for i in range(len(points)):
            f.write(
                f"{points[i,0]:.3f} {points[i,1]:.3f} {points[i,2]:.3f} "
                f"{colors[i,0]} {colors[i,1]} {colors[i,2]} {values[i]:.6f}\n"
            )


def save_pcd_pointcloud(points, colors, values, output_path):
    """儲存為PCD格式（PCL library格式）"""
    with open(output_path, "w") as f:
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
            rgb = (
                (int(colors[i, 0]) << 16) | (int(colors[i, 1]) << 8) | int(colors[i, 2])
            )
            f.write(
                f"{points[i,0]:.3f} {points[i,1]:.3f} {points[i,2]:.3f} {rgb} {values[i]:.6f}\n"
            )


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

def otsu_threshold_1d_adaptive(data, max_retention_rate=0.5, max_iterations=10):
    """
    自適應 Otsu threshold，確保保留點數不超過指定比例
    
    Args:
        data: 1D numpy array of values
        max_retention_rate: 最大保留率 (0-1)，預設 0.5 (50%)
        max_iterations: 最大迭代次數
        
    Returns:
        optimal_threshold: 調整後的閾值
        retention_rate: 實際保留率
        iterations: 迭代次數
    """
    data = data.flatten()
    total_points = len(data)
    
    # 先計算原始 Otsu threshold
    original_threshold = otsu_threshold_1d(data)
    current_threshold = original_threshold
    
    for iteration in range(max_iterations):
        # 計算當前閾值的保留率
        retained_points = np.sum(data > current_threshold)
        retention_rate = retained_points / total_points
        
        # 如果保留率已經符合要求，跳出迭代
        if retention_rate <= max_retention_rate:
            break
            
        # 如果保留率太高，提高閾值
        # 使用數據標準差來調整步長
        data_std = np.std(data)
        adjustment = data_std * 0.7 * (retention_rate - max_retention_rate)
        current_threshold += adjustment
        
        # 防止閾值超出數據範圍
        current_threshold = min(current_threshold, np.max(data))
    
    final_retained = np.sum(data > current_threshold)
    final_retention_rate = final_retained / total_points
    
    return current_threshold, final_retention_rate, iteration + 1


# Make Histogram of top5 prob
def plot_top5_prob_histogram(
    top5_probs, num_bins=50, output_path="./output/prob_histogram"
):
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
        plt.hist(top5_probs[:, k], bins=num_bins, color="blue", alpha=0.7)
        plt.title(f"Top {k+1} Probabilities Histogram")
        plt.xlabel("Probability")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.savefig(os.path.join(output_path, f"rank_{k+1}.png"))


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
        plt.hist(top5_indices[:, k], bins=2048)

        plt.title(f"Top {k+1} Tokens counts")
        plt.xlabel("Token Index")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.savefig(os.path.join(output_path, f"token_count_rank={k+1}.png"))
