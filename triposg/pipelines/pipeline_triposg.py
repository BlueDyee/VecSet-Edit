import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import os

import numpy as np
import PIL
import PIL.Image
import torch
import trimesh
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor
from transformers import (
    BitImageProcessor,
    Dinov2Model,
)

from ..inference_utils import hierarchical_extract_geometry, flash_extract_geometry

from ..models.autoencoders import TripoSGVAEModel
from ..models.transformers import TripoSGDiTModel
from .pipeline_triposg_output import TripoSGPipelineOutput
from .pipeline_utils import TransformerDiffusionMixin

from custom_control.dit_control import TripoSGDiTControl
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def save_grid_logits_as_pointcloud(
    xyz_samples: torch.Tensor,
    grid_logits: torch.Tensor,
    output_path: str = "grid_logits_pointcloud.ply",
    format_type: str = "ply",
    threshold: float = -5.0,
    downsample_factor: int = 1,
    colormap: str = "coolwarm",
):
    """
    將3D logits列表轉換為點雲格式並儲存，使用提供的xyz座標

    適合的3D點雲格式：
    1. PLY (Polygon File Format) - 最常用，支援顏色
    2. PCD (Point Cloud Data) - PCL library格式
    3. XYZ - 簡單文字格式
    4. OFF - Object File Format

    Args:
        xyz_samples: shape (N, 3) 的tensor，包含每個點的實際xyz座標
        grid_logits: shape (N,) 或 (1, N) 的tensor，包含每個點的logit值
        output_path: 輸出檔案路徑（不含副檔名）
        format_type: 格式類型 ("ply", "xyz", "pcd")
        threshold: 只保存小於此閾值的點（過濾背景，SDF通常負值表示內部）
        downsample_factor: 採樣因子，用於減少點的數量（每downsample_factor個點取1個）
        colormap: 色彩映射類型 ("viridis", "plasma", "inferno", "magma", "hot", "coolwarm", "RdYlBu_r")
    """

    if isinstance(grid_logits, torch.Tensor):
        grid_logits = grid_logits.cpu().numpy()

    if isinstance(xyz_samples, torch.Tensor):
        xyz_samples = xyz_samples.cpu().numpy()

    # 處理維度：確保grid_logits是1D數組
    if grid_logits.ndim > 1:
        grid_logits = grid_logits.squeeze()

    # 確保xyz_samples是2D數組 (N, 3)
    if xyz_samples.ndim == 3:
        xyz_samples = xyz_samples.squeeze(0)

    print(f"XYZ samples shape: {xyz_samples.shape}")
    print(f"Grid logits shape: {grid_logits.shape}")
    print(f"Value range: {grid_logits.min():.3f} to {grid_logits.max():.3f}")

    # 確保數據維度匹配
    if len(xyz_samples) != len(grid_logits):
        raise ValueError(
            f"xyz_samples length {len(xyz_samples)} doesn't match grid_logits length {len(grid_logits)}"
        )

    # 使用所有點（因為輸入已經是列表格式，不是網格）
    points = xyz_samples
    values = grid_logits

    # 可選的下採樣：每downsample_factor個點取1個
    if downsample_factor > 1:
        indices = np.arange(0, len(points), downsample_factor)
        points = points[indices]
        values = values[indices]
        print(f"Downsampled to {len(points)} points (factor: {downsample_factor})")

    # 根據閾值過濾點（對於SDF，通常threshold=0，負值表示物體內部）
    mask = values < threshold
    filtered_points = points[mask]
    filtered_values = values[mask]

    print(f"Filtered points: {len(filtered_points)} / {len(points)}")
    print(
        f"Point coordinate range: X[{filtered_points[:, 0].min():.3f}, {filtered_points[:, 0].max():.3f}], "
        f"Y[{filtered_points[:, 1].min():.3f}, {filtered_points[:, 1].max():.3f}], "
        f"Z[{filtered_points[:, 2].min():.3f}, {filtered_points[:, 2].max():.3f}]"
    )

    if len(filtered_points) == 0:
        print("Warning: No points remain after filtering!")
        return None

    # 將數值映射到顏色（使用matplotlib heatmap顏色映射）
    # 正規化到0-1範圍
    min_val, max_val = filtered_values.min(), filtered_values.max()
    normalized_values = (
        (filtered_values - min_val) / (max_val - min_val)
        if max_val > min_val
        else np.zeros_like(filtered_values)
    )

    # 使用matplotlib色彩映射
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    # 選擇色彩映射 - 提供多種heatmap顏色選擇
    colormap_obj = cm.get_cmap(colormap)

    # 將正規化值映射到RGB顏色
    colors_rgba = colormap_obj(normalized_values)
    colors = (colors_rgba[:, :3] * 255).astype(np.uint8)  # 轉換為0-255範圍的RGB

    if format_type.lower() == "ply":
        output_file = f"{output_path}"
        save_ply_pointcloud(filtered_points, colors, filtered_values, output_file)
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
        for i in range(len(points)):
            f.write(
                f"{points[i,0]:.3f} {points[i,1]:.3f} {points[i,2]:.3f} "
                f"{colors[i,0]} {colors[i,1]} {colors[i,2]} {values[i]:.6f}\n"
            )


def retrieve_latents(
    encoder_output: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    sample_mode: str = "sample",
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

def attention2d(dit_control, step, token_mask, output_dir=None,
                 k_attentive=5, top_k_percent=0.2, threshold_percent=0.2, block_num=20, **kwargs):
    kl_strength = []
    for block_idx in range(block_num):
        control=dit_control.idx2control[block_idx]
        attn_prob=control.prob_list[step][1:,1:] # remove temb token, cls token
        N3d,N2d= attn_prob.shape
        selected_attn_prob = attn_prob[:, token_mask]
        kl_attn_dist = attn_prob.mean(dim=0)
        kl_attn_dist_ = kl_attn_dist.expand_as(attn_prob) 
        eps = 1e-9
        kl_per_sample = (attn_prob * (torch.log(attn_prob + eps) - torch.log(kl_attn_dist_ + eps))).sum(dim=1)
        # inv_kl_per_sample = (kl_attn_dist * (torch.log(kl_attn_dist + eps) - torch.log(attn_prob + eps))).sum(dim=1)
        kl_strength.append(kl_per_sample.mean().item())
    attentive_idx_list = torch.argsort(torch.tensor(kl_strength),descending=True)[:k_attentive] 
    attentive_idx_list = attentive_idx_list.tolist()

    token_selected_freq = torch.zeros((N3d, ), dtype=torch.float32, device=attn_prob.device)
    for block_idx in attentive_idx_list:
        if output_dir is not None:
            save_dir = os.path.join(output_dir, f"step_{step}", f"block_{block_idx}")
            os.makedirs(save_dir, exist_ok=True)
        control=dit_control.idx2control[block_idx]
        attn_prob=control.prob_list[step][1:,1:] # remove temb token, cls token
        selected_attn_prob = attn_prob[:, token_mask]

        sel_kl_dist = selected_attn_prob.mean(dim=0)
        sel_kl_dist_ = sel_kl_dist.expand_as(selected_attn_prob)
        sel_kl_sample = (selected_attn_prob * (torch.log(selected_attn_prob + eps) - torch.log(sel_kl_dist_ + eps))).sum(dim=1)
        token_selected_freq += sel_kl_sample

    # Fixed TopK Threshold
    sort_val, sort_idx = torch.topk(torch.tensor(token_selected_freq), k=token_selected_freq.shape[0])
    n_highest = max(1, int(len(sort_val) * top_k_percent))
    threshold = sort_val[:n_highest].mean()*threshold_percent
    mask = sort_val > threshold
    filtered_idx = sort_idx[mask]
    return filtered_idx, token_selected_freq, kl_strength


def attention_self(dit_control, step, tokens_len, selected_tokens, topk_each_block=None, cut_off_std=3, cut_off_p=0.1, output_dir=None, **kwargs):
    # 初始化頻率計數器
    tokens_freq = torch.zeros(tokens_len).to("cuda")  

    for block_idx in range(20):
        if output_dir is not None:
            save_dir = os.path.join(output_dir, f"block_{block_idx}")
            os.makedirs(save_dir, exist_ok=True)

        control = dit_control.idx2control[block_idx]
        attn_prob = control.prob_list_self[step][1:, 1:]  # remove temb token, cls token
        N3d, M3d = attn_prob.shape
        
        # 計算選中token的強度
        k_selected_strength = attn_prob[:, selected_tokens].mean(dim=-1)
        if topk_each_block is not None:
            topk_values, topk_indices = torch.topk(k_selected_strength, k=topk_each_block, dim=0)
            tokens_freq[topk_indices] += 1
        else:
            tokens_freq += k_selected_strength
    # mean_freq = tokens_freq[selected_tokens].mean()
    # std_freq = tokens_freq[selected_tokens].std()
    # min_str = tokens_freq[selected_tokens].min()
    # threshold = torch.max(mean_freq - cut_off_std*std_freq, min_str)
    # final_selected_tokens = torch.nonzero(tokens_freq >= threshold).squeeze(1)

    selected_freq_values = tokens_freq[selected_tokens]
    n_lowest = max(1, int(len(selected_freq_values) * cut_off_p))
    lowest_values, _ = torch.topk(selected_freq_values, k=n_lowest, largest=False)
    threshold = lowest_values.mean()
    final_selected_tokens = torch.nonzero(tokens_freq >= threshold).squeeze(1)

    return final_selected_tokens, tokens_freq

class TripoSGPipeline(DiffusionPipeline, TransformerDiffusionMixin):
    """
    Pipeline for image-to-3D generation.
    """

    def __init__(
        self,
        vae: TripoSGVAEModel,
        transformer: TripoSGDiTModel,
        scheduler: FlowMatchEulerDiscreteScheduler,  # Actually <class 'triposg.schedulers.scheduling_rectified_flow.RectifiedFlowScheduler'>
        image_encoder_dinov2: Dinov2Model,
        feature_extractor_dinov2: BitImageProcessor,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            image_encoder_dinov2=image_encoder_dinov2,
            feature_extractor_dinov2=feature_extractor_dinov2,
        )

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def decode_progressive(self):
        return self._decode_progressive

    def encode_image(self, image, device, num_images_per_prompt):
        dtype = next(self.image_encoder_dinov2.parameters()).dtype

        # Image Processing
        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor_dinov2(
                image, return_tensors="pt"
            ).pixel_values

        image = image.to(device=device, dtype=dtype)
        # Feature Extraction
        image_embeds = self.image_encoder_dinov2(image).last_hidden_state
        image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        uncond_image_embeds = torch.zeros_like(image_embeds)

        return image_embeds, uncond_image_embeds

    def prepare_latents(
        self,
        batch_size,
        num_tokens,
        num_channels_latents,
        dtype,
        device,
        generator,
        latents: Optional[torch.Tensor] = None,
    ):
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (batch_size, num_tokens, num_channels_latents)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        return latents

    @torch.no_grad()
    def __call__(
        self,
        image: PipelineImageInput,
        num_inference_steps: int = 50,
        num_tokens: int = 2048,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        num_shapes_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        bounds: Union[Tuple[float], List[float], float] = (
            -1.005,
            -1.005,
            -1.005,
            1.005,
            1.005,
            1.005,
        ),
        dense_octree_depth: int = 8,
        hierarchical_octree_depth: int = 8,
        flash_octree_depth: int = 8,
        use_flash_decoder: bool = False,
        return_dict: bool = True,
        debug_dict={},
    ):
        # 1. Define call parameters
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs

        # 2. Define call parameters
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        elif isinstance(image, torch.Tensor):
            batch_size = image.shape[0]
        else:
            raise ValueError("Invalid input type for image")

        device = self._execution_device

        # 3. Encode condition
        print("TripoSGPipeline: Encoding image...")
        #
        image_embeds, negative_image_embeds = self.encode_image(
            image, device, num_shapes_per_prompt
        )
        print("image_embeds.shape:", image_embeds.shape)

        if self.do_classifier_free_guidance:
            image_embeds = torch.cat([negative_image_embeds, image_embeds], dim=0)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_shapes_per_prompt,
            num_tokens,
            num_channels_latents,
            image_embeds.dtype,
            device,
            generator,
            latents,
        )
        output_debug_dict = {"vis_intermediate": []}
        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if "vis_intermediate" in debug_dict:
                    step = debug_dict["vis_intermediate"].get("interval", 5)
                    start = debug_dict["vis_intermediate"].get("start", 0)
                    if i >= start and (i - start) % step == 0:
                        geometric_func = lambda x: self.vae.decode(
                            latents, sampled_points=x
                        ).sample
                        save_sdf = False
                        if "sdf_cloud" in debug_dict["vis_intermediate"]:
                            save_sdf = True
                        output = hierarchical_extract_geometry(
                            geometric_func,
                            device,
                            bounds=bounds,
                            dense_octree_depth=dense_octree_depth,
                            hierarchical_octree_depth=hierarchical_octree_depth,
                            debug_print=save_sdf,
                            debug_print_dict={
                                "sdf_cloud": "./output/sdf_cloud/step_{}_t_{}".format(
                                    i, t.item()
                                ),
                            },
                        )
                        meshes = [
                            trimesh.Trimesh(mesh_v_f[0].astype(np.float32), mesh_v_f[1])
                            for mesh_v_f in output
                        ]
                        out_mesh_path = "./output/inter_mesh/step_{}_t_{}.glb".format(
                            i, t.item()
                        )
                        meshes[0].export(out_mesh_path) if save_sdf else None
                        (
                            print(
                                "DEBUG: Meshes exported for step {} at t={} at {}".format(
                                    i, t.item(), out_mesh_path
                                )
                            )
                            if save_sdf
                            else None
                        )
                        # meshes = [trimesh.Trimesh(mesh_v_f[0].astype(np.float32), mesh_v_f[1]) for mesh_v_f in output]
                        output_debug_dict["vis_intermediate"].append(
                            {
                                "step": i,
                                "timestep": t.item(),
                                "output_tensor": output,
                            }
                        )
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=image_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_image = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_image - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        # 7. decoder mesh
        if not use_flash_decoder:
            geometric_func = lambda x: self.vae.decode(latents, sampled_points=x).sample
            output = hierarchical_extract_geometry(
                geometric_func,
                device,
                bounds=bounds,
                dense_octree_depth=dense_octree_depth,
                hierarchical_octree_depth=hierarchical_octree_depth,
            )
        else:
            self.vae.set_flash_decoder()
            output = flash_extract_geometry(
                latents,
                self.vae,
                bounds=bounds,
                octree_depth=flash_octree_depth,
            )

        meshes = [
            trimesh.Trimesh(mesh_v_f[0].astype(np.float32), mesh_v_f[1])
            for mesh_v_f in output
        ]

        # Offload all models
        self.maybe_free_model_hooks()

        if "latents" in debug_dict:
            output_debug_dict["latents"] = latents

        return (output, meshes, output_debug_dict)

        # return TripoSGPipelineOutput(samples=output, meshes=meshes)

    def get_img2img_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = torch.tensor(
            min(num_inference_steps * strength, num_inference_steps)
        ).to(device)

        t_start = int(max(num_inference_steps - init_timestep, 0))
        timesteps = torch.tensor(
            self.scheduler.timesteps[t_start * self.scheduler.order :]
        ).to(device)
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)

        return timesteps, num_inference_steps - t_start

        # SDEdit: strength controls how much noise to add
        # strength=1.0 means start from pure noise (like normal generation)
        # strength=0.0 means no denoising (keep original)
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:].to(device)
        self.scheduler.set_begin_index(t_start)
        # No need to set begin_index for Flow Matching scheduler

    def normalize_points_to_range(self, points, target_range=(-1, 1)):
        """
        将点云居中并归一化到指定范围，保持几何结构

        参数:
            points: 形状为(N, 3)的numpy数组，表示3D点云
            target_range: 目标范围的元组，默认为(-1, 1)

        返回:
            归一化后的点云
        """
        # 计算中心点
        center = (points.max(axis=0) + points.min(axis=0)) / 2

        # 将点云平移到原点
        centered_points = points - center

        # 计算最大距离(保持比例)
        max_distance = np.max(np.abs(centered_points))

        # 归一化到目标范围
        min_val, max_val = target_range
        scale = (max_val - min_val) / 2
        normalized_points = centered_points / max_distance * scale

        return normalized_points

    def mesh2latents(
        self, mesh, dtype=None, num_points=204800, num_tokens=2048, nm=False
    ):
        """
        Converts a mesh to latents by encoding the mesh using the VAE.
        """
        print("Converting mesh to latents...")
        if dtype is None:
            dtype = next(self.vae.parameters()).dtype
        if isinstance(mesh, trimesh.Trimesh):
            # 檢查是否有足夠的面來採樣
            if len(mesh.faces) == 0:
                print("錯誤: mesh 沒有面")
                raise ValueError("Mesh has no faces for sampling")

            try:
                surface_points, face_indices = trimesh.sample.sample_surface(
                    mesh, num_points, seed=42
                )

                # 驗證面索引是否在有效範圍內
                max_face_index = len(mesh.face_normals) - 1
                if len(face_indices) > 0 and face_indices.max() > max_face_index:
                    print(
                        f"警告: 發現無效的面索引，修正中... (max: {face_indices.max()}, limit: {max_face_index})"
                    )
                    face_indices = np.clip(face_indices, 0, max_face_index)

                # 確保面索引不為負數
                if len(face_indices) > 0 and face_indices.min() < 0:
                    print(f"警告: 發現負數面索引 (最小索引: {face_indices.min()})")
                    face_indices = np.clip(face_indices, 0, max_face_index)

                if nm:
                    # Normalize the points to the range [-1, 1]
                    surface_points = self.normalize_points_to_range(
                        surface_points, target_range=(-1, 1)
                    )

                surface_normals = mesh.face_normals[face_indices]
                surface = torch.FloatTensor(surface_points)
                normal = torch.FloatTensor(surface_normals)

                # 使用指定的device而不是硬編碼cuda()
                device = (
                    next(self.vae.parameters()).device
                    if hasattr(self, "vae")
                    else torch.device("cuda:0")
                )
                surface = (
                    torch.cat([surface, normal], dim=-1)
                    .unsqueeze(0)
                    .to(device=device, dtype=dtype)
                )

                return self.vae.encode(surface, num_tokens=num_tokens)

            except Exception as e:
                print(f"採樣表面時出錯: {e}")
                raise e
        elif isinstance(mesh, torch.Tensor):
            return self.vae.encode(mesh, num_tokens=num_tokens)
        elif isinstance(mesh, list):
            return [self.vae.encode(m, num_tokens=num_tokens) for m in mesh]
        else:
            raise ValueError(f"Unsupported mesh type: {type(mesh)}")

    def latent2mesh(
        self,
        latents,
        bounds=(-1.005, -1.005, -1.005, 1.005, 1.005, 1.005),
        dense_octree_depth=8,
        hierarchical_octree_depth=8,
        flash_octree_depth=8,
        use_flash_decoder=False,
    ):
        if not use_flash_decoder:
            geometric_func = lambda x: self.vae.decode(latents, sampled_points=x).sample
            output = hierarchical_extract_geometry(
                geometric_func,
                self.device,
                bounds=bounds,
                dense_octree_depth=dense_octree_depth,
                hierarchical_octree_depth=hierarchical_octree_depth,
            )
        else:
            self.vae.set_flash_decoder()
            output = flash_extract_geometry(
                latents,
                self.vae,
                bounds=bounds,
                octree_depth=flash_octree_depth,
            )
        meshes = [
            trimesh.Trimesh(mesh_v_f[0].astype(np.float32), mesh_v_f[1])
            for mesh_v_f in output
        ]
        return meshes

    def prepare_noise_latents(
        self,
        mesh,
        timestep,
        batch_size,
        num_shapes_per_prompt,
        dtype,
        device,
        generator=None,
        init_latents=None,
    ):
        if not isinstance(mesh, (torch.Tensor, trimesh.Trimesh, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `trimesh.Trimesh` or list but is {type(mesh)}"
            )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        elif isinstance(generator, list):
            print("not support  list input yet")
            # init_latents = [
            #     retrieve_latents(self.mesh2latent(image[i : i + 1]), generator=generator[i])
            #     for i in range(batch_size)
            # ]
            # init_latents = torch.cat(init_latents, dim=0)
        else:
            if init_latents is None:
                init_latents = retrieve_latents(
                    self.mesh2latents(mesh, dtype=dtype), generator=generator
                ).to(device)
            else:
                print("Using provided init_latents in prepare_noise_latents")
                if len(init_latents.shape) == 2:
                    init_latents = init_latents.unsqueeze(0)
                # Apply VAE scaling if available
        # if hasattr(self.vae.config, 'scaling_factor') and self.vae.config.scaling_factor != 1.0:
        #     init_latents = init_latents * self.vae.config.scaling_factor
        # 3D VAE don't have shift fctor
        # init_latents = (init_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        if (
            batch_size > init_latents.shape[0]
            and batch_size % init_latents.shape[0] == 0
        ):
            # expand init_latents for batch_size
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat(
                [init_latents] * additional_image_per_prompt, dim=0
            )
        elif (
            batch_size > init_latents.shape[0]
            and batch_size % init_latents.shape[0] != 0
        ):
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        shape = init_latents.shape
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # get latents
        timestep = timestep.to(device=device, dtype=dtype)
        # NOTE: The scedulet provide here has different parameters order
        init_latents = self.scheduler.scale_noise(init_latents, noise, timestep)
        latents = init_latents.to(device=device, dtype=dtype)

        return latents

    def prepare_sdedit_latents(
        self,
        init_latents,
        timestep,
        batch_size,
        num_shapes_per_prompt,
        dtype,
        device,
        generator=None,
    ):

        if len(init_latents.shape) == 2:
            init_latents = init_latents.unsqueeze(0)
            # Apply VAE scaling if available
        # if hasattr(self.vae.config, 'scaling_factor') and self.vae.config.scaling_factor != 1.0:
        #     init_latents = init_latents * self.vae.config.scaling_factor
        # 3D VAE don't have shift fctor
        # init_latents = (init_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        if (
            batch_size > init_latents.shape[0]
            and batch_size % init_latents.shape[0] == 0
        ):
            # expand init_latents for batch_size
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat(
                [init_latents] * additional_image_per_prompt, dim=0
            )
        elif (
            batch_size > init_latents.shape[0]
            and batch_size % init_latents.shape[0] != 0
        ):
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        shape = init_latents.shape
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # get latents
        timestep = timestep.to(device=device, dtype=dtype)
        # NOTE: The scedulet provide here has different parameters order
        init_latents = self.scheduler.scale_noise(init_latents, noise, timestep)
        latents = init_latents.to(device=device, dtype=dtype)

        return latents, noise

    @torch.no_grad()
    def mesh2mesh(
        self,
        mesh: trimesh.Trimesh,
        image: PipelineImageInput,
        num_inference_steps: int = 50,
        num_tokens: int = 2048,
        strength: float = 0.6,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        num_shapes_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        bounds: Union[Tuple[float], List[float], float] = (
            -1.005,
            -1.005,
            -1.005,
            1.005,
            1.005,
            1.005,
        ),
        dense_octree_depth: int = 8,
        hierarchical_octree_depth: int = 8,
        flash_octree_depth: int = 8,
        use_flash_decoder: bool = False,
        return_dict: bool = True,
        debug_dict={},
        no_output_mesh=False,
        add_on_dict={},
    ):
        # 1. Define call parameters
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs

        # 2. Define call parameters
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        elif isinstance(image, torch.Tensor):
            batch_size = image.shape[0]
        else:
            raise ValueError("Invalid input type for image")

        device = self._execution_device

        # 3. Encode condition
        image_embeds, negative_image_embeds = self.encode_image(
            image, device, num_shapes_per_prompt
        )

        if self.do_classifier_free_guidance:
            image_embeds = torch.cat([negative_image_embeds, image_embeds], dim=0)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )
        timesteps, num_inference_steps = self.get_img2img_timesteps(
            num_inference_steps, strength, device
        )
        latent_timestep = timesteps[:1].repeat(batch_size * num_shapes_per_prompt)

        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        # 5. Prepare latent variables
        if latents is None:
            latents = self.prepare_noise_latents(
                mesh,
                latent_timestep,
                batch_size,
                num_shapes_per_prompt,
                image_embeds.dtype,
                device,
                generator,
            )
        else:
            print("‼️ Using provided latents in mesh2mesh (ignore input mesh) ‼️")
            latents = self.prepare_noise_latents(
                mesh,
                latent_timestep,
                batch_size,
                num_shapes_per_prompt,
                image_embeds.dtype,
                device,
                generator,
                init_latents=latents,
            )
        output_debug_dict = {"vis_intermediate": []}
        if "latents_history" in debug_dict:
            output_debug_dict["latents_history"] = []
            output_debug_dict["latents_history"].append(latents.clone())

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=image_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_image = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_image - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                # if "repaint" in add_on_dict:
                #     init_latents_proper = image_latents
                #     if self.do_classifier_free_guidance:
                #         init_mask, _ = mask.chunk(2)
                #     else:
                #         init_mask = mask

                #     if i < len(timesteps) - 1:
                #         noise_timestep = timesteps[i + 1]
                #     init_latents = self.scheduler.scale_noise(init_latents, noise, timestep)

                #     latents = (
                #         1 - init_mask
                #     ) * init_latents_proper + init_mask * latents

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        if "latents_history" in debug_dict:
            output_debug_dict["latents_history"].append(latents.clone())
        # 7. decoder mesh
        if no_output_mesh:
            print("No output mesh: using minimum depth")
            meshes = None
            output = None

        else:
            print("Extracting Mesh ...")
            if not use_flash_decoder:
                geometric_func = lambda x: self.vae.decode(
                    latents, sampled_points=x
                ).sample
                output = hierarchical_extract_geometry(
                    geometric_func,
                    device,
                    bounds=bounds,
                    dense_octree_depth=dense_octree_depth,
                    hierarchical_octree_depth=hierarchical_octree_depth,
                )
            else:
                self.vae.set_flash_decoder()
                output = flash_extract_geometry(
                    latents,
                    self.vae,
                    bounds=bounds,
                    octree_depth=flash_octree_depth,
                )
            meshes = [
                trimesh.Trimesh(mesh_v_f[0].astype(np.float32), mesh_v_f[1])
                for mesh_v_f in output
            ]

        # Offload all models
        self.maybe_free_model_hooks()

        if "latents" in debug_dict:
            output_debug_dict["latents"] = latents

        return (output, meshes, output_debug_dict)

    @torch.no_grad()
    def latent_edit(
        self,
        image: PipelineImageInput,
        latents: Optional[torch.FloatTensor],
        num_inference_steps: int = 50,
        num_tokens: int = 2048,
        strength: float = 0.6,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        num_shapes_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        bounds: Union[Tuple[float], List[float], float] = (
            -1.005,
            -1.005,
            -1.005,
            1.005,
            1.005,
            1.005,
        ),
        dense_octree_depth: int = 8,
        hierarchical_octree_depth: int = 8,
        flash_octree_depth: int = 8,
        use_flash_decoder: bool = False,
        return_dict: bool = True,
        debug_dict={},
        no_output_mesh=False,
        add_on_dict={},
    ):
        # 1. Define call parameters
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs

        # 2. Define call parameters
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        elif isinstance(image, torch.Tensor):
            batch_size = image.shape[0]
        else:
            raise ValueError("Invalid input type for image")

        device = self._execution_device

        # 3. Encode condition
        image_embeds, negative_image_embeds = self.encode_image(
            image, device, num_shapes_per_prompt
        )

        if self.do_classifier_free_guidance:
            image_embeds = torch.cat([negative_image_embeds, image_embeds], dim=0)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )
        timesteps, num_inference_steps = self.get_img2img_timesteps(
            num_inference_steps, strength, device
        )
        latent_timestep = timesteps[:1].repeat(batch_size * num_shapes_per_prompt)

        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        # 5. Prepare latent variables

        if "repaint" in add_on_dict:

            select_idx = add_on_dict["repaint"]["fixed_tokens"]
            latents_input = latents.clone()
            latents_mask = torch.zeros_like(latents)
            latents_mask[:, select_idx, :] = 1.0

        latents, noise = self.prepare_sdedit_latents(
            latents,
            latent_timestep,
            batch_size,
            num_shapes_per_prompt,
            image_embeds.dtype,
            device,
            generator,
        )
        # output_debug_dict = {"vis_intermediate": []}
        output_debug_dict = {}
        if "latents_history" in debug_dict:
            output_debug_dict["latents_history"] = []
        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if "latents_history" in debug_dict:
                    output_debug_dict["latents_history"].append(latents.clone())
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=image_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_image = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_image - noise_pred_uncond
                    )
                    if "mute_cross" in add_on_dict:
                        mute_idx = add_on_dict["mute_cross"]["mute_index"]
                        # print("MOISE PRED:", noise_pred.shape)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (
                            noise_pred_image - noise_pred_uncond
                        )
                        noise_pred[:, mute_idx, :] = noise_pred_uncond[:, mute_idx, :]

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                if "repaint" in add_on_dict:
                    init_mask = latents_mask

                    if i < len(timesteps) - 1:
                        noise_timestep = timesteps[i + 1]
                        init_latents = self.scheduler.scale_noise(
                            latents_input.clone(), noise, noise_timestep
                        )
                    else:
                        init_latents = latents_input.clone()
                    # latents = (1 - init_mask) * init_latents + init_mask * latents
                    latents = (1 - init_mask) * latents + init_mask * init_latents

                    


                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
        if len(output_debug_dict) > 0:
            return latents, output_debug_dict
        return latents

    @torch.no_grad()
    def adaptive_latent_edit(
        self,
        image: PipelineImageInput,
        latents: Optional[torch.FloatTensor],
        mask_image: Optional[torch.FloatTensor],
        fix_tokens: Optional[List[int]],
        num_inference_steps: int = 50,
        num_tokens: int = 2048,
        strength: float = 0.6,
        step_start_pruning: int = 10,
        step_stop_pruning: int = 20,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        num_shapes_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        bounds: Union[Tuple[float], List[float], float] = (
            -1.005,
            -1.005,
            -1.005,
            1.005,
            1.005,
            1.005,
        ),
        debug_dict={},
        add_on_dict={},
        **kwargs, # Control of adap
    ):
        # 1. Define call parameters
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs

        # 2. Define call parameters
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        elif isinstance(image, torch.Tensor):
            batch_size = image.shape[0]
        else:
            raise ValueError("Invalid input type for image")

        device = self._execution_device

        # 3. Encode condition
        image_embeds, negative_image_embeds = self.encode_image(
            image, device, num_shapes_per_prompt
        )

        if self.do_classifier_free_guidance:
            image_embeds = torch.cat([negative_image_embeds, image_embeds], dim=0)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )
        timesteps, num_inference_steps = self.get_img2img_timesteps(
            num_inference_steps, strength, device
        )
        latent_timestep = timesteps[:1].repeat(batch_size * num_shapes_per_prompt)

        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels

        if fix_tokens is not None:
            latents_input = latents.clone()
            latents_mask = torch.ones_like(latents)
            latents_mask[:, fix_tokens, :] = 0.0
            init_mask = latents_mask
            fix_tokens_mask = torch.zeros(latents.shape[1], dtype=bool, device=latents.device)
            fix_tokens_mask[fix_tokens] = True

        # 5. Prepare latent variables
        latents, noise = self.prepare_sdedit_latents(
            latents,
            latent_timestep,
            batch_size,
            num_shapes_per_prompt,
            image_embeds.dtype,
            device,
            generator,
        )

        # output_debug_dict = {"vis_intermediate": []}
        output_debug_dict = {}
        if "latents_history" in debug_dict:
            output_debug_dict["latents_history"] = []
        # 6. Denoising loop
        dit_control = TripoSGDiTControl(self_attention=True) # Set False for memory saving
        # Register the transformer with DiT control
        dit_control.register(self.transformer)
        # load 2d mask -> token mask
        N2d=256 # Dino2 Patchsize
        mask = mask_image.resize((int(N2d**0.5), int(N2d**0.5)))
        mask = np.array(mask) / 255.0  # Normalize to [0, 1]
        mask = np.where(mask > 0.5, 1, 0)  # Binarize the mask
        token_mask = mask.flatten().astype(bool)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if "latents_history" in debug_dict:
                    output_debug_dict["latents_history"].append(latents.clone())

                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=image_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_image = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_image - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                if fix_tokens is not None:
                    # Pruning Unimportant Tokens based on Attention
                    if i > step_start_pruning and i < step_stop_pruning:
                        # Tracking attention focus
                        token_attentive_2d, token_freq, kl_str = attention2d(dit_control, i, token_mask, **kwargs)
                        print(f"☀️ Step = {i}. First stage attentive 2d tokens:", len(token_attentive_2d))
                        # Further Focusing on Specially Aggregated 
                        # topk_each_block=int(min(len(token_freq)*0.95,2*len(token_attentive_2d)))
                        token_attentive_3d, kl_freq = attention_self(dit_control, i, latents.shape[1], token_attentive_2d, **kwargs)
                        print(f"☀️ Step = {i}. Second stage attentive 3d tokens:", len(token_attentive_3d))
                        print(f"☀️ topk_each block = {int(min(len(token_freq)*0.95,2*len(token_attentive_3d)))}, token_freq len = {len(token_freq)}")
                        # Attention2d return important tokens, we want to remove the res
                        total_tokens = token_freq.shape[0] 
                        device = latents.device
                        
                        # 確保 token_attentive_2d 在正確設備上
                        if isinstance(token_attentive_3d, torch.Tensor):
                            token_attentive_3d = token_attentive_3d.to(device)
                        else:
                            token_attentive_3d = torch.tensor(token_attentive_3d, device=device)
                        
                        # 分離fix_tokens和flex_tokens區域
                        num_fix_tokens = len(fix_tokens)  # 1248
                        
                        # 只從flex_tokens區域選擇要移除的tokens
                        flex_start_idx = num_fix_tokens  # 1248
                        flex_indices = torch.arange(flex_start_idx, latents.shape[1], device=device)
                        
                        # 找出flex區域中的attentive tokens (要保留的)
                        flex_attentive_mask = torch.isin(token_attentive_3d, flex_indices)
                        flex_attentive_tokens = token_attentive_3d[flex_attentive_mask]

                        # 計算flex區域中要移除的tokens
                        flex_remove_mask = ~torch.isin(flex_indices, flex_attentive_tokens)
                        flex_tokens_to_remove = flex_indices[flex_remove_mask]
                        
                        if len(flex_tokens_to_remove) > 0:
                            # 創建最終的keep_mask
                            all_indices = torch.arange(latents.shape[1], device=device)
                            keep_mask = ~torch.isin(all_indices, flex_tokens_to_remove)
                            
                            # 同時移除所有相關張量中的tokens
                            latents = latents[:, keep_mask, :]
                            init_mask = init_mask[:, keep_mask, :]
                            latents_input = latents_input[:, keep_mask, :]
                            noise = noise[:, keep_mask, :]
                            
                            # 更新fix_tokens_mask，但fix_tokens的位置保持不變
                            new_fix_tokens_mask = torch.zeros(latents.shape[1], dtype=bool, device=device)
                            new_fix_tokens_mask[:num_fix_tokens] = True  # 前1248個仍然是fix_tokens
                            fix_tokens_mask = new_fix_tokens_mask 
                            print(f"🔥remove {len(flex_tokens_to_remove)} tokens")
                        else:
                            print("🔥No tokens removed")
                            pass
                    # Repaint replace latent with initial fixed latents
                    if i < len(timesteps) - 1:
                        noise_timestep = timesteps[i + 1]
                        init_latents = self.scheduler.scale_noise(
                            latents_input.clone(), noise, noise_timestep
                        )
                    else:
                        init_latents = latents_input.clone()
                    latents = (1 - init_mask) * init_latents + init_mask * latents
                    

                # Cut off
                # Check and Dropoutliers

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    print("🔥CALL BACK")
                    latents = callback_outputs.pop("latents", latents)

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
        dit_control.unregister()
        if len(output_debug_dict) > 0:
            return latents, output_debug_dict
        return latents

    @torch.no_grad()
    def prune_relax_latent_edit(
        self,
        image: PipelineImageInput,
        latents: Optional[torch.FloatTensor],
        mask_image: Optional[torch.FloatTensor],
        fix_tokens: Optional[List[int]],
        num_inference_steps: int = 50,
        num_tokens: int = 2048,
        strength: float = 0.6,
        step_pruning: int = 10,
        step_relaxing: int = 10,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        num_shapes_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        bounds: Union[Tuple[float], List[float], float] = (
            -1.005,
            -1.005,
            -1.005,
            1.005,
            1.005,
            1.005,
        ),
        debug_dict={},
        add_on_dict={},
        pruning_cut_off_p=0.1,
        relaxing_cut_off_p=0.1,
        **kwargs, # Control of adap
    ):
        # 1. Define call parameters
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs

        # 2. Define call parameters
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        elif isinstance(image, torch.Tensor):
            batch_size = image.shape[0]
        else:
            raise ValueError("Invalid input type for image")

        device = self._execution_device

        # 3. Encode condition
        image_embeds, negative_image_embeds = self.encode_image(
            image, device, num_shapes_per_prompt
        )

        if self.do_classifier_free_guidance:
            image_embeds = torch.cat([negative_image_embeds, image_embeds], dim=0)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )
        timesteps, num_inference_steps = self.get_img2img_timesteps(
            num_inference_steps, strength, device
        )
        latent_timestep = timesteps[:1].repeat(batch_size * num_shapes_per_prompt)

        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels

        if fix_tokens is not None:
            latents_input = latents.clone()
            latents_mask = torch.ones_like(latents)
            latents_mask[:, fix_tokens, :] = 0.0
            init_mask = latents_mask
            fix_tokens_mask = torch.zeros(latents.shape[1], dtype=bool, device=latents.device)
            fix_tokens_mask[fix_tokens] = True

        # 5. Prepare latent variables
        latents, noise = self.prepare_sdedit_latents(
            latents,
            latent_timestep,
            batch_size,
            num_shapes_per_prompt,
            image_embeds.dtype,
            device,
            generator,
        )

        # output_debug_dict = {"vis_intermediate": []}
        output_debug_dict = {}
        if "latents_history" in debug_dict:
            output_debug_dict["latents_history"] = []
        # 6. Denoising loop
        dit_control = TripoSGDiTControl(self_attention=True) # Set False for memory saving
        # Register the transformer with DiT control
        dit_control.register(self.transformer)
        # load 2d mask -> token mask
        N2d=256 # Dino2 Patchsize
        mask = mask_image.resize((int(N2d**0.5), int(N2d**0.5)))
        mask = np.array(mask) / 255.0  # Normalize to [0, 1]
        mask = np.where(mask > 0.5, 1, 0)  # Binarize the mask
        token_mask = mask.flatten().astype(bool)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if "latents_history" in debug_dict:
                    output_debug_dict["latents_history"].append(latents.clone())

                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=image_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_image = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_image - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                if fix_tokens is not None:
                    
                    # Relax editing related tokens
                    if i==step_relaxing:
                        # Tracking attention focus
                        token_attentive_2d, token_freq, kl_str = attention2d(dit_control, i, token_mask, **kwargs)
                        print(f"🧸 (Relaxing) Step = {i}. First stage attentive 2d tokens:", len(token_attentive_2d))
                        # Further Focusing on Specially Aggregated 
                        token_attentive_3d, kl_freq = attention_self(dit_control, i, latents.shape[1], token_attentive_2d, cut_off_p=relaxing_cut_off_p, **kwargs)
                        print(f"🧸 (Relaxing) Step = {i}. Second stage attentive 3d tokens:", len(token_attentive_3d))
                        
                        device = latents.device
                        # 確保 token_attentive_2d 在正確設備上
                        if isinstance(token_attentive_3d, torch.Tensor):
                            token_attentive_3d = token_attentive_3d.to(device)
                        else:
                            token_attentive_3d = torch.tensor(token_attentive_3d, device=device)
                        # Relax all above threshold tokens for editing
                        non_zero_count = init_mask.mean(dim=-1).nonzero(as_tuple=True)[1]
                        print(f"🧸 Before relaxing {len(non_zero_count)} tokens for editing")
                        init_mask[:, token_attentive_3d, :] = 1
                        non_zero_count = init_mask.mean(dim=-1).nonzero(as_tuple=True)[1]
                        print(f"🧸 After relaxing {len(non_zero_count)} tokens for editing")
                        print(f"🧸 Unregister dit_control after relaxing")
                        dit_control.unregister()
                    # Pruning Unimportant Tokens based on Attention
                    if i ==step_pruning:
                        # Tracking attention focus
                        token_attentive_2d, token_freq, kl_str = attention2d(dit_control, i, token_mask, **kwargs)
                        print(f"☀️ Step = {i}. First stage attentive 2d tokens:", len(token_attentive_2d))
                        # Further Focusing on Specially Aggregated 
                        token_attentive_3d, kl_freq = attention_self(dit_control, i, latents.shape[1], token_attentive_2d, cut_off_p=pruning_cut_off_p, **kwargs)
                        print(f"☀️ Step = {i}. Second stage attentive 3d tokens:", len(token_attentive_3d))
                        
                        # Attention2d return important tokens, we want to remove the res
                        total_tokens = token_freq.shape[0] 
                        device = latents.device
                        
                        # 確保 token_attentive_2d 在正確設備上
                        if isinstance(token_attentive_3d, torch.Tensor):
                            token_attentive_3d = token_attentive_3d.to(device)
                        else:
                            token_attentive_3d = torch.tensor(token_attentive_3d, device=device)
                        
                        # 分離fix_tokens和flex_tokens區域
                        num_fix_tokens = len(fix_tokens)
                        
                        # 只從flex_tokens區域選擇要移除的tokens
                        flex_start_idx = num_fix_tokens 
                        flex_indices = torch.arange(flex_start_idx, latents.shape[1], device=device)
                        # 找出flex區域中的attentive tokens (要保留的)
                        flex_attentive_mask = torch.isin(token_attentive_3d, flex_indices)
                        flex_attentive_tokens = token_attentive_3d[flex_attentive_mask]
                        # 計算flex區域中要移除的tokens
                        flex_remove_mask = ~torch.isin(flex_indices, flex_attentive_tokens)
                        flex_tokens_to_remove = flex_indices[flex_remove_mask]
                        
                        if len(flex_tokens_to_remove) > 0:
                            # 創建最終的keep_mask
                            all_indices = torch.arange(latents.shape[1], device=device)
                            keep_mask = ~torch.isin(all_indices, flex_tokens_to_remove)
                            
                            # 同時移除所有相關張量中的tokens
                            latents = latents[:, keep_mask, :]
                            init_mask = init_mask[:, keep_mask, :]
                            latents_input = latents_input[:, keep_mask, :]
                            noise = noise[:, keep_mask, :]
                            
                            # 更新fix_tokens_mask，但fix_tokens的位置保持不變
                            new_fix_tokens_mask = torch.zeros(latents.shape[1], dtype=bool, device=device)
                            new_fix_tokens_mask[:num_fix_tokens] = True  # 前1248個仍然是fix_tokens
                            fix_tokens_mask = new_fix_tokens_mask 
                            print(f"🔥remove {len(flex_tokens_to_remove)} tokens")
                        else:
                            print("🔥No tokens removed")
                            pass

                    # Repaint replace latent with initial fixed latents
                    if i < len(timesteps) - 1:
                        noise_timestep = timesteps[i + 1]
                        noisy_init_latents = self.scheduler.scale_noise(
                            latents_input.clone(), noise, noise_timestep
                        )
                    else:
                        noisy_init_latents = latents_input.clone()
                    latents = (1 - init_mask) * noisy_init_latents + init_mask * latents
                    

                # Cut off
                # Check and Dropoutliers

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    print("🔥CALL BACK")
                    latents = callback_outputs.pop("latents", latents)

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
        
        if len(output_debug_dict) > 0:
            return latents, output_debug_dict
        return latents

    @torch.no_grad()
    def prune_latent_edit_v2(
        self,
        image: PipelineImageInput,
        latents: Optional[torch.FloatTensor],
        mask_image: Optional[torch.FloatTensor],
        fix_tokens: Optional[List[int]],
        num_inference_steps: int = 50,
        num_tokens: int = 2048,
        strength: float = 0.6,
        step_pruning: int = 10,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        num_shapes_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        bounds: Union[Tuple[float], List[float], float] = (
            -1.005,
            -1.005,
            -1.005,
            1.005,
            1.005,
            1.005,
        ),
        debug_dict={},
        add_on_dict={},
        pruning_cut_off_p=0.1,
        **kwargs, # Control of adap
    ):
        # 1. Define call parameters
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs

        # 2. Define call parameters
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        elif isinstance(image, torch.Tensor):
            batch_size = image.shape[0]
        else:
            raise ValueError("Invalid input type for image")

        device = self._execution_device

        # 3. Encode condition
        image_embeds, negative_image_embeds = self.encode_image(
            image, device, num_shapes_per_prompt
        )

        if self.do_classifier_free_guidance:
            image_embeds = torch.cat([negative_image_embeds, image_embeds], dim=0)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )
        timesteps, num_inference_steps = self.get_img2img_timesteps(
            num_inference_steps, strength, device
        )
        latent_timestep = timesteps[:1].repeat(batch_size * num_shapes_per_prompt)

        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels

        if fix_tokens is not None:
            latents_input = latents.clone()
            latents_mask = torch.ones_like(latents)
            latents_mask[:, fix_tokens, :] = 0.0
            init_mask = latents_mask
            fix_tokens_mask = torch.zeros(latents.shape[1], dtype=bool, device=latents.device)
            fix_tokens_mask[fix_tokens] = True

        # 5. Prepare latent variables
        latents, noise = self.prepare_sdedit_latents(
            latents,
            latent_timestep,
            batch_size,
            num_shapes_per_prompt,
            image_embeds.dtype,
            device,
            generator,
        )

        # output_debug_dict = {"vis_intermediate": []}
        output_debug_dict = {}
        if "latents_history" in debug_dict:
            output_debug_dict["latents_history"] = []
        # 6. Denoising loop
        dit_control = TripoSGDiTControl(self_attention=True) # Set False for memory saving
        # Register the transformer with DiT control
        dit_control.register(self.transformer)
        unregister_flag = False
        # load 2d mask -> token mask
        N2d=256 # Dino2 Patchsize
        mask = mask_image.resize((int(N2d**0.5), int(N2d**0.5)))
        mask = np.array(mask) / 255.0  # Normalize to [0, 1]
        mask = np.where(mask > 0.5, 1, 0)  # Binarize the mask
        token_mask = mask.flatten().astype(bool)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if "latents_history" in debug_dict:
                    output_debug_dict["latents_history"].append(latents.clone())

                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=image_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_image = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_image - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                if fix_tokens is not None:
                    # Pruning Unimportant Tokens based on Attention
                    if i ==step_pruning:
                        # 1. Overlap tokens (REMOVE)
                        token_overlap, kl_freq = attention_self(dit_control, i, latents.shape[1], fix_tokens, cut_off_p=pruning_cut_off_p, **kwargs)
                        print(f"🧸 Overlap tokens between fix and attentive tokens:", len(token_overlap))
                        # 2. 2d-related attentive tokens (PRESERVE)
                        token_attentive_2d, token_freq, kl_str = attention2d(dit_control, i, token_mask, **kwargs)
                        print(f"☀️ Step = {i}. First stage (image) attentive 2d tokens:", len(token_attentive_2d))
                        # Further Focusing on Specially Aggregated 
                        token_attentive_2d, kl_freq = attention_self(dit_control, i, latents.shape[1], token_attentive_2d, cut_off_p=pruning_cut_off_p, **kwargs)
                        print(f"☀️ Step = {i}. Second stage (spatial) attentive 2d tokens:", len(token_attentive_2d))
                        
                        # Attention2d return important tokens, we want to remove the res
                        total_tokens = token_freq.shape[0] 
                        device = latents.device
                        
                        # 確保 token_attentive_2d 在正確設備上
                        if isinstance(token_attentive_2d, torch.Tensor):
                            token_attentive_2d = token_attentive_2d.to(device)
                        else:
                            token_attentive_2d = torch.tensor(token_attentive_2d, device=device)

                        # 確保 token_overlap 在正確設備上（若存在）
                        if isinstance(token_overlap, torch.Tensor):
                            token_overlap = token_overlap.to(device)
                        else:
                            token_overlap = torch.tensor(token_overlap, device=device)

                        # 分離fix_tokens和flex_tokens區域
                        num_fix_tokens = len(fix_tokens)

                        # 只從flex_tokens區域選擇要移除的tokens
                        flex_start_idx = num_fix_tokens
                        flex_indices = torch.arange(flex_start_idx, latents.shape[1], device=device)

                        # 找出flex區域中的attentive tokens (要保留的)
                        if token_attentive_2d.numel() > 0:
                            flex_attentive_mask = torch.isin(token_attentive_2d, flex_indices)
                            flex_attentive_tokens = token_attentive_2d[flex_attentive_mask]
                        else:
                            flex_attentive_tokens = torch.tensor([], device=device, dtype=torch.long)

                        # 找出flex區域中重疊的 tokens (屬於 token_overlap)
                        if token_overlap.numel() > 0:
                            flex_overlap_mask = torch.isin(token_overlap, flex_indices)
                            flex_overlap_tokens = token_overlap[flex_overlap_mask]
                        else:
                            flex_overlap_tokens = torch.tensor([], device=device, dtype=torch.long)

                        # 移除條件：在 flex 區域 且 (不在 attentive_2d) OR (在 token_overlap)
                        if flex_attentive_tokens.numel() > 0:
                            in_attentive = torch.isin(flex_indices, flex_attentive_tokens)
                        else:
                            in_attentive = torch.zeros_like(flex_indices, dtype=torch.bool, device=device)

                        if flex_overlap_tokens.numel() > 0:
                            in_overlap = torch.isin(flex_indices, flex_overlap_tokens)
                        else:
                            in_overlap = torch.zeros_like(flex_indices, dtype=torch.bool, device=device)
                    
                        remove_mask = (~in_attentive)&in_overlap
                        flex_tokens_to_remove = flex_indices[remove_mask]
                        print(f"🔥 flex_indices len: {len(flex_indices)}, in_attentive sum: {in_attentive.sum().item()}, in_overlap sum: {in_overlap.sum().item()}")
                        print(f"🔥 flex_tokens_to_remove len: {len(flex_tokens_to_remove)} Remaining: {len(flex_indices) - len(flex_tokens_to_remove)}")
                        if len(flex_tokens_to_remove) > 0:
                            # 創建最終的keep_mask
                            all_indices = torch.arange(latents.shape[1], device=device)
                            keep_mask = ~torch.isin(all_indices, flex_tokens_to_remove)
                            
                            # 同時移除所有相關張量中的tokens
                            latents = latents[:, keep_mask, :]
                            init_mask = init_mask[:, keep_mask, :]
                            latents_input = latents_input[:, keep_mask, :]
                            noise = noise[:, keep_mask, :]
                            
                            # 更新fix_tokens_mask，但fix_tokens的位置保持不變
                            new_fix_tokens_mask = torch.zeros(latents.shape[1], dtype=bool, device=device)
                            new_fix_tokens_mask[:num_fix_tokens] = True  # 前1248個仍然是fix_tokens
                            fix_tokens_mask = new_fix_tokens_mask 
                        else:
                            print("🔥No tokens removed")
                            pass
                        
                        dit_control.unregister()
                        unregister_flag = True

                    # Repaint replace latent with initial fixed latents
                    if i < len(timesteps) - 1:
                        noise_timestep = timesteps[i + 1]
                        noisy_init_latents = self.scheduler.scale_noise(
                            latents_input.clone(), noise, noise_timestep
                        )
                    else:
                        noisy_init_latents = latents_input.clone()
                    latents = (1 - init_mask) * noisy_init_latents + init_mask * latents
                    
                # Check and Dropoutliers

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    print("🔥CALL BACK")
                    latents = callback_outputs.pop("latents", latents)

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
        if not unregister_flag:
            dit_control.unregister()
        if len(output_debug_dict) > 0:
            return latents, output_debug_dict
        return latents


    @torch.no_grad()
    def adaptive_latent_flow_edit(
        self,
        tar_image: PipelineImageInput, # 🌊FlowEdit
        src_image: PipelineImageInput, # 🌊FlowEdit
        latents: Optional[torch.FloatTensor],
        mask_image: Optional[torch.FloatTensor],
        fix_tokens: Optional[List[int]],
        num_inference_steps: int = 50,
        num_tokens: int = 2048,
        timesteps: List[int] = None,
        src_guidance_scale: float = 2.5, # 🌊FlowEdit
        tar_guidance_scale: float = 7.0, # 🌊FlowEdit
        n_min=0, # 🌊FlowEdit
        n_max=33, # 🌊FlowEdit
        n_avg=1, # 🌊FlowEdit
        num_shapes_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        bounds: Union[Tuple[float], List[float], float] = (
            -1.005,
            -1.005,
            -1.005,
            1.005,
            1.005,
            1.005,
        ),
        dense_octree_depth: int = 8,
        hierarchical_octree_depth: int = 8,
        flash_octree_depth: int = 8,
        use_flash_decoder: bool = False,
        return_dict: bool = True,
        debug_dict={},
        no_output_mesh=False,
        add_on_dict={},
        **kwargs, # 🥷Adaptive Tokens Pruning
    ):
        def calc_v_tripo(pipe, src_tar_latent_model_input, src_tar_prompt_embeds, src_guidance_scale, tar_guidance_scale, t):
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(src_tar_latent_model_input.shape[0])

            with torch.no_grad():
                # # predict the noise for the source prompt
                noise_pred_src_tar = pipe.transformer(
                    src_tar_latent_model_input,
                    timestep,
                    encoder_hidden_states=src_tar_prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance source
                if pipe.do_classifier_free_guidance:
                    src_noise_pred_uncond, src_noise_pred_text, tar_noise_pred_uncond, tar_noise_pred_text = noise_pred_src_tar.chunk(4)
                    noise_pred_src = src_noise_pred_uncond + src_guidance_scale * (src_noise_pred_text - src_noise_pred_uncond)
                    noise_pred_tar = tar_noise_pred_uncond + tar_guidance_scale * (tar_noise_pred_text - tar_noise_pred_uncond)

            return noise_pred_src, noise_pred_tar
        

        # 1. Define call parameters
        self._guidance_scale = tar_guidance_scale
        self._attention_kwargs = attention_kwargs

        # 2. Define call parameters
        if isinstance(tar_image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(tar_image, list):
            batch_size = len(tar_image)
        elif isinstance(tar_image, torch.Tensor):
            batch_size = tar_image.shape[0]
        else:
            raise ValueError("Invalid input type for image")

        device = self._execution_device

        # 3. Encode condition
        self._guidance_scale = src_guidance_scale
        src_image_embeds, negative_src_image_embeds = self.encode_image(
            src_image, device, num_shapes_per_prompt
        )
        self._guidance_scale = tar_guidance_scale
        tar_image_embeds, negative_tar_image_embeds = self.encode_image(
            tar_image, device, num_shapes_per_prompt
        )

        if self.do_classifier_free_guidance:
            src_tar_image_embeds = torch.cat([negative_src_image_embeds, src_image_embeds, negative_tar_image_embeds, tar_image_embeds], dim=0)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )

        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels

        if fix_tokens is not None:
            latents_input = latents.clone()
            latents_mask = torch.ones_like(latents)
            latents_mask[:, fix_tokens, :] = 0.0
            init_mask = latents_mask
            fix_tokens_mask = torch.zeros(latents.shape[1], dtype=bool, device=latents.device)
            fix_tokens_mask[fix_tokens] = True

        output_debug_dict = {}
        if "latents_history" in debug_dict:
            output_debug_dict["latents_history"] = []

        # 6. Record Attention Distribution for dynamic token pruning
        # Register the transformer with DiT control
        dit_control = TripoSGDiTControl(self_attention=True) # Set False for memory saving
        dit_control.register(self.transformer)
        # load 2d mask -> token mask
        N2d=256 # Dino2 Patchsize
        mask = mask_image.resize((int(N2d**0.5), int(N2d**0.5)))
        mask = np.array(mask) / 255.0  # Normalize to [0, 1]
        mask = np.where(mask > 0.5, 1, 0)  # Binarize the mask
        token_mask = mask.flatten().astype(bool)

        # 7. Denoising loop with 🌊FLOWEDIT
        x_src = latents.clone()
        zt_edit = latents
        T_steps=num_inference_steps
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if T_steps - i > n_max:
                    continue
                t_i = t/1000
                if i+1 < len(timesteps): 
                    t_im1 = (timesteps[i+1])/1000
                else:
                    t_im1 = torch.zeros_like(t_i).to(t_i.device)
                
                if T_steps - i > n_min:
                    # Calculate the average of the V predictions
                    V_delta_avg = torch.zeros_like(x_src)
                    latents_dtype = latents.dtype
                    for k in range(n_avg):
                        
                        fwd_noise = torch.randn_like(x_src).to(x_src.device)
                        zt_src = self.scheduler.scale_noise(
                            x_src.clone(), fwd_noise, timesteps[i]
                        )

                        zt_tar = zt_edit + zt_src - x_src

                        src_tar_latent_model_input = torch.cat([zt_src, zt_src, zt_tar, zt_tar]) if self.do_classifier_free_guidance else (zt_src, zt_tar) 

                        Vt_src, Vt_tar = calc_v_tripo(self, src_tar_latent_model_input,src_tar_image_embeds, src_guidance_scale, tar_guidance_scale, t)

                        V_delta_avg += (1/n_avg) * (Vt_tar - Vt_src) # - (hfg-1)*( x_src))

                    # propagate direct ODE
                    zt_edit = zt_edit.to(torch.float32)

                    zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg
                    
                    zt_edit = zt_edit.to(V_delta_avg.dtype)

                if "latents_history" in debug_dict:
                    output_debug_dict["latents_history"].append(latents.clone())
                if fix_tokens is not None:
                    # Tracking attention focus
                    attentive_token, token_freq, kl_str = attention2d(dit_control, i, token_mask, **kwargs)
                    print("☀️ First stage attentive tokens:", len(attentive_token))
                    # Further Focusing on Specially Aggregated Tokens
                    attentive_token,  kl_freq = attention_self(dit_control, i, attentive_token,
                                                                topk_each_block=int(min(len(token_freq)*0.95,2*len(attentive_token))), **kwargs)
                    print("☀️ Second stage attentive tokens:", len(attentive_token))
                    print(f"☀️ topk_each block = {int(min(len(token_freq)*0.95,2*len(attentive_token)))}, token_freq len = {len(token_freq)}")
                    # Attention2d return important tokens, we want to remove the rest
                    device = latents.device
                    
                    # 確保 attentive_token 在正確設備上
                    if isinstance(attentive_token, torch.Tensor):
                        attentive_token = attentive_token.to(device)
                    else:
                        attentive_token = torch.tensor(attentive_token, device=device)
                    
                    # 分離fix_tokens和flex_tokens區域
                    num_fix_tokens = len(fix_tokens)  # 1248
                    
                    # 只從flex_tokens區域選擇要移除的tokens
                    flex_start_idx = num_fix_tokens  # 1248
                    flex_indices = torch.arange(flex_start_idx, latents.shape[1], device=device)
                    
                    # 找出flex區域中的attentive tokens (要保留的)
                    flex_attentive_mask = torch.isin(attentive_token, flex_indices)
                    flex_attentive_tokens = attentive_token[flex_attentive_mask]
                    
                    # 計算flex區域中要移除的tokens
                    flex_remove_mask = ~torch.isin(flex_indices, flex_attentive_tokens)
                    flex_tokens_to_remove = flex_indices[flex_remove_mask]
                    
                    if len(flex_tokens_to_remove) > 0:
                        # 創建最終的keep_mask
                        all_indices = torch.arange(latents.shape[1], device=device)
                        keep_mask = ~torch.isin(all_indices, flex_tokens_to_remove)
                        
                        # 同時移除所有相關張量中的tokens
                        latents = latents[:, keep_mask, :]
                        init_mask = init_mask[:, keep_mask, :]
                        latents_input = latents_input[:, keep_mask, :]
                        noise = noise[:, keep_mask, :]
                        
                        # 更新fix_tokens_mask，但fix_tokens的位置保持不變
                        new_fix_tokens_mask = torch.zeros(latents.shape[1], dtype=bool, device=device)
                        new_fix_tokens_mask[:num_fix_tokens] = True 
                        fix_tokens_mask = new_fix_tokens_mask 
                        print(f"🔥remove {len(flex_tokens_to_remove)} tokens")
                    else:
                        print("🔥No tokens removed")
                        pass
                    # Repaint replace latent with initial fixed latents
                    if i < len(timesteps) - 1:
                        noise_timestep = timesteps[i + 1]
                        init_latents = self.scheduler.scale_noise(
                            latents_input.clone(), noise, noise_timestep
                        )
                    else:
                        init_latents = latents_input.clone()

                    latents = (1 - init_mask) * init_latents + init_mask * latents
                

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    print("🔥CALL BACK")
                    latents = callback_outputs.pop("latents", latents)

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
        dit_control.unregister()
        if len(output_debug_dict) > 0:
            return zt_edit, output_debug_dict
        return zt_edit



    def grid2tokens(
        self, latents, xyz_samples, sdf_path="./output/tf3de_functions/2d_attn_sdf.ply"
    ):

        geometric_func = lambda x: self.vae.decode(latents, sampled_points=x).sample
        grid_logits = geometric_func(xyz_samples.unsqueeze(0))

        save_grid_logits_as_pointcloud(
            xyz_samples=xyz_samples,
            grid_logits=grid_logits,
            output_path=sdf_path,
            format_type="ply",
            threshold=0,  # logits<threshold
            downsample_factor=2,  # 下採樣減少點數量
            colormap="coolwarm",
        )
        return grid_logits
