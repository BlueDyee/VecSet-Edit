"""
VecSet Edit Functions
Core functions for 3D mesh editing using VecSet representation.
"""

import trimesh
import numpy as np
import imageio
from PIL import Image
import nvdiffrast.torch as dr
import torch
from tqdm import tqdm
import os
import argparse
import cv2
from diffusers.utils import make_image_grid
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
from triposg.scripts.image_process import prepare_image, prepare_image_no_resize
from triposg.pipelines.pipeline_triposg import retrieve_latents

from custom_control.visualization_control import VAEvisualizationControl
from custom_control.dit_control import TripoSGDiTControl
from custom_control.hull import *


def load_pipes(
    pipe_3d="pretrained_weights/TripoSG",
    rmbg_net="pretrained_weights/RMBG-1.4",
    pipe_2d=None,
    gpu_id=0,
    device="cuda",
):
    """
    Load required pipelines for 3D mesh processing.

    Args:
        pipe_3d: Path to TripoSG pretrained weights
        rmbg_net: Path to background removal model weights
        pipe_2d: Path to 2D image editing pipeline (optional)
        gpu_id: GPU device ID
        device: Device to load models on

    Returns:
        pipe_dict: Dictionary containing loaded pipelines
    """
    pipe_dict = {}
    if pipe_3d is not None:
        from triposg.pipelines.pipeline_triposg import TripoSGPipeline

        pipe_dict["pipe_3d"] = TripoSGPipeline.from_pretrained(pipe_3d).to(device)
    if rmbg_net is not None:
        from triposg.scripts.briarmbg import BriaRMBG

        rmbg_net = BriaRMBG.from_pretrained(rmbg_net).to(device)
        rmbg_net.eval()
        pipe_dict["rmbg_net"] = rmbg_net
    if pipe_2d is not None:
        from diffusers import FluxFillPipeline

        pipe_dict["pipe_2d"] = FluxFillPipeline.from_pretrained(
            pipe_2d,
            torch_dtype=torch.bfloat16,
        ).to(device)
    return pipe_dict


def rotate(mesh_path, vertical_angle, horizontal_angle, output_mesh_path=None):
    """
    Rotate mesh model

    Args:
        mesh_path: Input mesh file path or loaded mesh object
        vertical_angle: Vertical rotation angle in radians
        horizontal_angle: Horizontal rotation angle in radians
        output_mesh_path: Output path for rotated mesh (if None, won't save)

    Returns:
        rotated_mesh: Rotated mesh object
    """
    # Load or process mesh
    if isinstance(mesh_path, str):
        mesh = trimesh.load(mesh_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.to_geometry()
    else:
        # Assume input is a mesh object
        mesh = mesh_path

    # Create rotation matrix
    # First horizontal rotation (around Y-axis), then vertical rotation (around X-axis)
    rotation_y = trimesh.transformations.rotation_matrix(horizontal_angle, [0, 1, 0])
    rotation_x = trimesh.transformations.rotation_matrix(vertical_angle, [1, 0, 0])
    rotation_matrix = np.dot(rotation_x, rotation_y)

    # Apply rotation to mesh
    rotated_mesh = mesh.copy()
    rotated_mesh.apply_transform(rotation_matrix)

    # Save rotated mesh if output path is specified
    if output_mesh_path:
        rotated_mesh.export(output_mesh_path)
        print(f"Rotated mesh saved to: {output_mesh_path}")

    print(
        f"Rotation angles - Vertical: {np.degrees(vertical_angle):.1f}°, Horizontal: {np.degrees(horizontal_angle):.1f}°"
    )

    return rotated_mesh


def render(
    mesh,
    resolution=1024,
    r=2.0,
    fov=40,
    bbox_size=None,
    bbox_center=None,
    return_details=False,
):
    """
    Render mesh object to image

    Args:
        mesh: Trimesh mesh object or file path
        resolution: Rendering resolution
        r: Camera distance multiplier
        fov: Field of view angle in degrees
        bbox_size: Bounding box size (optional)
        bbox_center: Bounding box center (optional)
        return_details: Whether to return additional rendering details

    Returns:
        rendered_image: Rendered PIL Image
        details: Additional rendering information (if return_details=True)
    """
    if type(mesh) is str:
        # Load mesh from file path
        mesh = trimesh.load(mesh)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.to_geometry()

    # Prepare mesh data
    vertices = mesh.vertices
    vertices_tensor = torch.tensor(vertices, dtype=torch.float32).cuda().contiguous()
    faces_tensor = torch.tensor(mesh.faces, dtype=torch.int32).cuda().contiguous()

    # Handle texture and UV coordinates
    has_texture = (
        hasattr(mesh.visual, "material")
        and mesh.visual.material is not None
        and hasattr(mesh.visual, "uv")
        and mesh.visual.uv is not None
        and hasattr(mesh.visual.material, "baseColorTexture")
    )

    if has_texture:
        # Process UV coordinates
        uv_coords_raw = mesh.visual.uv
        uv_coords_processed = uv_coords_raw.copy()

        # Ensure UV coordinates are in [0,1] range
        if uv_coords_processed.min() < 0 or uv_coords_processed.max() > 1:
            uv_coords_processed[:, 0] = (
                uv_coords_processed[:, 0] - uv_coords_processed[:, 0].min()
            ) / (uv_coords_processed[:, 0].max() - uv_coords_processed[:, 0].min())
            uv_coords_processed[:, 1] = (
                uv_coords_processed[:, 1] - uv_coords_processed[:, 1].min()
            ) / (uv_coords_processed[:, 1].max() - uv_coords_processed[:, 1].min())

        # Flip V coordinate
        uv_coords_processed[:, 1] = 1.0 - uv_coords_processed[:, 1]
        uv_coords = (
            torch.tensor(uv_coords_processed, dtype=torch.float32).cuda().contiguous()
        )

        # Process texture image
        texture_image = mesh.visual.material.baseColorTexture
        if hasattr(texture_image, "size"):  # PIL Image
            texture_array = np.array(texture_image)
        else:
            texture_array = texture_image

        # Convert texture to torch tensor
        if len(texture_array.shape) == 3:
            if texture_array.shape[2] == 4:  # RGBA
                texture_array = texture_array[:, :, :3]  # Take RGB only
            elif texture_array.shape[2] == 1:  # Grayscale
                texture_array = np.repeat(texture_array, 3, axis=2)

            texture_tensor = (
                torch.tensor(texture_array, dtype=torch.float32).cuda() / 255.0
            )
            texture_tensor = texture_tensor.unsqueeze(0).contiguous()
        else:
            texture_tensor = (
                torch.tensor(texture_array, dtype=torch.float32).cuda() / 255.0
            )
            texture_tensor = (
                texture_tensor.unsqueeze(-1).repeat(1, 1, 3).unsqueeze(0).contiguous()
            )

    else:
        # No texture, use vertex normals for colors
        if hasattr(mesh, "vertex_normals") and mesh.vertex_normals is not None:
            vertex_normals = mesh.vertex_normals
        else:
            # Compute vertex normals
            face_normals = mesh.face_normals
            vertex_normals = np.zeros_like(vertices)

            for i, face in enumerate(mesh.faces):
                for vertex_idx in face:
                    vertex_normals[vertex_idx] += face_normals[i]

            norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
            norms[norms == 0] = 1
            vertex_normals = vertex_normals / norms

        # Convert normals to colors
        vertex_colors = (vertex_normals + 1.0) / 2.0
        vertex_colors = (
            torch.tensor(vertex_colors, dtype=torch.float32).cuda().contiguous()
        )

        uv_coords = (
            torch.zeros((len(vertices), 2), dtype=torch.float32).cuda().contiguous()
        )
        texture_tensor = (
            torch.ones((1, 1, 1, 3), dtype=torch.float32).cuda().contiguous()
        )

    # Create rendering context
    try:
        glctx = dr.RasterizeCudaContext()
    except:
        glctx = dr.RasterizeGLContext()

    # Compute scene bounding box
    bbox_min = vertices_tensor.min(dim=0)[0]
    bbox_max = vertices_tensor.max(dim=0)[0]
    if bbox_center is None:
        bbox_center = (bbox_min + bbox_max) / 2
    if bbox_size is None:
        bbox_size = (bbox_max - bbox_min).max()

    # Set camera position (front view)
    yaw = 0.0
    pitch = 0.25  # Slight top-down view

    orig = (
        torch.tensor(
            [
                np.sin(yaw) * np.cos(pitch),
                np.sin(pitch),
                np.cos(yaw) * np.cos(pitch),
            ],
            dtype=torch.float32,
        ).cuda()
        * r
    )

    cam_pos = bbox_center + orig * bbox_size
    target = bbox_center
    up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32).cuda()

    # Compute view matrix
    forward = torch.nn.functional.normalize(target - cam_pos, dim=0)
    right = torch.nn.functional.normalize(torch.cross(forward, up, dim=0), dim=0)
    up_corrected = torch.cross(right, forward, dim=0)

    view_matrix = torch.eye(4, dtype=torch.float32).cuda()
    view_matrix[0, :3] = right
    view_matrix[1, :3] = up_corrected
    view_matrix[2, :3] = -forward
    view_matrix[0, 3] = -torch.dot(right, cam_pos)
    view_matrix[1, 3] = -torch.dot(up_corrected, cam_pos)
    view_matrix[2, 3] = torch.dot(forward, cam_pos)

    # Projection matrix
    fov_rad = fov * np.pi / 180.0
    aspect = 1.0
    near = bbox_size * 0.1
    far = bbox_size * 10.0

    f = 1.0 / np.tan(fov_rad / 2.0)
    proj_matrix = torch.zeros((4, 4), dtype=torch.float32).cuda()
    proj_matrix[0, 0] = f / aspect
    proj_matrix[1, 1] = f
    proj_matrix[2, 2] = (far + near) / (near - far)
    proj_matrix[2, 3] = (2 * far * near) / (near - far)
    proj_matrix[3, 2] = -1.0

    # MVP matrix
    mvp = proj_matrix @ view_matrix

    # Transform vertices
    vertices_homo = torch.cat(
        [
            vertices_tensor,
            torch.ones((len(vertices_tensor), 1), device=vertices_tensor.device),
        ],
        dim=1,
    )
    vertices_clip = (mvp @ vertices_homo.T).T.contiguous()

    # Rasterize
    rast, _ = dr.rasterize(
        glctx,
        vertices_clip.unsqueeze(0),
        faces_tensor,
        resolution=(resolution, resolution),
    )

    if has_texture:
        # Use texture
        uv_interp, _ = dr.interpolate(uv_coords.unsqueeze(0), rast, faces_tensor)
        color = dr.texture(texture_tensor, uv_interp, filter_mode="linear")
    else:
        # Use vertex normal colors
        color, _ = dr.interpolate(vertex_colors.unsqueeze(0), rast, faces_tensor)

    # Convert to image
    color = color[0].cpu().numpy()
    mask = rast[0, :, :, 3].cpu().numpy() > 0

    # Create black background
    final_image = np.zeros((resolution, resolution, 3), dtype=np.float32)
    final_image[mask] = color[mask]

    # Convert to uint8
    final_image = np.clip(final_image * 255, 0, 255).astype(np.uint8)
    # Flip Y-axis
    final_image = np.flipud(final_image)

    if return_details:
        return Image.fromarray(final_image), {
            "bbox_center": bbox_center,
            "bbox_size": bbox_size,
        }
    return Image.fromarray(final_image)


def demo_views(mesh, view_num=4):
    """
    Generate demo views of the mesh from multiple angles

    Args:
        mesh: Trimesh mesh object
        view_num: Number of views to generate

    Returns:
        Image grid showing multiple views
    """
    # Evenly distribute views around the mesh
    angles = np.linspace(0, 2 * np.pi, view_num, endpoint=False)
    views = []
    for angle in angles:
        rotated_mesh = rotate(mesh, vertical_angle=0, horizontal_angle=angle)
        rendered_image = render(rotated_mesh)
        views.append(rendered_image)
    return make_image_grid(views, rows=1, cols=view_num)


def attend_2d(
    pipe_3d,
    mesh,  # For point sampling
    latent,
    image,
    mask_image,
    k_attentive=5,  # Top k attentive blocks
    sd_str=0.5,
    output_dir="./output/vecset_edit_functions",
    point="surface",
    top_k_percent=0.2,
    threshold_percent=0.2,
    save_attn_cloud=False,
):
    """
    Attend to the 2D image space using 3D mesh attention mechanism

    Args:
        pipe_3d: TripoSG pipeline
        mesh: Trimesh mesh object for point sampling
        latent: Input latent representation
        image: Target image for editing
        mask_image: Editing mask image
        k_attentive: Number of top attentive blocks to select
        sd_str: Diffusion strength
        output_dir: Output directory for visualizations
        point: Point sampling method ("surface" or "grid")
        top_k_percent: Percentage of top tokens to keep
        threshold_percent: Threshold percentage for filtering
        save_attn_cloud: Whether to save attention point cloud

    Returns:
        filtered_idx: Indices of selected 3D tokens
    """
    # Sample points from mesh
    if point == "surface":
        surface_points, face_indices = trimesh.sample.sample_surface(
            mesh, 50000, seed=42
        )
        # Add random noise between -0.01 and 0.01
        noise = np.random.uniform(-0.01, 0.01, surface_points.shape)
        surface_points_noisy = surface_points + noise
        p_tensor = (
            torch.tensor(surface_points_noisy, dtype=torch.float32).unsqueeze(0).cuda()
        )
    elif point == "grid":
        bbox = mesh.bounding_box_oriented
        corners = bbox.vertices
        # Generate grid points within the bounding box
        num_points_per_axis = 128
        x = np.linspace(corners[0][0], corners[6][0], num_points_per_axis)
        y = np.linspace(corners[0][1], corners[6][1], num_points_per_axis)
        z = np.linspace(corners[0][2], corners[6][2], num_points_per_axis)
        grid_x, grid_y, grid_z = np.meshgrid(x, y, z)
        grid_points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T
        p_tensor = torch.tensor(grid_points, dtype=torch.float32).unsqueeze(0).cuda()
    else:
        raise NotImplementedError("Point sampling method not implemented")

    input_latents = latent
    # Initialize DiT control
    dit_control = TripoSGDiTControl(store_value=True)  # Set False for memory saving
    # Register the transformer with DiT control
    dit_control.register(pipe_3d.transformer)
    latents_e = pipe_3d.latent_edit(
        latents=input_latents,
        image=image,
        generator=torch.Generator(device=pipe_3d.device).manual_seed(50),
        num_inference_steps=50,
        guidance_scale=15,
        strength=sd_str,
        debug_dict={"latents": True},
        no_output_mesh=True,
    )

    # Track token selection frequency
    token_selected_freq = torch.zeros(
        latents_e.shape[1], dtype=torch.float32, device=pipe_3d.device
    )

    for step in tqdm(range(int(50 * sd_str)), total=int(50 * sd_str)):
        kl_strength = []
        for block_idx in range(20):
            control = dit_control.idx2control[block_idx]
            attn_prob = control.prob_list[step][1:, 1:]  # Remove temb token, cls token
            attn_value = control.value_list[step][1:]  # Remove temb token, cls token
            N3d, N2d = attn_prob.shape

            # Load 2D mask and create token mask
            mask = mask_image.resize((int(N2d**0.5), int(N2d**0.5)))
            mask = np.array(mask) / 255.0  # Normalize to [0, 1]
            mask = np.where(mask > 0.5, 1, 0)  # Binarize the mask
            token_mask = mask.flatten().astype(bool)
            selected_attn_prob = attn_prob[:, token_mask]

            # Compute KL divergence
            kl_attn_dist = attn_prob.mean(dim=0)
            kl_attn_dist_ = kl_attn_dist.expand_as(attn_prob)
            eps = 1e-7
            kl_per_sample = (
                attn_prob
                * (torch.log(attn_prob + eps) - torch.log(kl_attn_dist_ + eps))
            ).sum(dim=1)
            kl_strength.append(kl_per_sample.mean().item())

        # Select top k attentive blocks for further processing
        attentive_idx_list = torch.argsort(torch.tensor(kl_strength), descending=True)[
            :k_attentive
        ]
        attentive_idx_list = attentive_idx_list.tolist()

        for block_idx in attentive_idx_list:
            save_dir = os.path.join(output_dir, f"step_{step}", f"block_{block_idx}")
            control = dit_control.idx2control[block_idx]
            attn_prob = control.prob_list[step][1:, 1:]  # Remove temb token, cls token
            selected_attn_prob = attn_prob[:, token_mask]

            sel_kl_dist = selected_attn_prob.mean(dim=0)
            sel_kl_dist_ = sel_kl_dist.expand_as(selected_attn_prob)
            sel_kl_sample = (
                selected_attn_prob
                * (torch.log(selected_attn_prob + eps) - torch.log(sel_kl_dist_ + eps))
            ).sum(dim=1)
            token_selected_freq += sel_kl_sample

    # Filter tokens by threshold
    sort_val, sort_idx = torch.topk(
        torch.tensor(token_selected_freq), k=token_selected_freq.shape[0]
    )
    n_highest = max(1, int(len(sort_val) * top_k_percent))
    threshold = sort_val[:n_highest].mean() * threshold_percent
    mask = sort_val > threshold
    filtered_idx = sort_idx[mask]
    print(f"Final Selected 3D Tokens (Drop count < {threshold}):", filtered_idx.shape)

    # Visualize selected tokens
    control = VAEvisualizationControl(
        vis_token_list=filtered_idx,
        vis_attn_dim_list=[i for i in range(0, 1, 1)],
        save_dir=output_dir,
        save_attn_cloud=True,
        attn_cloud_name="attn_cloud_kl_avg_kl_acc",
    )
    control.register(pipe_3d.vae)
    sdf_func = lambda x: pipe_3d.vae.decode(latents_e, sampled_points=x).sample
    sdf_logits = sdf_func(p_tensor)
    control.unregister()
    dit_control.unregister()

    # Plot and save token strength distribution
    token_indices = np.arange(len(sort_val))
    plt.figure(figsize=(12, 8))
    plt.bar(
        token_indices,
        sort_val.cpu().numpy() - sort_val.min().cpu().item(),
        color="skyblue",
        width=1.0,
    )
    plt.title("VecSet Tokens Accumulated Strength (sorted)", fontsize=26)
    plt.axhline(
        y=threshold.cpu().item() - sort_val.min().cpu().item(),
        color="green",
        linestyle="--",
        linewidth=3,
        label=f"threshold={threshold.cpu().item()}",
    )
    plt.xlabel(f"Tokens ({len(filtered_idx)} selected)", fontsize=20)
    plt.ylabel("Strength", fontsize=20)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "selected_count.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    return filtered_idx


def attend_self(
    pipe_3d,
    latent,
    image,
    selected_tokens,
    sd_str=0.5,
    guidance_scale=7.5,
    topk_each_block=None,
    seed=50,
    plot_fig=False,
    output_dir="./output/spatial_aggregation_self_attention",
    close_fig=False,
    cut_off_p=0.1,
):
    """
    Analyze spatial aggregation using self-attention mechanism

    Args:
        pipe_3d: TripoSG pipeline
        latent: Initial latent representation
        image: Input image
        selected_tokens: Pre-selected token indices (torch.Tensor or list)
        sd_str: Diffusion strength
        guidance_scale: Guidance scale for diffusion
        topk_each_block: Top-k tokens per block (optional)
        seed: Random seed
        plot_fig: Whether to plot visualization
        output_dir: Output directory
        close_fig: Whether to close figure after saving
        cut_off_p: Cut-off percentage for filtering

    Returns:
        final_selected_tokens: Spatially aggregated token indices
        tokens_freq: Selection frequency for each token
    """
    # Ensure selected_tokens is in correct format
    if isinstance(selected_tokens, torch.Tensor):
        selected_tokens = selected_tokens.cpu().numpy().tolist()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize DiT controller with self-attention tracking
    dit_control = TripoSGDiTControl(
        store_value=True, self_attention=True, self_attention_device="GPU"
    )
    dit_control.register(pipe_3d.transformer)

    # Perform latent editing
    latents_e = pipe_3d.latent_edit(
        latents=latent,
        image=image,
        generator=torch.Generator(device=pipe_3d.device).manual_seed(seed),
        num_inference_steps=50,
        guidance_scale=guidance_scale,
        strength=sd_str,
        debug_dict={"latents": True},
        no_output_mesh=True,
    )

    # Initialize frequency counter
    if latent.dim() == 3:
        tokens_freq = torch.zeros(latent.shape[1]).to("cuda")
    elif latent.dim() == 2:
        tokens_freq = torch.zeros(latent.shape[0]).to("cuda")

    # Process each diffusion step
    for step in tqdm(range(int(50 * sd_str)), total=int(50 * sd_str)):
        for block_idx in range(20):
            save_dir = os.path.join(output_dir, f"step_{step}", f"block_{block_idx}")
            os.makedirs(save_dir, exist_ok=True)

            control = dit_control.idx2control[block_idx]
            attn_prob = control.prob_list_self[step][
                1:, 1:
            ]  # Remove temb token, cls token
            N3d, M3d = attn_prob.shape

            # Compute strength for selected tokens
            k_selected_strength = attn_prob[:, selected_tokens].mean(dim=-1)
            if topk_each_block is None:
                tokens_freq += k_selected_strength
            else:
                topk_values, topk_indices = torch.topk(
                    k_selected_strength, k=topk_each_block, dim=0
                )
                tokens_freq[topk_indices] += 1

    # Clean up controller
    dit_control.unregister()

    # Filter tokens based on frequency threshold
    selected_freq_values = tokens_freq[selected_tokens]
    n_lowest = max(1, int(len(selected_freq_values) * cut_off_p))
    lowest_values, _ = torch.topk(selected_freq_values, k=n_lowest, largest=False)
    threshold = lowest_values.mean()
    final_selected_tokens = torch.nonzero(tokens_freq >= threshold).squeeze(1)

    # Plot and save results if requested
    if plot_fig:
        fig, ax = plt.subplots(figsize=(12, 8))
        sorted_freq, sorted_indices = torch.sort(tokens_freq, descending=True)
        colors = ["lightblue"] * len(sorted_freq)

        # Highlight selected tokens with different color
        for i, token_idx in enumerate(sorted_indices):
            if token_idx.item() in selected_tokens:
                colors[i] = "red"

        ax.bar(
            np.arange(len(sorted_freq)),
            sorted_freq.cpu().numpy(),
            color=colors,
            width=1.0,
        )
        # Draw threshold line
        threshold_val = float(threshold)
        ax.axhline(
            y=threshold_val,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"threshold={threshold_val:.3f}",
        )
        ax.set_title("Token Selection Frequency across Blocks and Steps", fontsize=14)
        ax.set_xlabel("Token Index (Sorted by Frequency)", fontsize=12)
        ax.set_ylabel("Selection Frequency", fontsize=12)

        # Add legend
        import matplotlib.patches as mpatches

        normal_patch = mpatches.Patch(color="lightblue", label="Other Tokens")
        selected_patch = mpatches.Patch(color="red", label="Selected Tokens")
        ax.legend(handles=[normal_patch, selected_patch])

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "token_selection_frequency.png"),
            dpi=300,
            bbox_inches="tight",
        )
        if close_fig:
            plt.close()

    return final_selected_tokens, tokens_freq


def prune_latent_edit_v2(
    pipe_3d,
    latents,
    image,
    mask_image,
    fix_tokens,
    num_inference_steps=50,
    guidance=7.5,
    strength=0.7,
    generator=None,
    **kwargs,
):
    """
    Perform pruning-based latent editing (version 2)

    Args:
        pipe_3d: TripoSG pipeline
        latents: Input latent representation
        image: Target image for editing
        mask_image: Editing mask
        fix_tokens: Tokens to keep fixed during editing
        num_inference_steps: Number of diffusion steps
        guidance: Guidance scale
        strength: Editing strength
        generator: Random generator for reproducibility
        **kwargs: Additional arguments

    Returns:
        Edited latent representation and debug information
    """
    output = pipe_3d.prune_latent_edit_v2(
        latents=latents,
        image=image,
        mask_image=mask_image,
        fix_tokens=fix_tokens,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance,
        strength=strength,
        generator=generator,
        **kwargs,
    )
    return output


def latent2mesh(pipe_3d, latents, hierarchical_octree_depth=8, dense_octree_depth=8):
    """
    Convert latent representation to 3D mesh

    Args:
        pipe_3d: TripoSG pipeline
        latents: Latent representation
        hierarchical_octree_depth: Depth for hierarchical octree
        dense_octree_depth: Depth for dense octree

    Returns:
        Converted 3D mesh
    """
    print("Converting latents to mesh...")
    meshes = pipe_3d.latent2mesh(
        latents,
        hierarchical_octree_depth=hierarchical_octree_depth,
        dense_octree_depth=dense_octree_depth,
    )
    return meshes[0]  # Return the first mesh


def mesh2latent(pipe_3d, mesh, return_sdf=False, sample=True, device="cuda", **kwargs):
    """
    Convert 3D mesh to latent representation

    Args:
        pipe_3d: TripoSG pipeline
        mesh: Input 3D mesh (trimesh object or file path)
        return_sdf: Whether to return SDF function
        sample: Whether to sample from latent distribution
        device: Device to use
        **kwargs: Additional arguments

    Returns:
        latents: Latent representation
        geometric_func: SDF function (if return_sdf=True)
    """
    if type(mesh) == str:
        mesh = trimesh.load(mesh)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.to_geometry()

    if sample:
        latents = retrieve_latents(
            pipe_3d.mesh2latents(mesh, dtype=pipe_3d.vae.dtype, **kwargs)
        ).to(device)
        latents = latents.detach().requires_grad_(False)
    else:
        latents = pipe_3d.mesh2latents(
            mesh, dtype=pipe_3d.vae.dtype, **kwargs
        ).latent_dist

    if return_sdf:
        geometric_func = lambda x: pipe_3d.vae.decode(latents, sampled_points=x).sample
        return latents, geometric_func
    return latents
