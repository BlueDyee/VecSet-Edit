import os
import torch
from PIL import Image
from vecset_edit_functions import *
import numpy as np
import argparse
import trimesh
import logging
import gc


def apply_transparent_red_overlay(image, mask, weight=0.5):
    """
    Apply transparent red overlay to visualize mask on image

    Args:
        image: Input PIL Image
        mask: Binary mask PIL Image
        weight: Blend weight for red overlay (0-1)

    Returns:
        Image with transparent red overlay on masked regions
    """
    # Create red overlay
    red_overlay = Image.new("RGB", image.size, (255, 0, 0))
    rb_img = Image.blend(image, red_overlay, weight)
    # Apply mask
    transparent_red_mask = Image.composite(rb_img, image, mask)
    return transparent_red_mask


parser = argparse.ArgumentParser(description="3D Mesh Editing with Single Input/Output")
# Input paths
parser.add_argument(
    "--input_dir",
    type=str,
    default="example/chicken_racer",
    help="Input directory containing mesh and images",
)
parser.add_argument(
    "--output_dir", type=str, default="output", help="Output directory for results"
)
# File names (optional, with defaults)
parser.add_argument(
    "--mesh_file", type=str, default="model.glb", help="Mesh filename in input_dir"
)
parser.add_argument(
    "--render_image",
    type=str,
    default="2d_render.png",
    help="Rendered image filename (optional)",
)
parser.add_argument(
    "--edit_image", type=str, default="2d_edit.png", help="Edited image filename"
)
parser.add_argument(
    "--mask_image", type=str, default="2d_mask.png", help="Mask image filename"
)
# Rotation parameters
parser.add_argument(
    "--azimuth", type=float, default=0.0, help="Azimuth angle in radians"
)
parser.add_argument(
    "--elevation", type=float, default=0.0, help="Elevation angle in radians"
)
# Processing parameters
parser.add_argument(
    "--scale", type=float, default=2.0, help="Scale factor for point cloud"
)
parser.add_argument(
    "--attentive_2d", type=int, default=8, help="Number of attentive 2D tokens"
)
parser.add_argument("--cut_off_p", type=float, default=0.5, help="Cut-off percentage")
parser.add_argument(
    "--topk_percent_2d",
    type=float,
    default=0.2,
    help="Top k percent of 2D attentive tokens",
)
parser.add_argument(
    "--threshold_percent_2d",
    type=float,
    default=0.1,
    help="Threshold percent for 2D attention",
)
parser.add_argument("--step_pruning", type=int, default=5, help="Pruning step interval")
parser.add_argument("--edit_strength", type=float, default=0.7, help="Editing strength")
parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale")
args = parser.parse_args()

# Setup
device = "cuda"
torch.cuda.empty_cache()

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(args.output_dir, "gen_log.txt"),
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Load pipelines
print("🔧 Loading pipelines...")
pipes = load_pipes()
pipe_3d = pipes["pipe_3d"]
pipe_2d = pipes["pipe_2d"] if "pipe_2d" in pipes else None
rmbg_net = pipes["rmbg_net"] if "rmbg_net" in pipes else None

# Construct file paths
mesh_path = os.path.join(args.input_dir, args.mesh_file)
edit_image_path = os.path.join(args.input_dir, args.edit_image)
mask_image_path = os.path.join(args.input_dir, args.mask_image)
render_image_path = os.path.join(args.input_dir, args.render_image)

# Validate required files
if not os.path.exists(mesh_path):
    raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
if not os.path.exists(edit_image_path):
    raise FileNotFoundError(f"Edit image not found: {edit_image_path}")
if not os.path.exists(mask_image_path):
    raise FileNotFoundError(f"Mask image not found: {mask_image_path}")

print(f"✨ Processing mesh: {mesh_path}")
logging.info(f"Processing mesh: {mesh_path}")

# Load mesh
if isinstance(mesh_path, str):
    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()
else:
    mesh = mesh_path

# Apply rotation transformations
# Rotating back to original pose
mesh.apply_transform(
    trimesh.transformations.euler_matrix(-args.elevation, -args.azimuth, 0, "sxyz")
)

# Save source model
mesh.export(os.path.join(args.output_dir, "source_model.glb"))

# Rotate to azimuth=0 for editing
mesh.apply_transform(trimesh.transformations.euler_matrix(0, args.azimuth, 0, "sxyz"))
mesh.export(os.path.join(args.output_dir, "reference_model.glb"))

# Scale mesh
mesh.vertices *= args.scale
init_latent = mesh2latent(pipe_3d, mesh)

# Load images
mask = Image.open(mask_image_path).convert("L")

if os.path.exists(render_image_path):
    rendered_image = Image.open(render_image_path)
    if rendered_image.mode == "RGBA":
        background = Image.new("RGB", rendered_image.size, (255, 255, 255))
        background.paste(rendered_image, mask=rendered_image.split()[3])
        rendered_image = background
    else:
        rendered_image = rendered_image.convert("RGB")
else:
    # If no rendered image, use edited image as reference
    print("⚠️  No rendered image found, using edited image as reference")
    rendered_image = Image.open(edit_image_path).convert("RGB")

image_edited = Image.open(edit_image_path).convert("RGB")
image_edited.save(os.path.join(args.output_dir, "2d_edit.png"))
apply_transparent_red_overlay(rendered_image, mask).save(
    os.path.join(args.output_dir, "2d_masked_input.png")
)

# Step 1: Attend 2D
print("🔍 Step 1: Attending 2D tokens...")
selected_tokens = attend_2d(
    pipe_3d=pipe_3d,
    mesh=mesh,
    latent=init_latent,
    image=rendered_image,
    mask_image=mask,
    sd_str=0.5,
    top_k_percent=args.topk_percent_2d,
    k_attentive=args.attentive_2d,
    output_dir=os.path.join(args.output_dir, "tmp/attend_2d"),
    point="surface",
)
logging.info(f"| Step 1: attend_2d selected tokens: {len(selected_tokens)} tokens")
print(f"   Selected {len(selected_tokens)} tokens")

# Step 2: Attend Self
print("🔍 Step 2: Attending self tokens...")
torch.cuda.empty_cache()
spatial_related_tokens, tokens_freq = attend_self(
    pipe_3d=pipe_3d,
    latent=init_latent,
    image=rendered_image,
    selected_tokens=selected_tokens,
    sd_str=0.2,
    guidance_scale=7.5,
    seed=50,
    cut_off_p=args.cut_off_p,
    plot_fig=True,
    output_dir=os.path.join(args.output_dir, "tmp/attend_self"),
)

if len(spatial_related_tokens) == 0:
    logging.info("No spatial related tokens found")
    print("   ⚠️  No spatial related tokens found")
elif len(spatial_related_tokens) > 2000:
    logging.info("Too many spatial related tokens found")
    print(f"   ⚠️  Too many spatial related tokens found: {len(spatial_related_tokens)}")
else:
    logging.info(
        f"| Step 2: attend_self selected tokens: {len(spatial_related_tokens)} tokens"
    )
    print(f"   Selected {len(spatial_related_tokens)} spatial tokens")

# Prepare latents
spatial_tokens = spatial_related_tokens.tolist()
spatial_latent = init_latent[:, spatial_tokens, :].clone()
mesh1 = latent2mesh(pipe_3d=pipe_3d, latents=spatial_latent)
view1 = demo_views(mesh1, view_num=4)

# Unspatial tokens
all_tokens = set(range(2048))
spatial_set = set(spatial_tokens)
unspatial_tokens = list(all_tokens - spatial_set)
unspatial_latent = init_latent[:, unspatial_tokens, :].clone()
mesh2 = latent2mesh(pipe_3d=pipe_3d, latents=unspatial_latent)
view2 = demo_views(mesh2, view_num=4)
make_image_grid([view1, view2], rows=2, cols=1).save(
    os.path.join(args.output_dir, "selected_fixed_tokens_views.png")
)

# Double the spatial latent for pruning
new_input_latent = torch.cat(
    [unspatial_latent, spatial_latent, spatial_latent.clone(), spatial_latent.clone()],
    dim=1,
)
new_fix_tokens = list(range(0, unspatial_latent.shape[1]))
print(f"🔥 New input shape: {new_input_latent.shape}")

# Step 3: Prune and edit
print("✏️  Step 3: Pruning and editing...")
torch.cuda.empty_cache()
output = prune_latent_edit_v2(
    generator=torch.Generator(device=pipe_3d.device).manual_seed(42),
    pipe_3d=pipe_3d,
    latents=new_input_latent,
    image=image_edited,
    mask_image=mask,
    fix_tokens=new_fix_tokens,
    num_inference_steps=50,
    guidance=args.guidance_scale,
    strength=args.edit_strength,
    debug_dict={"latents_history": True},
    step_pruning=args.step_pruning,
    k_attentive=8,
    top_k_percent=args.topk_percent_2d,
    threshold_percent=args.threshold_percent_2d,
)
edited_latent, debug_info = output
logging.info(
    f"| Step 3: prune_relax_latent_edit final latent size: {edited_latent.shape}"
)
print(f"   Final latent size: {edited_latent.shape}")

# Generate edited mesh
print("🎨 Generating edited mesh...")
edited_mesh = latent2mesh(pipe_3d=pipe_3d, latents=edited_latent)

# Scaling back
edited_mesh.vertices /= args.scale

# Rotating back azimuth
edited_mesh.apply_transform(
    trimesh.transformations.euler_matrix(0, -args.azimuth, 0, "sxyz")
)
edited_mesh.export(os.path.join(args.output_dir, "edited_mesh.glb"))
print("   Saved edited_mesh.glb")

# Generate views
demo_views(edited_mesh, view_num=4).save(
    os.path.join(args.output_dir, "edited_mesh_views.png")
)
print("   Saved edited_mesh_views.png")
