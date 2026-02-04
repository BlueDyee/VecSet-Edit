import os
import numpy as np
import torch
from PIL import Image
import trimesh
import random
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
from huggingface_hub import hf_hub_download, snapshot_download
import subprocess
import shutil
import cv2
import json
from diffusers.utils import make_image_grid
import sys

sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../VoxHammer")
)

# Import bpy only when using bpy render method (conditional import to avoid errors)

def erode_mask(mask_pil, kernel_size=5, iterations=1):

    mask_np = np.array(mask_pil)
    # Create erosion kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Apply erosion
    eroded = cv2.erode(mask_np, kernel, iterations=iterations)

    return Image.fromarray(eroded.astype(np.uint8))


def dilate_mask(mask_pil, kernel_size=5, iterations=1):
    """
    Dilate (expand) a binary mask

    Args:
        mask_pil: PIL Image of the mask
        kernel_size: Size of the dilation kernel
        iterations: Number of times to apply dilation

    Returns:
        PIL Image of the dilated mask
    """
    mask_np = np.array(mask_pil)
    # Create dilation kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Apply dilation
    dilated = cv2.dilate(mask_np, kernel, iterations=iterations)

    return Image.fromarray(dilated.astype(np.uint8))


DEVICE = "cuda"
DEFAULT_FACE_NUMBER = 100000
NUM_VIEWS = 6
MAX_SEED = np.iinfo(np.int32).max
TRIPOSG_REPO_URL = "https://github.com/VAST-AI-Research/TripoSG.git"
MV_ADAPTER_REPO_URL = "https://github.com/huanngzh/MV-Adapter.git"

RMBG_PRETRAINED_MODEL = "checkpoints/RMBG-1.4"
TRIPOSG_PRETRAINED_MODEL = "checkpoints/TripoSG"

TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
os.makedirs(TMP_DIR, exist_ok=True)

# TRIPOSG_CODE_DIR = "./triposg"
# if not os.path.exists(TRIPOSG_CODE_DIR):
#     os.system(f"git clone {TRIPOSG_REPO_URL} {TRIPOSG_CODE_DIR}")

# MV_ADAPTER_CODE_DIR = "./mv_adapter"
# if not os.path.exists(MV_ADAPTER_CODE_DIR):
#     os.system(f"git clone {MV_ADAPTER_REPO_URL} {MV_ADAPTER_CODE_DIR} && cd {MV_ADAPTER_CODE_DIR} && git checkout 7d37a97e9bc223cdb8fd26a76bd8dd46504c7c3d")

# mv adapter
NUM_VIEWS = 6
from mvadapter.inference_ig2mv_sdxl import prepare_pipeline, preprocess_image, remove_bg
from mvadapter.utils import get_orthogonal_camera, tensor_to_image, make_image_grid
from mvadapter.utils.mesh_utils import NVDiffRastContextWrapper, load_mesh, render

mv_adapter_pipe = prepare_pipeline(
    base_model="stabilityai/stable-diffusion-xl-base-1.0",
    vae_model="madebyollin/sdxl-vae-fp16-fix",
    unet_model=None,
    lora_model=None,
    adapter_path="huanngzh/mv-adapter",
    scheduler=None,
    num_views=NUM_VIEWS,
    device=DEVICE,
    dtype=torch.float16,
)
birefnet = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet", trust_remote_code=True
)
birefnet.to(DEVICE)
transform_image = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
remove_bg_fn = lambda x: remove_bg(x, birefnet, transform_image, DEVICE)

if not os.path.exists("checkpoints/RealESRGAN_x2plus.pth"):
    hf_hub_download(
        "dtarnow/UPscaler", filename="RealESRGAN_x2plus.pth", local_dir="checkpoints"
    )
if not os.path.exists("checkpoints/big-lama.pt"):
    subprocess.run(
        "wget -P checkpoints/ https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
        shell=True,
        check=True,
    )


def get_random_hex():
    random_bytes = os.urandom(8)
    random_hex = random_bytes.hex()
    return random_hex


@torch.no_grad()
def run_texture(image: Image, mesh_path: str, seed: int, output_dir="./"):
    height, width = 768, 768
    TMP_DIR = output_dir
    # Prepare cameras
    cameras = get_orthogonal_camera(
        elevation_deg=[0, 0, 0, 0, 89.99, -89.99],
        distance=[1.8] * NUM_VIEWS,
        left=-0.55,
        right=0.55,
        bottom=-0.55,
        top=0.55,
        azimuth_deg=[x - 90 for x in [0, 90, 180, 270, 180, 180]],
        device=DEVICE,
    )
    ctx = NVDiffRastContextWrapper(device=DEVICE, context_type="cuda")

    mesh = load_mesh(mesh_path, rescale=True, device=DEVICE)

    render_out = render(
        ctx,
        mesh,
        cameras,
        height=height,
        width=width,
        render_attr=False,
        normal_background=0.0,
    )
    control_images = (
        torch.cat(
            [
                (render_out.pos + 0.5).clamp(0, 1),
                (render_out.normal / 2 + 0.5).clamp(0, 1),
            ],
            dim=-1,
        )
        .permute(0, 3, 1, 2)
        .to(DEVICE)
    )

    image = Image.open(image).convert("RGB")
    print(f"Image size: {image.size}")
    image = remove_bg_fn(image)
    image = preprocess_image(image, height, width)

    pipe_kwargs = {}
    if seed != -1 and isinstance(seed, int):
        pipe_kwargs["generator"] = torch.Generator(device=DEVICE).manual_seed(seed)

    images = mv_adapter_pipe(
        prompt="high quality",
        height=height,
        width=width,
        num_inference_steps=15,
        guidance_scale=3.0,
        num_images_per_prompt=NUM_VIEWS,
        control_image=control_images,
        control_conditioning_scale=1.0,
        reference_image=image,
        reference_conditioning_scale=1.0,
        negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast",
        cross_attention_kwargs={"scale": 1.0},
        **pipe_kwargs,
    ).images
    torch.cuda.empty_cache()

    save_dir = os.path.join(TMP_DIR)
    mv_image_path = os.path.join(save_dir, f"mv_adapter_{NUM_VIEWS}_views.png")
    make_image_grid(images, rows=1).save(mv_image_path)
    print(f"🌊 mv image saving at {mv_image_path}")

    from mvadapter.pipelines.pipeline_texture import TexturePipeline, ModProcessConfig
    texture_pipe = TexturePipeline(
        upscaler_ckpt_path="checkpoints/RealESRGAN_x2plus.pth",
        inpaint_ckpt_path="checkpoints/big-lama.pt",
        device=DEVICE,
    )

    textured_glb_path = texture_pipe(
        mesh_path=mesh_path,
        save_dir=save_dir,
        save_name=f"texture_mesh",
        uv_unwarp=True,
        uv_size=4096,
        rgb_path=mv_image_path,
        rgb_process_config=ModProcessConfig(view_upscale=True, inpaint_mode="view"),
        camera_azimuth_deg=[x - 90 for x in [0, 90, 180, 270, 180, 180]],
        shaded_model_save_path=os.path.join(save_dir, "textured_mesh.glb"),
        debug_mode=True,
    )
    print(f"Textured mesh saved at: {textured_glb_path}")
    return textured_glb_path

@torch.no_grad()
def run_texture_repaint(
    image: Image,
    mesh_path: str,
    ref_mesh_path: str,
    seed: int,
    output_dir="./",
    render_method="nvdiffrast",
    diff_threshold=0.001,
):
    """
    Args:
        render_method: "nvdiffrast" or "bpy". Method to render multi-view images.
    """
    TMP_DIR = output_dir
    height, width = 768, 768
    # Prepare cameras
    cameras = get_orthogonal_camera(
        elevation_deg=[0, 0, 0, 0, 89.99, -89.99],
        distance=[1.8] * NUM_VIEWS,
        left=-0.55,
        right=0.55,
        bottom=-0.55,
        top=0.55,
        azimuth_deg=[x - 90 for x in [0, 90, 180, 270, 180, 180]],
        device=DEVICE,
    )
    ctx = NVDiffRastContextWrapper(device=DEVICE, context_type="cuda")

    if render_method == "bpy":
        try:
            import bpy
            from bpy_render import render_3d_model
            from mathutils import Vector

            BPY_AVAILABLE = True
        except ImportError:
            BPY_AVAILABLE = False
            print(
                "[WARNING] Blender Python API (bpy) not available. Use --render_method nvdiffrast"
            )

        # Check if bpy is available
        if not BPY_AVAILABLE:
            raise RuntimeError(
                "Blender Python API (bpy) is not available. Please run this script with Blender Python or use --render_method nvdiffrast"
            )

        # Use Blender rendering to generate mv_prior images
        print(f"🎨 Using Blender (bpy) rendering for multi-view images")
        print(f"📐 Camera alignment: Fixed orthogonal views matching nvdiffrast")

        # Render reference mesh with bpy using custom camera angles to match nvdiffrast
        bpy_render_dir = os.path.join(TMP_DIR, "bpy_render")
        os.makedirs(bpy_render_dir, exist_ok=True)

        # Define fixed camera views matching nvdiffrast orthogonal camera setup
        # azimuth_deg=[x - 90 for x in [0, 90, 180, 270, 180, 180]] = [-90, 0, 90, 180, 90, 90]
        # elevation_deg=[0, 0, 0, 0, 89.99, -89.99]
        # These correspond to: front, right, back, left, top, bottom
        azimuth_angles = [-90, 0, 90, 180, 90, 90]  # in degrees
        elevation_angles = [0, 0, 0, 0, 89.99, -89.99]  # in degrees

        # Use custom rendering to ensure alignment with nvdiffrast
        import bpy
        from mathutils import Vector

        # Initialize Blender scene (similar to bpy_render)
        from voxhammer.bpy_render import BpyRenderer

        renderer = BpyRenderer(
            resolution=height, engine="CYCLES", geo_mode=False, split_normal=False
        )
        renderer.init_scene()
        renderer.load_object(ref_mesh_path)
        renderer.init_render_settings()

        # Normalize scene
        scale, offset = renderer.normalize_scene()
        print(f"[INFO] Scene normalized with scale: {scale}, offset: {offset}")

        # Setup camera and lighting
        cam = renderer.init_camera()
        renderer.init_lighting()

        # Distance from object (matching nvdiffrast distance=1.8)
        radius = 1.8

        mv_prior_list = []
        os.makedirs(os.path.join(TMP_DIR, "mv_priors"), exist_ok=True)

        for i, (azim, elev) in enumerate(zip(azimuth_angles, elevation_angles)):
            # Convert to radians
            azim_rad = np.deg2rad(azim)
            elev_rad = np.deg2rad(elev)

            # Calculate camera position (spherical coordinates)
            # x = r * cos(elev) * cos(azim)
            # y = r * cos(elev) * sin(azim)
            # z = r * sin(elev)
            cam.location = (
                radius * np.cos(elev_rad) * np.cos(azim_rad),
                radius * np.cos(elev_rad) * np.sin(azim_rad),
                radius * np.sin(elev_rad),
            )

            # Use orthogonal camera to match nvdiffrast
            # Set camera to orthographic projection
            cam.data.type = "ORTHO"
            # orthographic scale matching [-0.55, 0.55] range -> 1.1 total width
            cam.data.ortho_scale = 1.1

            # Render
            bpy.context.scene.render.filepath = os.path.join(
                bpy_render_dir, f"{i:03d}.png"
            )
            bpy.ops.render.render(write_still=True)
            bpy.context.view_layer.update()

            # Load and save rendered image
            img_path = os.path.join(bpy_render_dir, f"{i:03d}.png")
            img = Image.open(img_path).convert("RGB")
            if img.size != (width, height):
                img = img.resize((width, height), Image.LANCZOS)
            mv_prior_list.append(img)
            img.save(os.path.join(TMP_DIR, "mv_priors", f"mv_prior_{i}.png"))

        mv_prior_grid = make_image_grid(mv_prior_list, rows=1, cols=NUM_VIEWS)
        mv_prior_grid = make_image_grid(mv_prior_list, rows=1, cols=NUM_VIEWS)

        # Cleanup Blender scene
        bpy.ops.wm.read_factory_settings(use_empty=True)

        # Load rendered images as mv_prior
        # mv_prior_list is already populated above

        # Convert to tensor for consistency with nvdiffrast path
        mv_prior = torch.stack(
            [
                torch.from_numpy(np.array(img).astype(np.float32) / 255.0)
                for img in mv_prior_list
            ]
        ).to(DEVICE)

        # Still need to render control images with nvdiffrast for geometry
        mesh = load_mesh(mesh_path, rescale=True, device=DEVICE)
        render_out = render(
            ctx,
            mesh,
            cameras,
            height=height,
            width=width,
            render_attr=False,
            normal_background=0.0,
        )
        control_images = (
            torch.cat(
                [
                    (render_out.pos + 0.5).clamp(0, 1),
                    (render_out.normal / 2 + 0.5).clamp(0, 1),
                ],
                dim=-1,
            )
            .permute(0, 3, 1, 2)
            .to(DEVICE)
        )

        # Render reference mesh geometry for control
        ref_mesh = load_mesh(ref_mesh_path, rescale=True, device=DEVICE)
        render_out_ref = render(
            ctx,
            ref_mesh,
            cameras,
            height=height,
            width=width,
            render_attr=False,
            normal_background=0.0,
        )
        prior_images = torch.cat(
            [
                (render_out_ref.pos + 0.5).clamp(0, 1),
                (render_out_ref.normal / 2 + 0.5).clamp(0, 1),
            ],
            dim=-1,
        ).permute(0, 3, 1, 2)
    else:
        # Use nvdiffrast rendering (original implementation)
        print(f"🎨 Using nvdiffrast rendering for multi-view images")
        mesh = load_mesh(mesh_path, rescale=True, device=DEVICE)

        render_out = render(
            ctx,
            mesh,
            cameras,
            height=height,
            width=width,
            render_attr=False,
            normal_background=0.0,
        )
        control_images = (
            torch.cat(
                [
                    (render_out.pos + 0.5).clamp(0, 1),
                    (render_out.normal / 2 + 0.5).clamp(0, 1),
                ],
                dim=-1,
            )
            .permute(0, 3, 1, 2)
            .to(DEVICE)
        )
        ref_mesh = load_mesh(ref_mesh_path, rescale=True, device=DEVICE)
        render_out = render(
            ctx,
            ref_mesh,
            cameras,
            height=height,
            width=width,
            render_attr=True,
            normal_background=0.0,
        )
        prior_images = torch.cat(
            [
                (render_out.pos + 0.5).clamp(0, 1),
                (render_out.normal / 2 + 0.5).clamp(0, 1),
            ],
            dim=-1,
        ).permute(0, 3, 1, 2)
        mv_prior = render_out.attr
        mv_prior_list = []
        os.makedirs(os.path.join(TMP_DIR, "mv_priors"), exist_ok=True)
        for i, mv in enumerate(mv_prior):
            PIL_image = tensor_to_image(mv.cpu())
            mv_prior_list.append(PIL_image)
            PIL_image.save(os.path.join(TMP_DIR, "mv_priors", f"mv_prior_{i}.png"))
        mv_prior_grid = make_image_grid(mv_prior_list, rows=1, cols=NUM_VIEWS)
    diff_masks = []
    diff_mask_pil_list = []
    for i in range(NUM_VIEWS):
        control_img = control_images[i : i + 1, :3, :, :]  # Use only position channels
        prior_img = prior_images[i : i + 1, :3, :, :]  # Use only position channels
        diff = torch.abs(control_img - prior_img)
        diff_gray = diff.mean(dim=1, keepdim=True)  # Convert to grayscale
        diff_mask = (diff_gray > diff_threshold).float()  # Threshold to create binary mask
        diff_image = Image.fromarray(
            diff_mask[0][0].cpu().numpy().astype(np.uint8) * 255
        )
        diff_image.save(os.path.join(TMP_DIR, "mv_priors", f"diff_mask_{i}.png"))

        # Apply erosion to remove small edges
        diff_mask_pil = erode_mask(diff_image, kernel_size=7, iterations=2)
        diff_mask_pil = dilate_mask(diff_mask_pil, kernel_size=5, iterations=13)
        diff_mask_pil.save(
            os.path.join(TMP_DIR, "mv_priors", f"diff_mask_eroded_{i}.png")
        )
        diff_mask = (
            torch.from_numpy(np.array(diff_mask_pil).astype(np.float32) / 255.0)
            .unsqueeze(0)
            .to(DEVICE)
        )
        diff_masks.append(diff_mask)
        diff_mask_pil_list.append(diff_mask_pil)
    diff_grid = make_image_grid(diff_mask_pil_list, rows=1, cols=NUM_VIEWS)
    diff_masks = torch.stack(diff_masks, dim=0)

    # Reference image processing
    image = Image.open(image).convert("RGB")
    image = remove_bg_fn(image)
    image = preprocess_image(image, height, width)

    pipe_kwargs = {}
    if seed != -1 and isinstance(seed, int):
        pipe_kwargs["generator"] = torch.Generator(device=DEVICE).manual_seed(seed)

    images = mv_adapter_pipe.sdedit(
        mv_image=mv_prior_list,
        repaint_mask=diff_masks,
        strength=1,
        prompt="high quality",
        height=height,
        width=width,
        num_inference_steps=15,
        guidance_scale=3.0,
        num_images_per_prompt=NUM_VIEWS,
        control_image=control_images,
        control_conditioning_scale=1.0,
        reference_image=image,
        reference_conditioning_scale=1.0,
        negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast",
        cross_attention_kwargs={"scale": 1.0},
        **pipe_kwargs,
    ).images

    output_grid = make_image_grid(images, rows=1, cols=NUM_VIEWS)
    combined_grid = make_image_grid(
        [mv_prior_grid, diff_grid, output_grid], rows=3, cols=1
    )
    combined_grid.save(
        os.path.join(TMP_DIR, f"mv_adapter_repaint_{NUM_VIEWS}_views.png")
    )
    torch.cuda.empty_cache()

    save_dir = os.path.join(TMP_DIR)
    mv_image_path = os.path.join(save_dir, f"mv_adapter_{NUM_VIEWS}_views.png")
    make_image_grid(images, rows=1).save(mv_image_path)
    print(f"🌊 mv image saving at {mv_image_path}")

    from mvadapter.pipelines.pipeline_texture import TexturePipeline, ModProcessConfig

    texture_pipe = TexturePipeline(
        upscaler_ckpt_path="checkpoints/RealESRGAN_x2plus.pth",
        inpaint_ckpt_path="checkpoints/big-lama.pt",
        device=DEVICE,
    )

    textured_glb_path = texture_pipe(
        mesh_path=mesh_path,
        save_dir=save_dir,
        save_name=f"texture_mesh",
        uv_unwarp=True,
        uv_size=4096,
        rgb_path=mv_image_path,
        rgb_process_config=ModProcessConfig(view_upscale=True, inpaint_mode="view"),
        camera_azimuth_deg=[x - 90 for x in [0, 90, 180, 270, 180, 180]],
        shaded_model_save_path=os.path.join(save_dir, "mv_repaint_model.glb"),
        debug_mode=True,
    )
    print(f"Textured mesh saved at: {textured_glb_path}")
    return textured_glb_path


import argparse
from natsort import natsorted
dir_name="backpack_1"
parser = argparse.ArgumentParser()
parser.add_argument("--input_mesh", type=str, default=f"./output/edited_mesh.glb")
parser.add_argument("--ref_mesh", type=str, default=f"./output/source_model.glb")
parser.add_argument("--texture_image", type=str, default=f"./output/2d_edit.png")
parser.add_argument("--output_dir", type=str, default=f"./output/")
parser.add_argument("--seed", type=int, default=99999)
parser.add_argument(
    "--render_method",
    type=str,
    default="nvdiffrast",
    choices=["nvdiffrast", "bpy"],
    help="Rendering method for multi-view images: nvdiffrast or bpy",
)
args = parser.parse_args()


image_path = args.texture_image
mesh_path = args.input_mesh

textured_glb_path = run_texture_repaint(
    image_path,
    mesh_path,
    seed=args.seed,
    output_dir=args.output_dir,
    ref_mesh_path=args.ref_mesh,
    diff_threshold=0.005,
)
torch.cuda.empty_cache()
# demo_views(textured_glb_path, view_num=4).save(os.path.join(output_prompt_dir, 'textured_mesh_views.png'))
print("Texture generation completed successfully.")
