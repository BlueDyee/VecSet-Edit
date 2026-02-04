# VecSetEdit

A powerful 3D mesh editing framework using VecSet representation and attention-based mechanisms for precise, localized and image-only edits.

<div align="center">
  <img src="./assets/teaser_1214.jpg" alt="VecSetEdit Teaser" width="800"/>
</div>

## 📋 Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [vecset_edit.py - 3D Mesh Editing](#vecset_editpy---3d-mesh-editing)
  - [preserving_texture_baking.py - Texture Preservation](#preserving_texture_bakingpy---texture-preservation-and-baking)
- [Demo Videos](#demo-videos)
- [Pretrained Weights](#pretrained-weights)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Installation

### Prerequisites

- Linux operating system
- NVIDIA GPU with CUDA support (CUDA 12.4)
- Python 3.11
- Conda environment manager

### Dependency Installation Order

The installation order is critical due to CUDA compatibility requirements. Follow these steps carefully:

#### Create Conda Environment

```bash
conda create -n vecset_edit python=3.11 -y
conda activate vecset_edit
```

#### Install PyTorch with CUDA 12.4

**Must install first** - All other CUDA-dependent packages rely on this version.

```bash
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124
```

#### Install nvdiffrast (Special Installation Required)

**Cannot be installed via pip** - Requires downloading pre-built wheel from GitHub.

```bash
# Download the wheel file matching your CUDA version (12.4) and Python version (3.11)
# Visit: https://github.com/NVlabs/nvdiffrast/releases
pip install setuptools wheel ninja
pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation
```

####  Install torch-cluster (CUDA-Specific)

**Requires matching PyTorch and CUDA versions.**

```bash
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-2.6.0+cu124.html
```

#### Install diso (From Source)

**Must be installed from GitHub repository.**

```bash
pip install git+https://github.com/SarahWeiii/diso.git --no-build-isolation
```

#### Install Core Dependencies

```bash
# Install all core dependencies at once
pip install -r other_requirements.txt
```

#### Install Optional Dependencies

```bash
# Blender Python API (for advanced rendering)
pip install bpy==4.0.0 mathutils==3.3.0

# Additional image enhancement
pip install gfpgan realesrgan facexlib basicsr
```

### Critical Notes

⚠️ **CUDA Version Consistency**: All CUDA-related packages must match `cu124` (CUDA 12.4). Mixing versions will cause runtime errors.

⚠️ **Installation Order Matters**: Installing PyTorch before nvdiffrast is essential. nvdiffrast compilation depends on PyTorch's CUDA configuration.

⚠️ **Wheel Files**: For `nvdiffrast` and potentially `torch-cluster`, you may need to manually download wheel files if automated downloads fail.

## Usage

### vecset_edit.py - 3D Mesh Editing

The main script for performing 3D mesh editing using VecSet representation and attention mechanisms.

#### Basic Usage

```bash
python vecset_edit.py \
    --input_dir example/chicken_racer \
    --output_dir output \
    --mesh_file model.glb \
    --render_image 2d_render.png \
    --edit_image 2d_edit.png \
    --mask_image 2d_mask.png
```

#### Parameters

**Input/Output:**
- `--input_dir`: Directory containing input mesh and images (default: `example/chicken_racer`)
- `--output_dir`: Output directory for results (default: `output`)
- `--mesh_file`: Mesh filename in input directory (default: `model.glb`)
- `--render_image`: Original rendered image filename (default: `2d_render.png`)
- `--edit_image`: Edited 2D image filename (default: `2d_edit.png`)
- `--mask_image`: Binary mask image filename (default: `2d_mask.png`)

**Camera Parameters:**
- `--azimuth`: Azimuth angle in radians (default: `0.0`)
- `--elevation`: Elevation angle in radians (default: `0.0`)

**Processing Parameters:**
- `--scale`: Scale factor for point cloud (default: `2.0`)
- `--attentive_2d`: Number of attentive 2D tokens (default: `8`)
- `--cut_off_p`: Cut-off percentage for attention (default: `0.5`)
- `--topk_percent_2d`: Top k percent of 2D attentive tokens (default: `0.2`)
- `--threshold_percent_2d`: Threshold percent for 2D attention (default: `0.1`)
- `--step_pruning`: Pruning step interval (default: `5`)
- `--edit_strength`: Editing strength (default: `0.7`)
- `--guidance_scale`: Guidance scale for generation (default: `7.5`)

#### Example

```bash
python vecset_edit.py \
    --input_dir example/chicken_racer \
    --output_dir output \
    --edit_strength 0.8 \
    --guidance_scale 7.5 \
    --scale 2.0
```

### preserving_texture_baking.py - Texture Preservation and Baking

This script handles texture repaint and baking for 3D meshes while preserving the original texture quality.

#### Basic Usage

```bash
python preserving_texture_baking.py \
    --input_mesh output/edited_mesh.glb \
    --ref_mesh output/source_model.glb \
    --texture_image output/2d_edit.png \
    --output_dir output/
```

#### Parameters

**Required:**
- `--input_mesh`: Path to the edited mesh file (default: `./output/edited_mesh.glb`)
- `--ref_mesh`: Path to the reference/source mesh file (default: `./output/source_model.glb`)
- `--texture_image`: Path to the texture image for repaint (default: `./output/2d_edit.png`)
- `--output_dir`: Output directory for textured mesh (default: `./output/`)

**Optional:**
- `--seed`: Random seed for reproducibility (default: `99999`)
- `--render_method`: Rendering method - `nvdiffrast` or `bpy` (default: `nvdiffrast`)

#### Example

```bash
python preserving_texture_baking.py \
    --input_mesh output/edited_mesh.glb \
    --ref_mesh output/source_model.glb \
    --texture_image output/2d_edit.png \
    --output_dir output/ \
    --seed 42 \
    --render_method nvdiffrast
```

#### Workflow

1. The script loads the edited mesh and reference mesh
2. Performs texture repaint using multi-view generation
3. Applies inpainting and upscaling to preserve texture quality
4. Outputs the final textured mesh as `mv_repaint_model.glb`

## Demo Videos

<div align="center">
  
https://github.com/BlueDyee/VecSetEdit/assets/demo_video.mp4

*Example: 3D mesh editing with image-guided attention mechanisms*

</div>

For best viewing experience, you can also [download the demo video](./assets/demo_video.mp4) directly.

## Pretrained Weights

### Required Checkpoints

The following pretrained weights are required and should be placed in the `checkpoints/` directory:

1. **big-lama.pt** - Large-scale inpainting model
   - Download from: [LaMa GitHub](https://github.com/advimman/lama)
   - Place in: `checkpoints/big-lama.pt`

2. **RealESRGAN_x2plus.pth** - Image super-resolution model
   - Download from: [RealESRGAN GitHub](https://github.com/xinntao/Real-ESRGAN/releases)
   - Place in: `checkpoints/RealESRGAN_x2plus.pth`

## Acknowledgments

This project integrates and builds upon the following open-source projects:

- **[TripoSG](https://github.com/VAST-AI-Research/TripoSR)**: Used for 3D geometry generation and reconstruction
- **[MV-Adapter](https://github.com/huanngzh/MV-Adapter)**: Provides multi-view image generation capabilities
- **[Hunyuan3D-2.0](https://github.com/Tencent/Hunyuan3D-2.0)**: Utilized for texture painting and 3D generation features

We are grateful to the authors and contributors of these projects for making their work available to the research community.

**Note**: This is a research project. The code is provided as-is for academic and research purposes.

