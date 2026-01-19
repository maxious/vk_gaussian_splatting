# SAM 3D Body + Hybrid Gaussian Splatting Integration

This directory contains integration of Meta's **SAM 3D Body** human mesh recovery
into vk_gaussian_splatting offline pipeline.

## Overview

The hybrid approach combines:

1. **SAM 3D Body**: State-of-the-art human mesh recovery with:
   - Structured parametric meshes (Momentum Human Rig)
   - Accurate pose estimation (body, hands, feet)
   - Occlusion handling (parametric model infers occluded parts)
   - **Limitation**: Untextured meshes

2. **Depth Models** (DA3/MoGe/SHARP): Textured depth for:
   - Background objects
   - Non-human content
   - Full scene structure

3. **Texture Sampling**: Projects human mesh vertices back to 2D to sample colors from original image

## Installation

**SAM 3D Body is NOT available on PyPI and is vendored locally.**

```bash
cd python
# SAM 3D Body code is already vendored at python/sam_3d_body_git/
# No additional installation required!

# Install other dependencies
uv sync --extra cuda --extra offline
```

### Processing Your Test Data

You have:
- Images: `/media/maxious/Data/SIGA2025VVC-Dataset/compression/test/006_1_seq1/images.tar/images/03/`
- Masks: `/media/maxious/Data/SIGA2025VVC-Dataset/compression/test/006_1_seq1/masks.tar/masks/03/`

**Option 1: Use existing `images` command with masks**

```bash
cd /home/maxious/vk_gaussian_splatting/python
uv run --extra cuda python -m offline.cli images \
  --input "/media/maxious/Data/SIGA2025VVC-Dataset/compression/test/006_1_seq1/images.tar/images/03/" \
  --output ./test_output/ \
  --masks-dir "/media/maxious/Data/SIGA2025VVC-Dataset/compression/test/006_1_seq1/masks.tar/masks/03/" \
  --model "depth-anything/DA3-GIANT" \
  --mode frames
```

This will:
1. Load all images from the input directory
2. Apply corresponding masks (e.g., `000000.jpg` uses `000000.png` mask)
3. Process with depth model (DA3/MoGe/SHARP)
4. Export Gaussian PLY files

**Option 2: Hybrid mode (SAM 3D Body for humans + depth for background)**

```bash
cd /home/maxious/vk_gaussian_splatting/python
uv run --extra cuda python -m offline.cli export \
  --input "/media/maxious/Data/SIGA2025VVC-Dataset/compression/test/006_1_seq1/images.tar/images/03/" \
  --output ./test_hybrid_output/ \
  --model "hybrid:facebook/sam-3d-body-vith+depth-anything/DA3-GIANT" \
  --mode frames
```

This will:
1. Detect humans with SAM 3D Body and sample structured points
2. Estimate depth for full scene with DA3
3. Remove overlapping depth Gaussians in human regions
4. Combine human Gaussians (structured) + background Gaussians (textured)
5. Export combined Gaussian PLY files

## Usage Examples

### Processing Human Scenes

#### 1. SAM 3D Body Only (Human-only scenes)

```bash
uv run --extra cuda --extra sam3dbody python -m offline.cli images \
  --input ./images_folder/ \
  --output ./human_plys/ \
  --model "sam3dbody:facebook/sam-3d-body-vith" \
  --mode frames
```

#### 2. Hybrid: Humans + Background (Mixed scenes)

```bash
uv run --extra cuda --extra sam3dbody python -m offline.cli images \
  --input ./images_folder/ \
  --output ./hybrid_plys/ \
  --model "hybrid:facebook/sam-3d-body-vith+depth-anything/DA3-GIANT" \
  --mode frames
```

### Model Configuration Format

#### SAM 3D Body

```
--model "sam3dbody:<model_id>"
```

- `facebook/sam-3d-body-vith` (default, ViT-H backbone, 631M params)
- `facebook/sam-3d-body-dinov3` (DINOv3-H+ backbone, 840M params)
- Custom path: `sam3dbody:/path/to/model.ckpt`

#### Hybrid

```
--model "hybrid:<human_model>+<depth_model>"
```

- **Human model** (first part):
  - `facebook/sam-3d-body-vith` (default)
  - `facebook/sam-3d-body-dinov3`
  - Custom path: `/path/to/human.ckpt`

- **Depth model** (second part):
  - `depth-anything/DA3-GIANT` (default DA3)
  - `Ruicheng/moge-2-vitl-normal` (MoGe2 with normals)
  - `sharp` (Apple SHARP, uses local model)
  - Custom path: `/path/to/depth.pt`

Examples:
```
hybrid:facebook/sam-3d-body-vith+depth-anything/DA3-GIANT
hybrid:facebook/sam-3d-body-dinov3+Ruicheng/moge-2-vitl-normal
hybrid:/custom/human.ckpt+/custom/depth.pt
```

### Configuration Options

Common to all modes:
- `--process-res <pixels>`: Processing resolution for depth (default: 518)
- `--device <cuda|xpu|cpu|auto>`: Device selection
- `--frame-skip <N>`: Process every Nth frame (default: 5)
- `--opacity-threshold <0-1>`: Prune low-opacity Gaussians (default: 0.0)

SAM 3D Body specific:
- `--points-per-person <N>`: Points to sample per human (default: 10,000)

Hybrid specific:
- `--no-overlap-removal`: Disable overlap removal (keeps all depth Gaussians)
- `--overlap-threshold <0-1>`: IoU threshold for overlap detection (default: 0.5)

## Pipeline Flow

```
┌─────────────────────────────────────────────────────┐
│  Input Video                                    │
└──────────────────┬──────────────────────────────────┘
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
  ┌─────────────┐     ┌─────────────┐
  │   Humans    │     │ Background   │
  │ (SAM3DBody) │     │ (DA3/MoGe) │
  └──────┬──────┘     └──────┬──────┘
         │                   │
         ▼                   ▼
  ┌─────────────┐     ┌─────────────┐
  │ Mesh→Points │     │  Depth Map  │
  │ (sampled)   │     │             │
  └──────┬──────┘     └──────┬──────┘
         │                   │
         ▼                   ▼
  ┌─────────────┐     ┌─────────────┐
  │Texture Sample │     │Depth→Splats │
  │(2D proj)   │     │             │
  └──────┬──────┘     └──────┬──────┘
         │                   │
         └─────────┬─────────┘
                   ▼
    ┌──────────────────┐
    │  Remove Overlap │
    │(humans vs depth)│
    └───────┬────────┘
            ▼
    ┌───────────────────┐
    │  Combine Gaussians │
    │(human + background)│
    └───────┬──────────┘
            ▼
    ┌───────────────────┐
    │  Export to PLY    │
    └───────────────────┘
```

## Output Format

All modes produce **standard 3DGS PLY files** compatible with:
- vk_gaussian_splatting viewer (all pipelines)
- SplatTransform (PLY → SOG conversion)
- Other 3DGS renderers

### PLY Properties

Per Gaussian:
- `x, y, z`: Position (float32)
- `scale_x, scale_y, scale_z`: Log-scale (float32)
- `rot_x, rot_y, rot_z, rot_w`: Quaternion wxyz (float32)
- `f_dc_0, f_dc_1, f_dc_2`: SH DC term / color (float32)
- `opacity`: Logit opacity (float32)

### Hybrid Output Notes

For hybrid mode, each frame contains:
- **Human Gaussians**: From SAM 3D Body meshes with:
  - Accurate pose (skeleton-aware)
  - Structured surface (parametric)
  - Texture sampled from original image
- **Background Gaussians**: From depth model with:
  - Full scene depth
  - Textured color from image
  - No duplication in human regions (overlap removed)

## Performance Considerations

### SAM 3D Body

- **Model size**: ~2GB for ViT-H, ~3GB for DINOv3-H+
- **Inference time**: ~100-200ms per image (GPU)
- **Accuracy**: State-of-the-art pose and shape recovery

### Hybrid Mode

- **Inference time**: ~2-3x single depth model (runs both SAM 3D Body + depth)
- **Memory**: Requires both models loaded (~4-5GB GPU memory)
- **Quality**: Best for mixed scenes (humans + objects/foreground)

### Recommendations

- **Use SAM 3D Body** for: Human-only scenes, portraits, sports footage
- **Use Hybrid** for: Mixed scenes, casual videos, social media content
- **Use Depth-only** (DA3/MoGe/SHARP) for: Landscapes, indoor scenes, product shots

## Troubleshooting

### SAM 3D Body Import Error

```
ImportError: cannot import name 'sam_3d_body'
```

**Solution**: Install SAM 3D Body dependency:
```bash
uv sync --extra sam3dbody
```

Or manually:
```bash
pip install git+https://github.com/facebookresearch/sam-3d-body.git
```

### CUDA Out of Memory

Hybrid mode loads both models. If OOM:
1. Reduce `--process-res` (e.g., 320 instead of 518)
2. Reduce `--points-per-person` (e.g., 5000 instead of 10000)
3. Process in smaller chunks with `--chunk-size` (default: 10)

### No Humans Detected

SAM 3D Body may not detect humans in:
- Low resolution images (<256px height)
- Extreme poses (back to camera, extreme angles)
- Heavy occlusion (>80% body hidden)

**Solutions**:
- Increase input resolution
- Use manual bounding box if available
- Fall back to depth-only mode

## License Note

SAM 3D Body uses the **SAM License**. Check terms for commercial use.
- Research: Generally allowed with attribution
- Commercial: May require separate license

See: https://github.com/facebookresearch/sam-3d-body/blob/main/LICENSE
