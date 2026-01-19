# SAM 3D Body Setup Instructions

**SAM 3D Body is NOT available on PyPI and cannot be installed via pip/uv.**

## Quick Setup

The SAM 3D Body code is already vendored at `python/sam_3d_body_git/`. The processor automatically adds this directory to `sys.path` before importing.

**No additional installation required!**

## Manual Installation (if needed)

If you need to update or reinstall SAM 3D Body:

```bash
cd python
# Remove old copy (if exists)
rm -rf sam_3d_body_git
# Clone fresh copy
git clone --depth 1 https://github.com/facebookresearch/sam-3d-body.git sam_3d_body_git
```

## Troubleshooting

### ImportError: cannot import name 'sam_3d_body'

The processors automatically add `sam_3d_body_git` to `sys.path`. Ensure the directory exists:

```bash
ls -la /home/maxious/vk_gaussian_splatting/python/sam_3d_body_git
```

If missing, clone manually:
```bash
cd /home/maxious/vk_gaussian_splatting/python
git clone --depth 1 https://github.com/facebookresearch/sam-3d-body.git sam_3d_body_git
```

### Missing Dependencies

SAM 3D Body requires:
- `trimesh` (for mesh sampling) - already in pyproject.toml
- `torch` (for inference) - installed with `--extra cuda`
- `pyrootutils` - vendored in sam_3d_body directory
- `open3d` - vendored in sam_3d_body directory

All are available once SAM 3D Body code is present.

## Files Installed

- `python/sam_3d_body_git/` - Complete SAM 3D Body repository
  - `sam_3d_body/` - Main package
  - `tools/` - Helper modules (detector, FOV, etc.)
  - `notebook/` - Notebook utilities
  - `pyrootutils/` - Required for SAM 3D Body setup
  - `open3d/` - Required for SAM 3D Body
  - `trimesh/` - Used for mesh point sampling

## License

SAM 3D Body uses **SAM License**. Check terms for commercial use.
See: `python/sam_3d_body_git/LICENSE`
