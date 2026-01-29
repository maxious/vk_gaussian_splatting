# Future Work / TODO

This document tracks planned improvements and research directions for vk_gaussian_splatting.

## Ray Tracing Optimizations (GRTX Paper)

Based on [GRTX: Efficient Ray Tracing for 3D Gaussian-Based Rendering](https://arxiv.org/html/2601.20429v1) (arXiv 2601.20429).

### High Priority

- [x] **Default to AABB mode for RTX splats** ✅ IMPLEMENTED
  - Changed `useAABBs = true` in `parameters.h`
  - Eliminates 20-triangle icosahedron traversal overhead
  - The intersection shader already has efficient `particleDensityHitInstance` path
  - Effort: 5 minutes

- [x] **Global SoA K-buffer** ✅ IMPLEMENTED
  - Moved K-buffer arrays from ray payload to global GPU memory buffers
  - Structure-of-Arrays layout: `gKDist[slotIndex * numRays + rayIndex]` for coalesced access
  - Controlled by `prmRtx.useGlobalKBuffer` (enabled by default)
  - Reduced payload now only carries rayIndex/numRays and mesh hit data
  - Files modified:
    - `src/parameters.h` - Added `useGlobalKBuffer` option
    - `src/vk_viewer_rtx.cpp` - K-buffer creation/binding
    - `shaders/shaderio.h` - Conditional payload struct, K-buffer bindings
    - `shaders/threedgrt_raytrace.rgen.slang` - K-buffer init/read
    - `shaders/threedgrt_raytrace.rahit.slang` - K-buffer write in any-hit
    - `shaders/threedgrt_payload.h.slang` - K-buffer accessor helpers

### Medium Priority

- [x] **Blackwell native sphere primitives** ✅ IMPLEMENTED
  - Uses `VK_NV_ray_tracing_linear_swept_spheres` extension for hardware ray-sphere intersection
  - Build BLAS with single radius-3 sphere (3σ coverage) instead of AABB
  - Intersection shader computes density `t_closest` (Gaussian center), not surface hit `t`
  - Controlled by `prmRtxData.useSpheres` parameter (UI: "Sphere (Blackwell)")
  - Requires RTX 50 series (Blackwell) GPU with extension support
  - Falls back to AABB mode if extension not available
  - Files modified:
    - `src/main.cpp` - Enable VK_NV_ray_tracing_linear_swept_spheres extension
    - `src/parameters.h` - Added `useSpheres` option in RtxVramDataParameters
    - `src/splat_set_vk.h` - Added sphere buffers to SplatModel struct
    - `src/splat_set_vk.cpp` - Sphere buffer creation and BLAS building with VkAccelerationStructureGeometrySpheresDataNV
    - `src/vk_viewer_shaders.cpp` - Added RTX_USE_SPHERES macro
    - `shaders/shaderio.h` - Added PARTICLE_FORMAT_SPHERE constant
    - `shaders/threedgrt_raytrace.rint.slang` - Sphere intersection handling
    - `src/vk_viewer_ui.cpp` - Added "Sphere (Blackwell)" to particle format dropdown
    - `src/vk_viewer_ui_renderer.cpp` - UI logic for sphere mode selection
    - `src/vk_viewer_ui_project.cpp` - Save/load useSpheres setting

### Low Priority / Research

- [ ] **Traversal checkpointing** (hardware feature)
  - GRTX proposes hardware support for resuming BVH traversal from checkpointed nodes
  - Not implementable in software, but worth monitoring for future GPU architectures

- [ ] **RTNN-style query partitioning**
  - From [RTNN: Accelerating Neighbor Search Using Hardware Ray Tracing](https://github.com/horizon-research/rtnn)
  - Partition rays and build specialized BVHs per partition for tighter AABBs
  - May help with scenes that have highly non-uniform Gaussian distributions

## Offline PLY Generation Optimizations

### Reduced-3DGS Integration

Based on [Reducing the Memory Footprint of 3D Gaussian Splatting](https://github.com/graphdeco-inria/reduced-3dgs) (INRIA).

- [ ] **Resolution-aware primitive pruning**
  - Reduce splat count by ~50% with minimal quality loss
  - Fewer splats = faster RT acceleration structure builds and traversal
  - Integrate into `python/offline/export_gaussian_ply.py`
  - Effort: 1 day

- [ ] **Adaptive SH coefficient selection**
  - Per-Gaussian SH band selection (0-3 bands based on view-dependency)
  - Reduces memory and potentially speeds up color evaluation
  - Requires PLY format changes (already supported per reduced-3dgs)
  - Effort: 2-3 days

- [ ] **Codebook quantization + half-float storage**
  - 27× size reduction on disk
  - 1.7× render speedup claimed
  - May require loader changes for quantized formats
  - Effort: 1 week

### Existing Optimizations

The codebase already includes GPU-accelerated kNN for motion tracking:
- `python/offline/motion_tracking_cuda.py` - cuTile-based kNN kernel
- Used for FreeTimeGS 4D Gaussian motion vector computation
- Deterministic tie-breaking for reproducibility

## Denoiser Integration

- [ ] **DLSS integration for DoF convergence**
  - Currently DoF uses temporal accumulation which is slow to converge
  - DLSS denoiser would improve visual quality during convergence period
  - Mentioned in `doc/ray_tracing_3d_gaussians.md`

## References

- GRTX Paper: https://arxiv.org/html/2601.20429v1
- Reduced-3DGS: https://github.com/graphdeco-inria/reduced-3dgs
- RTNN: https://github.com/horizon-research/rtnn
- 3DGRT: https://gaussiantracer.github.io/
