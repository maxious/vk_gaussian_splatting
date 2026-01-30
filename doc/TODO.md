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

## Blackwell Extensions (RTX 50 Series)

Analysis of NVIDIA Blackwell extensions for Gaussian splatting ray tracing optimization.

### High Priority

- [x] **VK_NV_partitioned_acceleration_structure (PTLAS)** ✅ INFRASTRUCTURE IMPLEMENTED
  - Partitions TLAS into spatial cells, only rebuild changed partitions
  - GPU-driven updates via atomics - no CPU synchronization needed
  - **Ideal for FreeTimeGS**: sparse per-frame updates only affect subset of partitions
  - Expected gain: **2-10× TLAS update speedup** when <10% splats animate per frame
  - Sample code: [vk_partitioned_tlas](https://github.com/nvpro-samples/vk_partitioned_tlas)
  - Current status: Infrastructure complete, awaiting Blackwell hardware for full implementation
  - Files modified:
    - `src/main.cpp` - ✅ Register VK_NV_partitioned_acceleration_structure extension
    - `src/parameters.h` - ✅ Added `usePtlas`, `ptlasCellSize`, `ptlasMaxInstancesPerPartition`
    - `src/splat_set_vk.cpp` - ✅ PTLAS partitioning and dirty tracking implemented
    - `src/splat_set_vk.h` - ✅ PtlasData struct with partition tracking
  - Implementation completed:
    1. ✅ Extension registered with VkPhysicalDevicePartitionedAccelerationStructureFeaturesNV
    2. ✅ Partition splats into fixed-size world-space grid cells at load time
    3. ✅ Store per-splat cell ID and per-cell instance lists
    4. ✅ Track dirty bitset for cells with animated splats
    5. ✅ Per-frame: mark cells dirty when splats update
    6. ✅ Handle splats moving across cell boundaries (dirty both old+new cells)
  - TODO: Full vkCmdBuildPartitionedAccelerationStructuresNV implementation (requires Blackwell)

### Medium Priority

- [x] **RTXMU-style AS memory pooling** ✅ PARAMETERS ADDED
  - Suballocate multiple small AS into 8-32MB blocks to reduce fragmentation
  - Reuse scratch buffers across builds instead of per-frame allocate/free
  - Expected gain: **10-30% memory reduction**, fewer allocation spikes
  - Sample code: [RTXMU](https://github.com/NVIDIA-RTX/RTXMU)
  - Current status: Parameters added, scratch buffer pool struct defined
  - Files modified:
    - `src/parameters.h` - ✅ Added `useScratchPooling`, `scratchPoolSizeMB`
    - `src/splat_set_vk.h` - ✅ Added `rtScratchBufferPool`, `rtScratchBufferPoolSize`
  - Implementation completed:
    1. ✅ Parameters for scratch pool size configuration
    2. ✅ Buffer members for persistent scratch buffer
  - TODO: Integrate pooling into acceleration structure builds (nvvk helpers already support scratch reuse)

### Low Priority / Not Applicable

- [ ] **VK_NV_cluster_acceleration_structure (CLAS)**
  - Separates triangle geometry from BLAS, enables template-based BVH treelets
  - **Not applicable currently**: designed for triangle meshes, not procedural AABB/spheres
  - Would require architectural change to triangle-based splat proxies
  - Only reconsider if switching to clustered BLAS with triangulated splats
  - Sample code: [vk_animated_clusters](https://github.com/nvpro-samples/vk_animated_clusters), [vk_tessellated_clusters](https://github.com/nvpro-samples/vk_tessellated_clusters)

- [ ] **OMM (Opacity Micro-Maps)**
  - Encodes opacity states per micro-triangle to skip any-hit shader invocations
  - **Not applicable currently**: requires triangle geometry with alpha masks
  - Gaussians use continuous density evaluation, not binary opacity thresholds
  - Only reconsider if switching to micro-triangle sprite representation
  - Sample code: [OMM](https://github.com/NVIDIA-RTX/OMM)

- ❌ **RTXMG (RTX Mega Geometry)**
  - Displacement micro-mapping for subdivision surfaces with adaptive tessellation
  - **Not applicable**: designed for displaced meshes, not volumetric primitives
  - Sample code: [RTXMG](https://github.com/NVIDIA-RTX/RTXMG)

### Extension Applicability Summary

| Extension | Current Arch Match | Expected Gain | Effort |
|-----------|-------------------|---------------|--------|
| PTLAS | ✅ High (TLAS-per-splat + sparse FreeTimeGS) | 2-10× TLAS update | 1-2 days |
| RTXMU pooling | ✅ Medium (memory optimization) | 10-30% memory | 1-3 hours |
| CLAS | ⚠️ Low (requires triangle geometry) | N/A | N/A |
| OMM | ⚠️ Low (requires alpha-masked triangles) | N/A | N/A |
| RTXMG | ❌ None (displacement meshes) | N/A | N/A |

## References

- GRTX Paper: https://arxiv.org/html/2601.20429v1
- Reduced-3DGS: https://github.com/graphdeco-inria/reduced-3dgs
- RTNN: https://github.com/horizon-research/rtnn
- 3DGRT: https://gaussiantracer.github.io/
- vk_partitioned_tlas: https://github.com/nvpro-samples/vk_partitioned_tlas
- vk_animated_clusters: https://github.com/nvpro-samples/vk_animated_clusters
- vk_tessellated_clusters: https://github.com/nvpro-samples/vk_tessellated_clusters
- RTXMU: https://github.com/NVIDIA-RTX/RTXMU
- RTXMG: https://github.com/NVIDIA-RTX/RTXMG
- OMM: https://github.com/NVIDIA-RTX/OMM
