/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <nvutils/parameter_registry.hpp>

#include "shaderio.h"

namespace vk_viewer {

// Parameters that controls the scene
struct SceneParameters
{
#ifdef WITH_DEFAULT_SCENE_FEATURE
  // do we load a default scene at startup if none is provided through CLI
  bool enableDefaultScene = true;
#endif

  // triggers a scene load at next frame when set to non empty string
  // If addToExisting is true, adds to existing scene; otherwise replaces
  std::filesystem::path sceneToLoadFilename;
  // When true, the new scene will be added to existing radiance fields
  // When false, all existing radiance fields will be replaced
  bool addSceneToExisting = false;
  // triggers a project load at next frame when set to non empty string
  std::filesystem::path projectToLoadFilename;
  // triggers a depth frame (VDZ) load at next frame when set to non empty string
  std::filesystem::path depthFrameToLoadFilename;
  // triggers an obj file import at next frame when set to non empty string
  std::filesystem::path meshToImportFilename;

  // Color space conversion mode for loaded PLY files
  // 0 = None (assume sRGB, standard for most 3DGS)
  // 1 = sRGB to Linear (for ML-SHARP compatibility-exported files, undo sRGB conversion)
  int colorSpaceConversion = 0;
  // If true, splats with (almost) black color will be removed during load
  bool removeBlackSplats = false;
  // If true, splats are reordered using Morton/Z-order curve for better cache coherency
  bool mortonReorder = true;
};

// Parameters that controls the scene
extern SceneParameters prmScene;

// Parameters that controls data format and storage in VRAM, shared by all pipeline
struct VramDataParameters
{
  int shFormat    = FORMAT_FLOAT32;
  int dataStorage = STORAGE_BUFFERS;
};

// Parameters that controls data storage
extern VramDataParameters prmData;

// Parameters that controls data format and storage in VRAM, specific to RTX pipelines
// Mainly about acceleration structures and particle primitive geometry
struct RtxVramDataParameters
{
  // if true will compact BLAS
  bool compressBlas = true;
  // set to true to use AABBs instead of mesh ICOSA primitives
  // This will also make the Rtx pipeline use parametric intersections
  // AABB mode is faster: eliminates 20-triangle icosahedron traversal overhead
  // and uses efficient particleDensityHitInstance path in intersection shader
  bool useAABBs = true;
  // Blackwell native sphere primitives (VK_NV_ray_tracing_linear_swept_spheres)
  // When enabled, uses hardware ray-sphere intersection instead of AABB+intersection shader
  // Only available on RTX 50 series (Blackwell) and newer GPUs
  // Falls back to AABB mode if extension not supported
  bool useSpheres = false;
  // if true, use one instance per splat in TLAS and single splat model in BLAS
  // otherwise, only one instance in TLAS and all splats transformed in BLAS
  bool useTlasInstances = true;

  // RTXMU-style acceleration structure memory optimizations
  // When enabled, reuses a persistent scratch buffer instead of per-build allocation
  bool useScratchPooling = true;
  // Maximum scratch buffer size for pooling (32MB default)
  // Larger values allow more parallel builds but use more memory
  uint32_t scratchPoolSizeMB = 32;

  // VK_NV_partitioned_acceleration_structure (PTLAS) for sparse FreeTimeGS updates
  // When enabled, partitions TLAS into spatial cells for incremental updates
  // Only available on Blackwell (RTX 50 series) and newer GPUs
  bool usePtlas = false;
  // World-space cell size for PTLAS partitioning (in scene units)
  // Smaller values = more partitions = finer-grained updates but more overhead
  float ptlasCellSize = 1.0f;
  // Maximum number of instances per partition (for PTLAS sizing)
  uint32_t ptlasMaxInstancesPerPartition = 1024;
};

// Parameters that controls data storage
extern RtxVramDataParameters prmRtxData;

// Parameters common to all rendering pipelines and provided to shaders as a UniformBufffer
// FrameInfo is defined in shaderio.h since declaration is shared with shaders
extern shaderio::FrameInfo prmFrame;

// pipeline selector
extern uint32_t prmSelectedPipeline;

// Parameters common to all rendering pipelines
struct RenderParameters
{
  int  visualize               = VISUALIZE_FINAL;
  bool wireframe               = false;  // display bounding volume
  int  maxShDegree             = 3;      // in [0,3]
  bool showShOnly              = false;
  bool opacityGaussianDisabled = false;
};

// Parameters common to all rendering pipelines
extern RenderParameters prmRender;

// Parameters that control rasterization
struct RasterParameters
{
  int32_t sortingMethod           = SORTING_GPU_SYNC_RADIX;
  bool    cpuLazySort             = true;  // if true, sorting starts only if viewpoint changed
  bool    gpuSortSkipWhenStable   = false;  // if true, skip GPU sort when camera is stationary
  bool    gpuSortForceEveryFrame  = false; // debug: force GPU sort every frame (overrides stability check)
  float   gpuSortPositionEpsilon  = 0.001f;  // position change threshold in world units
  float   gpuSortAngleEpsilon     = 0.001f;  // angle change threshold in radians
  int     frustumCulling          = FRUSTUM_CULLING_AT_DIST;
  int     distShaderWorkgroupSize = 256;  // best default value set by experimentation on ADA6000
  int     meshShaderWorkgroupSize = 32;   // best default value set by experimentation on ADA6000
  bool    fragmentBarycentric     = false;
  bool    pointCloudModeEnabled   = false;
  int     extentProjection        = EXTENT_CONIC;
  // Whether gaussians should be rendered with mip-splat
  // antialiasing https://niujinshuchong.github.io/mip-splatting/
  bool msAntialiasing = false;
  
  // Chunk-based hierarchical frustum culling
  // When enabled, splats are grouped into chunks of 256 and chunk AABBs are tested
  // against the frustum before processing individual splats
  bool chunkCullingEnabled = false;
};

// Parameters that control rasterization
extern RasterParameters prmRaster;

// Parameters that control Raytracing (RTX)
struct RtxParameters
{
  // temporalSampling is controlled by temporalSamplingMode, it is not directly exposed
  bool  temporalSampling       = false;  // do we accumulate frame results over time (for DOF and other)
  int   temporalSamplingMode   = TEMPORAL_SAMPLING_AUTO;  // how do we control temporal sampling activation
  int   kernelDegree           = KERNEL_DEGREE_QUADRATIC;
  float kernelMinResponse      = 0.0113f;  // constant value from Paper
  bool  kernelAdaptiveClamping = true;
  int   payloadArraySize       = 18;  // best default value set by experimentation on ADA6000
  // GRTX optimization: use global SoA K-buffer instead of ray payload arrays
  // Structure-of-Arrays layout provides coalesced memory access and reduces register pressure
  bool  useGlobalKBuffer       = true;
};

// Parameters that control Raytracing (RTX)
extern RtxParameters prmRtx;

// Parameters that control Compute Stochastic GS rendering
struct StochasticParameters
{
  uint32_t stochasticSamplesPerPixel          = 1;      // Samples per pixel (1=interactive, 16+=converged)
  uint32_t stochasticMaxSamples               = 64;     // Max samples before auto-reset
  uint32_t stochasticSupersamplingFactor      = 1;      // 1, 2, or 4 (SSAA factor)
  uint32_t stochasticUseGps                   = 0;      // 0=Stochastic Transparency, 1=GPS
  bool     stochasticEnableDof                = false;  // Depth of field
  bool     stochasticEnableProgressive        = true;   // Progressive accumulation
};

// Parameters that control Compute Stochastic GS
extern StochasticParameters prmStochastic;

// Parameters that control PBR/IBL environment mapping
struct PbrParameters
{
  std::filesystem::path envMap;              // Path to HDR environment map file
  float                 envMapRotation      = 0.0f;  // Rotation of environment map (radians)
  float                 envMapExposure      = 1.0f;  // Exposure multiplier
  bool                  pbrEnabled          = false; // Enable PBR rendering
  bool                  irradianceEnabled   = false; // Enable irradiance
  bool                  toneMapEnabled      = false; // Enable tone mapping
};

// Parameters that control PBR/IBL
extern PbrParameters prmPbr;

// Invoked by main() to save defaults after command line options are applied at startup
void storeDefaultParameters();

// Reset prmData to defaults
void resetDataParameters();
// Reset prmRtxData to defaults
void resetRtxDataParameters();

// Reset prmFrame to defaults
void resetFrameParameters();
// Reset prmRender to defaults
void resetRenderParameters();
// Reset prmRaster to defaults
void resetRasterParameters();
// Reset prmRtx to defaults
void resetRtxParameters();

// register the set of global parameters
void registerCommandLineParameters(nvutils::ParameterRegistry* parameterRegistry);

}  // namespace vk_viewer
