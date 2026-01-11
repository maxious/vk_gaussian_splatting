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

#ifndef _DEVICE_HOST_H_
#define _DEVICE_HOST_H_

// type of method used for sorting
#define SORTING_GPU_SYNC_RADIX 0
#define SORTING_CPU_ASYNC_MONO 1
#define SORTING_CPU_ASYNC_MULTI 2

// type of model storage
#define STORAGE_BUFFERS 0
#define STORAGE_TEXTURES 1

// format for SH storage
#define FORMAT_FLOAT32 0
#define FORMAT_FLOAT16 1
#define FORMAT_UINT8 2

// type of pipeline used
#define PIPELINE_VERT 0
#define PIPELINE_MESH 1
#define PIPELINE_RTX 2
#define PIPELINE_HYBRID 3        // Hybrid rendering: raster primary rays (3DGS), raytrace secondary rays (3DGRT)
#define PIPELINE_MESH_3DGUT 4    // 3DGUT (Unscented Transform) rasterization using mesh shaders
#define PIPELINE_HYBRID_3DGUT 5  // Hybrid rendering: raster primary rays (3DGUT), raytrace secondary rays (3DGRT)

// visualization mode
#define VISUALIZE_FINAL 0
#define VISUALIZE_CLOCK 1
#define VISUALIZE_DEPTH 2
#define VISUALIZE_RAYHITS 3
#define VISUALIZE_VDZ_DEPTH 4  // Display loaded VDZ depth frames (grayscale overlay)
#define VISUALIZE_VDZ_MESH 5   // Render depth-displaced 2.5D mesh from VDZ data

// type of frustum culling
#define FRUSTUM_CULLING_NONE 0
#define FRUSTUM_CULLING_AT_DIST 1
#define FRUSTUM_CULLING_AT_RASTER 2

// method used to compute the 2D extent projection from the 3D covariance
#define EXTENT_EIGEN 0  // basis aligned rectangular extent
#define EXTENT_CONIC 1  // axis aligned rectangular extend as in original INRIA

// type of camera
#define CAMERA_PINHOLE 0
#define CAMERA_FISHEYE 1

// particle format (PF), RTX
// not used in shaders (using RTX_USE_AABBS compiler defined instead)
// used only by UI but here to be easier to find
#define PARTICLE_FORMAT_ICOSAHEDRON 0
#define PARTICLE_FORMAT_PARAMETRIC 1

// degree of the splat kernel, RTX
#define KERNEL_DEGREE_QUINTIC 5
#define KERNEL_DEGREE_TESSERACTIC 4
#define KERNEL_DEGREE_CUBIC 3
#define KERNEL_DEGREE_QUADRATIC 2
#define KERNEL_DEGREE_LAPLACIAN 1
#define KERNEL_DEGREE_LINEAR 0

// bindings for set 0 (common to Raster and RTX)
#define BINDING_FRAME_INFO_UBO 0
#define BINDING_CENTERS_TEXTURE 1
#define BINDING_COLORS_TEXTURE 2
#define BINDING_COVARIANCES_TEXTURE 3
#define BINDING_SH_TEXTURE 4
#define BINDING_DISTANCES_BUFFER 5
#define BINDING_INDICES_BUFFER 6
#define BINDING_INDIRECT_BUFFER 7
#define BINDING_CENTERS_BUFFER 8
#define BINDING_COLORS_BUFFER 9
#define BINDING_COVARIANCES_BUFFER 10
#define BINDING_SH_BUFFER 11
#define BINDING_SCALES_TEXTURE 12
#define BINDING_ROTATIONS_TEXTURE 13
#define BINDING_SCALES_BUFFER 14
#define BINDING_ROTATIONS_BUFFER 15
#define BINDING_OPACITY_TEXTURE 16
#define BINDING_OPACITY_BUFFER 17
#define BINDING_RTX_PAYLOAD_BUFFER 18
#define BINDING_MESH_DESCRIPTORS 19
#define BINDING_LIGHT_SET 20
#define BINDING_MOTION_TEXTURE 21
#define BINDING_TIME_TEXTURE 22
#define BINDING_MOTION_BUFFER 23
#define BINDING_TIME_BUFFER 24

// VDZ depth mesh rendering textures (for VISUALIZE_VDZ_MESH mode)
#define BINDING_VDZ_VIDEO_TEXTURE 25
#define BINDING_VDZ_DEPTH_TEXTURE 26
#define BINDING_ENV_DEPTH_TEXTURE 27
#define BINDING_VDZ_DEPTH_SAMPLER 28
#define BINDING_VDZ_VIDEO_SAMPLER 29
#define BINDING_OPACITY_SAMPLER 30
#define BINDING_JOINT_MATRICES 31

// bindings for set 1 of RTX
#define RTX_BINDING_OUTIMAGE 0        // Ray tracer output image
#define RTX_BINDING_TLAS_SPLATS 1     // Top-level acceleration structure for splats
#define RTX_BINDING_TLAS_MESH 2       // Top-level acceleration structure for meshes
#define RTX_BINDING_PAYLOAD_BUFFER 3  // the alternative to payload stack (less efficient)
#define RTX_BINDING_AUX1 4            // Ray tracer auxiliary output image, when using hybrid mode + temporal sampling
#define RTX_BINDING_OUTDEPTH 5        // depth buffer
// DLSS-RR G-buffer outputs (when WITH_DLSS_RR is defined)
#define RTX_BINDING_DLSS_DIFFUSE_ALBEDO 6   // RGB diffuse albedo
#define RTX_BINDING_DLSS_SPECULAR_ALBEDO 7  // RGB specular albedo
#define RTX_BINDING_DLSS_NORMAL_ROUGH 8     // RGB normal + A roughness
#define RTX_BINDING_DLSS_MOTION 9           // RG motion vectors
#define RTX_BINDING_DLSS_LINEAR_DEPTH 10    // R linear depth
#define RTX_BINDING_DLSS_SPEC_HIT_DIST 11   // R specular hit distance

// Temporal sampling mode
#define TEMPORAL_SAMPLING_AUTO 0  // Detects automatically if TS is needed for best visual results (e.g. if DoF is on)
#define TEMPORAL_SAMPLING_ENABLED 1   // Force enabled
#define TEMPORAL_SAMPLING_DISABLED 2  // Force disabled

// bindings for set 0 of Post Process (0 is reserved for BINDING_FRAME_INFO_UBO)
#define POST_BINDING_MAIN_IMAGE 1  // the image that is presented
#define POST_BINDING_AUX1_IMAGE 2  // optional aux image to be accumulated (for example)
#define POST_BINDING_DEPTH_TEXTURE 3  // VDZ depth texture for visualization
#define POST_BINDING_DEPTH_SAMPLER 4  // VDZ depth sampler



// Mesh texture binding (for textured mesh rendering)
#define BINDING_MESH_TEXTURES 32

// location for vertex attributes
// (only for vertex shader mode)
#define ATTRIBUTE_LOC_POSITION 0
#define ATTRIBUTE_LOC_SPLAT_INDEX 1
// used for mesh rasterization
#define ATTRIBUTE_LOC_MESH_POSITION 0
#define ATTRIBUTE_LOC_MESH_NORMAL 1
#define ATTRIBUTE_LOC_MESH_TEXCOORD 2

#ifdef __cplusplus
#include "nvshaders/slang_types.h"
#include "wavefront.h"  // must be included before definition of DEFAULT
// used to assign fields defaults
#define DEFAULT(val) = val
namespace shaderio {
#else
// we are in Slang here
#include "wavefront.h"  // must be included before definition of DEFAULT
// used to skip fields init
// when included in shaders
#define DEFAULT(val)
#endif

struct FrameInfo
{
  float3   cameraPosition;  // position in world space
  float4   viewQuat;        // quaternion storing the rotation part of the view matrix
  float3   viewTrans;       // translation part of the view matrix
  float4x4 viewMatrix;
  float4x4 viewInverse;  // Camera inverse view matrix

  float4x4 projectionMatrix;
  float4x4 projInverse;  // Camera inverse projection matrix
  float2   nearFar;
  float2   focal;
  float2   viewport;
  float2   basisViewport;
  float2   viewportOffset DEFAULT(float2(0.0f, 0.0f));  // Viewport offset for SBS stereo (x,y offset in pixels)
  float2   stereoShift DEFAULT(float2(0.0f, 0.0f));    // Principal point shift for off-axis stereo projection (in pixels)
  
  float currentTime DEFAULT(0.0f);

  int temporalCulling DEFAULT(1); // 1 = enabled, 0 = disabled
  int applyMotion     DEFAULT(1); // 1 = enabled, 0 = disabled (experimental)

  float fovRad                 DEFAULT(0.009f);  // Field of view in radians for fisheye camera
  float inverseFocalAdjustment DEFAULT(1.0f);

  // Ortho is not fully implemented
  float orthoZoom          DEFAULT(1.0f);  //
  int32_t orthographicMode DEFAULT(0);     // disabled, in [0,1]

  int32_t splatCount DEFAULT(0);     //
  float splatScale   DEFAULT(1.0f);  // in {0.1, 2.0}

  float frustumDilation    DEFAULT(0.2f);           // for frustum culling, 2% scale
  float alphaCullThreshold DEFAULT(1.0f / 255.0f);  // for alpha culling

  int32_t lightCount DEFAULT(0);
  int2               cursor;        // position of the mouse cursor for debug
  int32_t maxPasses  DEFAULT(200);  // RTX maximum hits during marching

  float alphaClamp       DEFAULT(0.99f);  // 0.99 in original paper
  float minTransmittance DEFAULT(0.01f);  // 0.1  in original paper ? TO check
  int32_t rtxMaxBounces  DEFAULT(3);

  float multiplier DEFAULT(1.0f);  // for alternative visualization modes

  int32_t frameSampleId  DEFAULT(0);    // the frame sample index since last frame sampling reset
  int32_t frameSampleMax DEFAULT(200);  // maximum number of frame after which we stop accumulating frames samples

  float focusDist DEFAULT(1.3f);    // focus distance to compute depth of field
  float aperture  DEFAULT(0.001f);  // aperture distance to compute depth of field, 0 does no DOF effect

  // DLSS-RR temporal jitter (in pixels, range [-0.5, 0.5])
  float2 dlssJitter DEFAULT(float2(0.0f, 0.0f));
  // Previous frame matrices for motion vector calculation
  float4x4 prevViewMatrix;
  float4x4 prevProjectionMatrix;

  // HDR10 encoding parameters
  float hdrPaperWhiteNits DEFAULT(200.0f);  // SDR white level in nits (typical: 80-400)
  float hdrMaxNits DEFAULT(1000.0f);        // Display peak luminance in nits
  float hdrExposure DEFAULT(1.0f);          // Exposure multiplier
  int32_t hdrEnabled DEFAULT(0);            // 0 = disabled, 1 = enabled

  // Color space conversion for ML-SHARP linearRGB Gaussians
  // When enabled, applies linear-to-sRGB gamma correction after rendering
  int32_t linearToSrgb DEFAULT(0);          // 0 = disabled, 1 = enabled (for SHARP PLY files)

  // Multiview (VK_KHR_multiview) support for mobile VR
  // When enabled, stores per-eye view/projection matrices for efficient stereo rendering
  int32_t multiviewEnabled DEFAULT(0);      // 0 = disabled, 1 = enabled
  // Eye-specific data (indexed by gl_ViewIndex in multiview mode)
  float4x4 viewMatrixArray[2];              // View matrices for eye 0 and eye 1
  float4x4 viewInverseArray[2];             // Inverse view matrices for eye 0 and eye 1
  float4x4 projectionMatrixArray[2];        // Projection matrices for eye 0 and eye 1
  float4x4 projInverseArray[2];             // Inverse projection matrices for eye 0 and eye 1
  float3   cameraPositionArray[2];          // Eye positions for eye 0 and eye 1
  // Per-eye sensor pose for 3DGUT projection (extracted from viewMatrixArray)
  float4   viewQuatArray[2];                // Rotation quaternions for eye 0 and eye 1
  float3   viewTransArray[2];               // Translation vectors for eye 0 and eye 1

  // VDZ depth mesh rendering parameters (for VISUALIZE_VDZ_MESH mode)
  float vdzZScale     DEFAULT(10.0f);  // Depth scale multiplier
  float vdzZBias      DEFAULT(2.0f);   // Global Z offset (added after scaling)
  float vdzZGamma     DEFAULT(5.0f);   // Gamma correction for depth (pow(depth, gamma))
  float vdzZMaxClip   DEFAULT(0.2f);   // Maximum depth clipping threshold
  float vdzZMin       DEFAULT(0.0f);   // Min depth from metadata (for colormap)
  float vdzZMax       DEFAULT(1.0f);   // Max depth from metadata (for colormap)
  float vdzPlaneScale DEFAULT(1.4f);   // Scale of the view plane
  float vdzAspect     DEFAULT(1.777f); // Aspect ratio (width/height) of depth texture
  float vdzEdgeThreshold DEFAULT(0.15f);  // Max depth gradient before discarding fragment (edge detection)
  int32_t visualize   DEFAULT(0);      // Current visualization mode
  int32_t vdzUseVideoTexture DEFAULT(0);  // 1 = use video RGB texture, 0 = use colormap
  int32_t vdzWorldSpaceMode DEFAULT(0);   // 0 = camera-attached (2.5D), 1 = world-space (VR)
  float4x4 vdzModelMatrix;                // Captured model matrix for world-space mode

  // Parallax rendering parameters (Tiefling-style depth-based displacement)
  float2 vdzParallaxOffset DEFAULT(float2(0.0f, 0.0f));   // Mouse/touch offset for parallax (normalized)
  float  vdzParallaxStrength DEFAULT(0.5f);              // Parallax intensity multiplier (0.0 = disabled)
  float  vdzParallaxFocus DEFAULT(0.5f);                 // Focus plane depth (0.0 = near, 1.0 = far)
  float  vdzParallaxEdgeSoftness DEFAULT(0.02f);         // Edge softening factor to reduce artifacts at borders

  // Hybrid mode: Combined mesh displacement + POM
  // 0 = divided mesh only, 1 = POM only, 2 = hybrid (mesh + POM)
  int32_t vdzHybridMode DEFAULT(0);
  float  vdzHybridMeshStrength DEFAULT(1.0f);  // Strength of mesh displacement (0-1)
  float  vdzHybridPomStrength DEFAULT(0.5f);   // Strength of POM effect (0-1)
  int32_t vdzHybridPomLayers DEFAULT(16);      // Number of POM layers (8-32)
  float  vdzHybridPomDepthScale DEFAULT(0.02f); // Height scale for POM (in texture UV space)

  // Gap filling parameters (for disocclusion handling)
  float3 vdzGapFillColor DEFAULT(float3(0.05f, 0.05f, 0.08f));  // Background color for filling gaps
  int32_t vdzGapFillEnabled DEFAULT(1);             // 0 = discard, 1 = fill with color
  int32_t vdzGapFillNeighbor DEFAULT(1);            // 0 = solid color, 1 = neighbor average fill
  float  vdzGapFeather DEFAULT(0.0f);               // Edge feathering amount (0.0 = hard edge, 1.0 = soft)
  int32_t vdzGapFillRadius DEFAULT(3);              // Radius for neighbor sampling (in pixels)

  // Bilateral filter parameters (for depth smoothing preprocessing)
  int32_t vdzBilateralEnabled DEFAULT(0);           // 1 = apply bilateral filter to depth
  float  vdzBilateralSigmaSpace DEFAULT(2.0f);      // Spatial sigma for bilateral filter
  float  vdzBilateralSigmaDepth DEFAULT(0.1f);      // Depth range sigma for bilateral filter

  // Environment Depth Occlusion
  int32_t envDepthAvailable DEFAULT(0);   // 1 = enabled/available, 0 = disabled
  float envDepthNear DEFAULT(0.01f);
  float envDepthFar DEFAULT(100.0f);
  float4x4 envDepthViewMatrixArray[2];
  float4x4 envDepthProjectionMatrixArray[2];
};

// Push constant for raster
struct PushConstant
{
  // model transformation
  float4x4 modelMatrix;
  float4x4 modelMatrixInverse;
  float4x4 modelMatrixRotScaleInverse;  // inverse of the rotation/scale part of the modelMatrix
  uint32_t objIndex;                    // index of the mesh being rendered if any
};

// indirect parameters for
// - vkCmdDrawIndexedIndirect (first 6 attr)
// - vkCmdDrawMeshTasksIndirectEXT (last 3 attr)
struct IndirectParams
{
  // for vkCmdDrawIndexedIndirect
  uint32_t indexCount    DEFAULT(6);  // allways = 6 indices for the quad (2 triangles)
  uint32_t instanceCount DEFAULT(0);  // will be incremented by the distance compute shader
  uint32_t firstIndex    DEFAULT(0);  // allways zero
  uint32_t vertexOffset  DEFAULT(0);  // allways zero
  uint32_t firstInstance DEFAULT(0);  // allways zero

  // for vkCmdDrawMeshTasksIndirectEXT
  uint32_t groupCountX DEFAULT(0);  // Will be incremented by the distance compute shader
  uint32_t groupCountY DEFAULT(1);  // Allways one workgroup on Y
  uint32_t groupCountZ DEFAULT(1);  // Allways one workgroup on Z

  // for debug info readback, TODO shall be in some other buffer
  int32_t particleID DEFAULT(-1);   // Will be set by the ID of the nearest splat on the ray path
  float particleDist DEFAULT(0.0);  // Will be set by the dist to the nearest splat on the ray path
  float val1         DEFAULT(0.0);
  float val2         DEFAULT(0.0);
  float val3         DEFAULT(0.0);
  float val4         DEFAULT(0.0);
  float val5         DEFAULT(0.0);
  float val6         DEFAULT(0.0);
  float val7         DEFAULT(0.0);
  float val8         DEFAULT(0.0);
};


// Push constant specific to raytracing
struct PushConstantRay
{
  // model transformation
  float4x4 modelMatrix;
  float4x4 modelMatrixInverse;
  float4x4 modelMatrixRotScaleInverse;  // inverse of the rotation/scale part of the modelMatrix
  // stores the geometry of the splat primitive
  // provisionaly placed here, shall not be in push constant,
  // rather in some objects description buffer
  uint64_t vertexAddress;  // Address of the Vertex buffer
  uint64_t indexAddress;   // Address of the index buffer
  // set to true will raytrace the mesh depth as a pre-pass
  bool meshDepthOnly;
  // viewport offset for SBS stereo rendering
  int2 viewportOffset DEFAULT(int2(0, 0));
  int useNdcMotion DEFAULT(0); // 1 = enabled, 0 = disabled
};

#ifdef __cplusplus
}  // namespace shaderio
#endif

#ifndef __cplusplus
struct HitPayload
{
  int   id[PAYLOAD_ARRAY_SIZE];
  float dist[PAYLOAD_ARRAY_SIZE];
#if WIREFRAME
  float2 bary[PAYLOAD_ARRAY_SIZE];  // hit barycentrics
#endif
};
#endif

#undef DEFAULT
#endif
