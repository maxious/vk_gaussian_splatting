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

#include "shaderio.h"
#include "threedgs.h.slang"
#include "threedgs_particles_storage.h.slang"
#include "threedgut.h.slang"

// Parallel Processing : Each global invocation (thread) processes one splat.
// Batch Processing : The workgroup can process up to RASTER_MESH_WORKGROUP_SIZE splats(outputQuadCount)

static const int MAX_VERTICES   = 4 * RASTER_MESH_WORKGROUP_SIZE;
static const int MAX_PRIMITIVES = 2 * RASTER_MESH_WORKGROUP_SIZE;

// clang-format off

// Per primitive output
struct PrimitiveOutput
{
  [[vk::location(0)]] nointerpolation uint splatId : SPLATID;
  [[vk::location(1)]] nointerpolation float4 splatCol : COLOR0;
  [[vk::location(2)]] nointerpolation float3 splatPosition : POSITION0;
  [[vk::location(3)]] nointerpolation float3 splatScale : SCALE0;
  [[vk::location(4)]] nointerpolation float4 splatRotation : ROTATION0;  // a quaternion
};

// Per vertex output
struct VertexOutput
{
  float4 position : SV_Position;
};

[[vk::binding(BINDING_FRAME_INFO_UBO, 0)]] ConstantBuffer<FrameInfo> frameInfo;
[[vk::push_constant]] ConstantBuffer<PushConstant> pcRaster;
// sorted indices
[[vk::binding(BINDING_INDICES_BUFFER, 0)]] RWStructuredBuffer<uint32_t> indices;
// to get the actual number of splats (after culling if any)
[[vk::binding(BINDING_INDIRECT_BUFFER, 0)]] RWStructuredBuffer<IndirectParams> indirectBuffer;

// clang-format on

// used when quad need to be discard
void emitDegeneratedQuad(uint localIndex, out OutputVertices<VertexOutput, MAX_VERTICES> verts)
{
  [unroll]
  for(uint i = 0; i < 4; ++i)
  {
    verts[localIndex * 4 + i].position = float4(0.0, 0.0, 2.0, 1.0);
  }
}

[numthreads(RASTER_MESH_WORKGROUP_SIZE, 1, 1)]
[outputtopology("triangle")]
[shader("mesh")]
void main(
  uint3 groupThreadID : SV_GroupThreadID,
  uint3 groupID : SV_GroupID,
  uint3 dispatchThreadID : SV_DispatchThreadID,
#ifdef MULTIVIEW_ENABLED
  uint viewIndex : SV_ViewID,
#endif
  out OutputVertices<VertexOutput, MAX_VERTICES> verts,
  out OutputPrimitives<PrimitiveOutput, MAX_PRIMITIVES> prims,
  out OutputIndices<uint3, MAX_PRIMITIVES> triangles)
{
  const uint localIndex = groupThreadID.x;
  const uint32_t baseIndex = dispatchThreadID.x;
#if FRUSTUM_CULLING_MODE == FRUSTUM_CULLING_AT_DIST
  const uint splatCount = indirectBuffer[0].instanceCount;
#else
  const uint splatCount = frameInfo.splatCount;
#endif
  const uint outputQuadCount = min(RASTER_MESH_WORKGROUP_SIZE, splatCount - groupID.x * RASTER_MESH_WORKGROUP_SIZE);

  if(localIndex == 0)
  {
    SetMeshOutputCounts(outputQuadCount * 4, outputQuadCount * 2);
  }

  // Select camera position based on multiview mode
#ifdef MULTIVIEW_ENABLED
  float3 cameraPos = (frameInfo.multiviewEnabled != 0) ? frameInfo.cameraPositionArray[viewIndex] : frameInfo.cameraPosition;
#else
  float3 cameraPos = frameInfo.cameraPosition;
#endif

  if(baseIndex < splatCount)
  {
    const uint splatIndex = indices[baseIndex];

    triangles[localIndex * 2 + 0] = uint3(0, 2, 1) + localIndex * 4;
    triangles[localIndex * 2 + 1] = uint3(2, 0, 3) + localIndex * 4;
    
    // Fetch particle minimal data (done as early as possible for performance)
    float4       splatColor  = fetchColor(splatIndex);
    float3       splatCenter = fetchCenter(splatIndex);
    const float3 splatScale  = exp(fetchScale(splatIndex));
    const float4 splatRotation = normalize(vec4toQuat(fetchRotation(splatIndex)));

    // Apply 4D Motion/Temporal
    float3 motion = fetchMotion(splatIndex);
    float2 timeData = fetchTime(splatIndex);
    float t_scale = timeData.x; // JS Order: x = scale
    float t_center = timeData.y; // JS Order: y = center
    float dt = frameInfo.currentTime - t_center;

    if (frameInfo.temporalCulling != 0 && t_center > -5.0) {
        // FreeTimeGS Gaussian opacity decay: sigma(t) = exp(-0.5 * ((t - mu) / s)^2)
        if (t_scale > 1e-5) {
            float v = dt / t_scale;
            float temporalOpacity = exp(-0.5 * v * v);
            splatColor.a *= temporalOpacity;

            // Cull if opacity is negligible to improve performance
            if (splatColor.a < 0.004) { // approx 1/255
                emitDegeneratedQuad(localIndex, verts);
                return;
            }
        } else {
            // Zero duration: only visible at exact time center
            if (abs(dt) > 1e-5) {
                emitDegeneratedQuad(localIndex, verts);
                return;
            }
        }
    }

    if (frameInfo.applyMotion != 0) {
        splatCenter += motion * dt;
    }

    prims[localIndex * 2 + 0].splatId = splatIndex;
    prims[localIndex * 2 + 1].splatId = splatIndex;
    
    prims[localIndex * 2 + 0].splatPosition = splatCenter;
    prims[localIndex * 2 + 1].splatPosition = splatCenter;
    prims[localIndex * 2 + 0].splatScale    = splatScale;
    prims[localIndex * 2 + 1].splatScale    = splatScale;
    prims[localIndex * 2 + 0].splatRotation = splatRotation;
    prims[localIndex * 2 + 1].splatRotation = splatRotation;

    // fetch radiance from SH coefs > degree 0, we work in model coordinates
    const float3 worldViewDir = normalize(splatCenter - mul(float4(cameraPos, 1.0), pcRaster.modelMatrixInverse).xyz);

#if SHOW_SH_ONLY == 1
    splatColor.rgb = float3(0.5);
#endif
    splatColor.rgb += fetchViewDependentRadiance(splatIndex, worldViewDir);

    // alpha based culling
    if(splatColor.a < frameInfo.alphaCullThreshold)
    {
      emitDegeneratedQuad(localIndex, verts);
      return;
    }

#if MS_ANTIALIASING == 0
    // emit per primitive color as early as possible for performance reasons
    prims[localIndex * 2 + 0].splatCol = splatColor;
    prims[localIndex * 2 + 1].splatCol = splatColor;
#endif

#if CAMERA_TYPE == CAMERA_PINHOLE
    CameraModelParameters sensorModel = initPerfectPinholeCamera(frameInfo.nearFar, frameInfo.viewport, frameInfo.focal, frameInfo.stereoShift);
#else
    CameraModelParameters sensorModel = initPerfectFisheyeCamera(frameInfo.viewport, frameInfo.focal, frameInfo.stereoShift);
#endif

    // For XR locomotion with model matrix: We need to work in model space to avoid
    // coordinate flip issues. The 3DGUT projection applies RUB-to-RUF coordinate flips
    // that assume the position is in a canonical orientation. When the model matrix
    // rotates the scene, the world-space Z-axis no longer corresponds to depth.
    //
    // Solution: Transform the camera pose to model space (like ray tracing fix),
    // keep particles in model space, and pass identity matrix to the projection.
    
    // Get the appropriate view matrix for this eye
#ifdef MULTIVIEW_ENABLED
    const float4x4 viewMat = (frameInfo.multiviewEnabled != 0) ? frameInfo.viewMatrixArray[viewIndex] : frameInfo.viewMatrix;
#else
    const float4x4 viewMat = frameInfo.viewMatrix;
#endif

    // Compute the model-view matrix that transforms from model space to camera space
    const float4x4 modelViewMatrix = mul(pcRaster.modelMatrix, viewMat);
    
    // Extract sensor pose from model-view matrix
    // The translation is in the last row (row-major layout)
    const float3 mvTrans = float3(modelViewMatrix[3].xyz);
    
    // For the quaternion: Slang uses row-major matrices with row-vector convention,
    // but mat3toQuat expects column-major convention (matching GLM/CPU side).
    // We must transpose the 3x3 rotation part before extracting the quaternion.
    // 
    // In row-major: matrix[i] gives row i, matrix[i][j] is element at row i, col j
    // We need to extract columns and build a transposed matrix for correct quaternion.
    float3x3 mvRot3x3;
    // Extract columns from the 4x4 matrix (column j = elements [0][j], [1][j], [2][j])
    mvRot3x3[0] = normalize(float3(modelViewMatrix[0][0], modelViewMatrix[1][0], modelViewMatrix[2][0]));
    mvRot3x3[1] = normalize(float3(modelViewMatrix[0][1], modelViewMatrix[1][1], modelViewMatrix[2][1]));
    mvRot3x3[2] = normalize(float3(modelViewMatrix[0][2], modelViewMatrix[1][2], modelViewMatrix[2][2]));
    const quat mvQuat = mat3toQuat(mvRot3x3);
    
    // Create sensor state with model-space pose
    SensorState sensorState = initGlobalShutterSensorState(mvTrans, mvQuat);

    int2 resolution = int2(int(frameInfo.viewport.x), int(frameInfo.viewport.y));

    float3 particleSensorRay;
    float2 particleProjCenter;
    float3 particleProjCovariance;

    // Pass identity matrix since model transform is now baked into the sensor state.
    // Particles stay in model space; the sensor pose has been transformed to model space.
    const float4x4 identityMatrix = float4x4(1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1);
    if(!threedgutParticleProjection(resolution, sensorModel, identityMatrix, sensorState, splatCenter, splatScale,
                                    quatToMat3(splatRotation), particleProjCenter, particleProjCovariance))
    {
      emitDegeneratedQuad(localIndex, verts);
      return;
    }

// Method from paper used for cuda raster
#if EXTENT_METHOD == EXTENT_CONIC
    float2  extent;
    float4  conicOpacity;
    float maxConicOpacityPower;

    if(!threedgutProjectedExtentConicOpacity(particleProjCovariance, splatColor.a, extent, conicOpacity, maxConicOpacityPower))
    {
      emitDegeneratedQuad(localIndex, verts);
      return;
    }

#if MS_ANTIALIASING == 1
    splatColor.a = conicOpacity.a;
#endif

// Method optimized for rasterisation pipelines
#elif EXTENT_METHOD == EXTENT_EIGEN

    float2 basisVector1, basisVector2;

    if(!threedgsProjectedExtentBasis(particleProjCovariance, 3.33, frameInfo.splatScale, splatColor.a, basisVector1, basisVector2))
    {
      emitDegeneratedQuad(localIndex, verts);
      return;
    }
#endif

#if MS_ANTIALIASING == 1
    // emit the fragment color with compensation
    prims[localIndex * 2 + 0].splatCol = splatColor;
    prims[localIndex * 2 + 1].splatCol = splatColor;
#endif

    // Convert projected particle center from pixel coordinates to NDC [-1, 1]
    // particleProjCenter is in pixel coordinates, need to convert to NDC
    const float2 ndcXY = (particleProjCenter / frameInfo.viewport) * 2.0 - 1.0;

    const float4 worldSplatCenter = mul(float4(splatCenter, 1.0), pcRaster.modelMatrix);
    // In multiview mode, use per-eye projection matrix for correct depth
    // Note: viewMat was already declared above for sensor pose computation
#ifdef MULTIVIEW_ENABLED
    const float4x4 projMat = (frameInfo.multiviewEnabled != 0) ? frameInfo.projectionMatrixArray[viewIndex] : frameInfo.projectionMatrix;
#else
    const float4x4 projMat = frameInfo.projectionMatrix;
#endif
    const float4 viewSplatCenter  = mul(worldSplatCenter, viewMat);
    // TODO: projection matrix is not correct with fisheye, this is a coarse aprox to compute depth
    const float4 clipSplatCenter = mul(viewSplatCenter, projMat);

    const float3 ndcCenter = float3(ndcXY, clipSplatCenter.z / clipSplatCenter.w);
#if EXTENT_METHOD == EXTENT_CONIC
    const float2 ndcExtent = extent * frameInfo.basisViewport * 2.0;
#endif

    /////////////////////////////
    // emiting quad vertices

    const float2 vertexPositions[4] = {{-1.0, -1.0}, {1.0, -1.0}, {1.0, 1.0}, {-1.0, 1.0}};

    [unroll]
    for(uint i = 0; i < 4; ++i)
    {
      const float2 vertPos = vertexPositions[i].xy;

#if EXTENT_METHOD == EXTENT_CONIC

      const float2 ndcOffset = vertPos * ndcExtent;

#elif EXTENT_METHOD == EXTENT_EIGEN

      const float2 ndcOffset = float2(vertPos.x * basisVector1 + vertPos.y * basisVector2) * frameInfo.basisViewport * 2.0
                             * frameInfo.inverseFocalAdjustment;

#endif
      const float4 quadPos = float4(ndcCenter.xy + ndcOffset, ndcCenter.z, 1.0);

      verts[localIndex * 4 + i].position = quadPos;
    }
  }
}