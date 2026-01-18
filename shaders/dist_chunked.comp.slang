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

// Chunk-aware distance compute shader
// Processes only splats from visible chunks when chunk culling is enabled
// Falls back to standard per-splat processing when disabled

#include "shaderio.h"
#include "threedgs_particles_storage.h.slang"
#include "threedgut_camera_models.h.slang"
#include "threedgut_camera_projections.h.slang"

// clang-format off
[[vk::binding(BINDING_FRAME_INFO_UBO, 0)]]               ConstantBuffer<FrameInfo>        frameInfo;
[[vk::push_constant]]                                    ConstantBuffer<PushConstant>     pcRaster;
[[vk::binding(BINDING_DISTANCES_BUFFER, 0)]]             RWStructuredBuffer<uint32_t>     distances;
[[vk::binding(BINDING_INDICES_BUFFER, 0)]]               RWStructuredBuffer<uint32_t>     indices;
[[vk::binding(BINDING_INDIRECT_BUFFER, 0)]]              RWStructuredBuffer<IndirectParams> indirectBuffer;
[[vk::binding(BINDING_VISIBLE_CHUNKS_BUFFER, 0)]]        StructuredBuffer<uint32_t>       visibleChunks;
[[vk::binding(BINDING_VISIBLE_CHUNK_COUNT_BUFFER, 0)]]   StructuredBuffer<uint32_t>       visibleChunkCount;
// clang-format on

// Encodes an fp32 into a uint32 that can be ordered
uint encodeMinMaxFp32(float val)
{
  uint bits = asuint(val);
  bits ^= (int(bits) >> 31) | 0x80000000u;
  return bits;
}

[numthreads(DISTANCE_COMPUTE_WORKGROUP_SIZE, 1, 1)]
[shader("compute")]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
  uint id;
  
  if (frameInfo.chunkCullingEnabled != 0)
  {
    // Chunk culling mode: compute which splat this thread processes
    // Each visible chunk contributes CHUNK_SIZE splats
    const uint totalVisibleChunks = visibleChunkCount[0];
    const uint localIdx = dispatchThreadID.x;
    
    // Compute which visible chunk and which splat within that chunk
    const uint chunkLocalIdx = localIdx / CHUNK_SIZE;
    const uint splatInChunk = localIdx % CHUNK_SIZE;
    
    if (chunkLocalIdx >= totalVisibleChunks)
      return;
    
    // Get the actual chunk index from the visible chunk list
    const uint actualChunkIdx = visibleChunks[chunkLocalIdx];
    
    // Compute the actual splat ID
    id = actualChunkIdx * CHUNK_SIZE + splatInChunk;
    
    // Bounds check for the last chunk which may be partial
    if (id >= frameInfo.splatCount)
      return;
  }
  else
  {
    // Standard mode: one thread per splat
    id = dispatchThreadID.x;
    if (id >= frameInfo.splatCount)
      return;
  }

  float4 splatPos = float4(fetchCenter(id), 1.0);

  // Row-major: multiply in reverse order (vector * matrix * matrix)
  float4 viewPos = mul(mul(splatPos, pcRaster.modelMatrix), frameInfo.viewMatrix);
  // TODO: projection matrix is not correct with fisheye, this is a coarse aprox to compute depth
  float4 ndcPos = mul(viewPos, frameInfo.projectionMatrix);
  ndcPos = ndcPos / ndcPos.w;
  const float depth = ndcPos.z;

#if FRUSTUM_CULLING_MODE == FRUSTUM_CULLING_AT_DIST
  // When using chunk culling, we can skip per-splat frustum culling for splats
  // that passed the chunk-level test (conservative AABB culling)
  // But we still apply fine-grained culling if chunk culling is disabled
  if (frameInfo.chunkCullingEnabled == 0)
  {
#if CAMERA_TYPE == CAMERA_PINHOLE
    // valid only when center is inside NDC clip space.
    const float clip = 1.0f + frameInfo.frustumDilation;
    if (abs(ndcPos.x) > clip || abs(ndcPos.y) > clip || ndcPos.z < 0.f - frameInfo.frustumDilation || ndcPos.z > 1.0)
      return;
#else
    // Fisheye camera
    CameraModelParameters sensorModel = initPerfectFisheyeCamera(frameInfo.viewport, frameInfo.focal);

    // Fisheye projection to perform culling
    float2 projected;
    bool isValid = projectPointFisheye(sensorModel.ocvFisheyeParams, frameInfo.viewport,
                                       float3(1.0, 1.0, -1.0) * viewPos.xyz, GUT_IN_IMAGE_MARGIN_FACTOR, projected);

    if (!isValid)
      return;

    if (ndcPos.z < 0.f - frameInfo.frustumDilation || ndcPos.z > 1.0)
      return;
#endif
  }
  else
  {
    // With chunk culling enabled, still do depth check but skip XY frustum test
    // since chunk AABB already passed frustum test
    if (ndcPos.z < 0.f - frameInfo.frustumDilation || ndcPos.z > 1.0)
      return;
  }
#endif

  // Increments the visible splat counter in the indirect buffer
  uint original;
  InterlockedAdd(indirectBuffer[0].instanceCount, 1, original);
  const uint instance_index = original;
  
  // Stores the distance
  distances[instance_index] = encodeMinMaxFp32(-depth);
  
  // Stores the base index
  indices[instance_index] = id;
  
  // Set the workgroup count for the mesh shading pipeline
  if (instance_index % RASTER_MESH_WORKGROUP_SIZE == 0)
  {
    InterlockedAdd(indirectBuffer[0].groupCountX, 1);
  }
}
