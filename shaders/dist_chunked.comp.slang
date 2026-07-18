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

groupshared uint sVisibleCount;
groupshared uint sBaseInstanceIndex;

[numthreads(DISTANCE_COMPUTE_WORKGROUP_SIZE, 1, 1)]
[shader("compute")]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupThreadID : SV_GroupThreadID)
{
  const uint localIndex = groupThreadID.x;
  if(localIndex == 0)
  {
    sVisibleCount = 0;
    sBaseInstanceIndex = 0;
  }
  GroupMemoryBarrierWithGroupSync();

  uint  id = 0;
  bool  idValid = false;
  bool  isVisible = false;
  float depth = 0.0;

  if(frameInfo.chunkCullingEnabled != 0)
  {
    const uint totalVisibleChunks = visibleChunkCount[0];
    const uint localDispatchIdx = dispatchThreadID.x;

    const uint chunkLocalIdx = localDispatchIdx / CHUNK_SIZE;
    const uint splatInChunk = localDispatchIdx % CHUNK_SIZE;

    if(chunkLocalIdx < totalVisibleChunks)
    {
      const uint actualChunkIdx = visibleChunks[chunkLocalIdx];
      id = actualChunkIdx * CHUNK_SIZE + splatInChunk;
      idValid = id < frameInfo.splatCount;
    }
  }
  else
  {
    id = dispatchThreadID.x;
    idValid = id < frameInfo.splatCount;
  }

  if(idValid)
  {
    float4 splatPos = float4(fetchCenter(id), 1.0);

#if CAMERA_TYPE == CAMERA_PINHOLE
    // OPTIMIZATION: use precomputed modelViewProj matrix (modelView * projection, stored in modelMatrixInverse)
    float4 ndcPos = mul(splatPos, pcRaster.modelMatrixInverse);
#else
    // Fisheye: need viewPos for projectPointFisheye, so use separate modelView + projection
    float4 viewPos = mul(splatPos, pcRaster.modelMatrixRotScaleInverse);
    float4 ndcPos = mul(viewPos, frameInfo.projectionMatrix);
#endif
    ndcPos = ndcPos / ndcPos.w;
    depth = ndcPos.z;
    isVisible = true;

#if FRUSTUM_CULLING_MODE == FRUSTUM_CULLING_AT_DIST
    if(frameInfo.chunkCullingEnabled == 0)
    {
#if CAMERA_TYPE == CAMERA_PINHOLE
      const float clip = 1.0f + frameInfo.frustumDilation;
      if(abs(ndcPos.x) > clip || abs(ndcPos.y) > clip || ndcPos.z < 0.f - frameInfo.frustumDilation || ndcPos.z > 1.0)
        isVisible = false;
#else
      CameraModelParameters sensorModel = initPerfectFisheyeCamera(frameInfo.viewport, frameInfo.focal);

      float2 projected;
      bool isValid = projectPointFisheye(sensorModel.ocvFisheyeParams, frameInfo.viewport,
                                         float3(1.0, 1.0, -1.0) * viewPos.xyz, GUT_IN_IMAGE_MARGIN_FACTOR, projected);

      if(!isValid)
        isVisible = false;

      if(ndcPos.z < 0.f - frameInfo.frustumDilation || ndcPos.z > 1.0)
        isVisible = false;
#endif
    }
    else
    {
      if(ndcPos.z < 0.f - frameInfo.frustumDilation || ndcPos.z > 1.0)
        isVisible = false;
    }
#endif
  }

  uint localSlot = 0;
  if(isVisible)
  {
    InterlockedAdd(sVisibleCount, 1, localSlot);
  }

  GroupMemoryBarrierWithGroupSync();

  if(localIndex == 0 && sVisibleCount > 0)
  {
    InterlockedAdd(indirectBuffer[0].instanceCount, sVisibleCount, sBaseInstanceIndex);

    const uint prevGroupCount = (sBaseInstanceIndex + RASTER_MESH_WORKGROUP_SIZE - 1) / RASTER_MESH_WORKGROUP_SIZE;
    const uint nextGroupCount = (sBaseInstanceIndex + sVisibleCount + RASTER_MESH_WORKGROUP_SIZE - 1) / RASTER_MESH_WORKGROUP_SIZE;
    const uint groupDelta = nextGroupCount - prevGroupCount;
    if(groupDelta > 0)
    {
      InterlockedAdd(indirectBuffer[0].groupCountX, groupDelta);
    }
  }

  GroupMemoryBarrierWithGroupSync();

  if(isVisible)
  {
    const uint instance_index = sBaseInstanceIndex + localSlot;
    distances[instance_index] = encodeMinMaxFp32(-depth);
    indices[instance_index] = id;
  }
}
