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

// Chunk-based hierarchical frustum culling compute shader
// Tests chunk AABBs against frustum planes and outputs visible chunk indices

#include "shaderio.h"

// clang-format off
[[vk::binding(BINDING_FRAME_INFO_UBO, 0)]]           ConstantBuffer<FrameInfo>      frameInfo;
[[vk::push_constant]]                                ConstantBuffer<PushConstant>   pcRaster;
[[vk::binding(BINDING_CHUNK_BOUNDS_BUFFER, 0)]]      StructuredBuffer<ChunkBounds>  chunkBounds;
[[vk::binding(BINDING_VISIBLE_CHUNKS_BUFFER, 0)]]    RWStructuredBuffer<uint32_t>   visibleChunks;
[[vk::binding(BINDING_VISIBLE_CHUNK_COUNT_BUFFER, 0)]] RWStructuredBuffer<uint32_t> visibleChunkCount;
// clang-format on

// Test if an AABB is visible against a single frustum plane
// Returns true if the AABB is on the positive side (visible) of the plane
bool testPlaneAABB(float4 plane, float3 minPos, float3 maxPos)
{
  // Find the positive vertex (corner most aligned with plane normal)
  float3 pVertex;
  pVertex.x = (plane.x >= 0.0) ? maxPos.x : minPos.x;
  pVertex.y = (plane.y >= 0.0) ? maxPos.y : minPos.y;
  pVertex.z = (plane.z >= 0.0) ? maxPos.z : minPos.z;

  // Test the positive vertex against the plane
  float dist = dot(plane.xyz, pVertex) + plane.w;
  return dist >= 0.0;
}

// Test if an AABB is visible against all 6 frustum planes
bool isAABBVisible(float3 minPos, float3 maxPos)
{
  // Test against each frustum plane
  for (int i = 0; i < 6; i++)
  {
    if (!testPlaneAABB(frameInfo.frustumPlanes[i], minPos, maxPos))
    {
      return false;  // AABB is completely outside this plane
    }
  }
  return true;  // AABB is at least partially inside all planes
}

[numthreads(CHUNK_CULL_WORKGROUP_SIZE, 1, 1)]
[shader("compute")]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
  const uint chunkIdx = dispatchThreadID.x;
  
  // Early exit if this thread is beyond the chunk count
  if (chunkIdx >= frameInfo.numChunks)
    return;

  // Load chunk bounds
  ChunkBounds bounds = chunkBounds[chunkIdx];
  
  float3 localCenter = 0.5 * (bounds.minPos + bounds.maxPos);
  float3 localExtent = 0.5 * (bounds.maxPos - bounds.minPos);

  float3 worldCenter = mul(float4(localCenter, 1.0), pcRaster.modelMatrix).xyz;
  float3 axisX = mul(float4(localExtent.x, 0.0, 0.0, 0.0), pcRaster.modelMatrix).xyz;
  float3 axisY = mul(float4(0.0, localExtent.y, 0.0, 0.0), pcRaster.modelMatrix).xyz;
  float3 axisZ = mul(float4(0.0, 0.0, localExtent.z, 0.0), pcRaster.modelMatrix).xyz;
  float3 worldExtent = abs(axisX) + abs(axisY) + abs(axisZ);

  float3 actualMin = worldCenter - worldExtent;
  float3 actualMax = worldCenter + worldExtent;

  // Test against frustum
  if (isAABBVisible(actualMin, actualMax))
  {
    // Atomically add to visible chunk list
    uint outIdx;
    InterlockedAdd(visibleChunkCount[0], 1, outIdx);
    visibleChunks[outIdx] = chunkIdx;
  }
}
