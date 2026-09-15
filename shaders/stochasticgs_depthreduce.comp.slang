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
 *
 * Stochastic Gaussian Splatting - previous-frame depth reduce kernel.
 *
 * Folds the per-pixel stochastic depth buffer into one max depth per 8 px block.
 * The next stochastic frame reprojects each splat through the previous model-view
 * and culls it when its front lies beyond the farthest surviving sample. This is
 * the occlusion cull from supersplat PR #1048.
 *
 * Dispatch: ceil(blockCols * blockRows / 64) groups in X.
 */

#include "shaderio.h"

// clang-format off
[[vk::binding(BINDING_FRAME_INFO_UBO, 0)]]
ConstantBuffer<FrameInfo> frameInfo;

[[vk::binding(BINDING_STOCHASTIC_DEPTH_BUFFER, 1)]]
StructuredBuffer<uint> depthBuffer;

[[vk::binding(BINDING_STOCHASTIC_OCCLUSION_BUFFER, 1)]]
RWStructuredBuffer<uint> occlusionBuffer;
// clang-format on

static const uint STOCHASTIC_GS_OCCLUSION_BLOCK   = 8u;
static const uint STOCHASTIC_GS_OCCLUSION_INVALID = 0xFFFFFFFFu;

[numthreads(64, 1, 1)]
[shader("compute")]
void main(uint3 id : SV_DispatchThreadID)
{
    const uint cols = uint(frameInfo.stochasticOcclusionBlockCols);
    const uint rows = uint(frameInfo.stochasticOcclusionBlockRows);
    const uint blockIdx = id.x;
    if(cols == 0u || rows == 0u || blockIdx >= cols * rows)
        return;

    const uint width  = uint(frameInfo.stochasticWidth);
    const uint height = uint(frameInfo.stochasticHeight);

    const uint bx = blockIdx % cols;
    const uint by = blockIdx / cols;

    const uint x0 = bx * STOCHASTIC_GS_OCCLUSION_BLOCK;
    const uint y0 = by * STOCHASTIC_GS_OCCLUSION_BLOCK;
    const uint x1 = min(x0 + STOCHASTIC_GS_OCCLUSION_BLOCK, width);
    const uint y1 = min(y0 + STOCHASTIC_GS_OCCLUSION_BLOCK, height);

    uint maxDepth = 0u;
    for(uint y = y0; y < y1; ++y)
    {
        for(uint x = x0; x < x1; ++x)
        {
            const uint d = depthBuffer[y * width + x];
            if(d != STOCHASTIC_GS_OCCLUSION_INVALID)
                maxDepth = max(maxDepth, d);
        }
    }

    // 0xFFFFFFFF marks a block with no surviving sample so the gather can ignore it.
    occlusionBuffer[blockIdx] = (maxDepth == 0u) ? STOCHASTIC_GS_OCCLUSION_INVALID : maxDepth;
}
