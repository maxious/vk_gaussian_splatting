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

#include "stochasticgs.h.slang"

// clang-format off
[[vk::binding(BINDING_STOCHASTIC_FRAMEBUFFER_SSBO, 1)]] RWStructuredBuffer<uint64_t> framebufferSSBO;
[[vk::binding(BINDING_FRAME_INFO_UBO, 0)]]             ConstantBuffer<FrameInfo>      frameInfo;
// clang-format on

[numthreads(256, 1, 1)]
[shader("compute")]
void main(uint3 id : SV_DispatchThreadID)
{
  uint totalPixels = uint(frameInfo.stochasticWidth) * uint(frameInfo.stochasticHeight);
  if(id.x >= totalPixels)
    return;
  framebufferSSBO[id.x] = 0xFFFFFFFFFFFFFFFF;
}
