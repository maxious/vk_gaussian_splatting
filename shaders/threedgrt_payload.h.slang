/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "shaderio.h"

static const int PAYLOAD_INVALID_ID   = -1;
static const float PAYLOAD_INF_DISTANCE = asfloat(0x7F7FFFFF);

// GRTX optimization: Global SoA K-buffer accessors
// When RTX_USE_GLOBAL_KBUFFER is enabled, the K-buffer data is stored in global memory
// using Structure-of-Arrays layout for coalesced access: buffer[slotIndex * numRays + rayIndex]
// This reduces ray payload register pressure and improves occupancy.

#if RTX_USE_GLOBAL_KBUFFER

// Helper functions to access K-buffer with SoA layout
// slotIndex: which slot in the K-buffer (0 to PAYLOAD_ARRAY_SIZE-1)
// rayIndex: linear index of the ray (y * width + x)
// numRays: total number of rays (width * height)

static uint getKBufferIndex(uint slotIndex, uint rayIndex, uint numRays)
{
    return slotIndex * numRays + rayIndex;
}

#endif // RTX_USE_GLOBAL_KBUFFER
