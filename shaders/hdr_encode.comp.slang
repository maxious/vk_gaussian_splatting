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

// HDR10 Encoding Compute Shader
// Converts linear BT.709 HDR to BT.2020 with PQ (ST.2084) transfer function
// Based on Microsoft DirectXTK implementation (MIT License)

#include "shaderio.h"

// clang-format off
[[vk::binding(BINDING_FRAME_INFO_UBO, 0)]] ConstantBuffer<FrameInfo> frameInfo;
[[vk::binding(HDR_BINDING_INPUT_IMAGE, 0)]] Texture2D<float4>        inputImage;
[[vk::binding(HDR_BINDING_OUTPUT_IMAGE, 0)]] RWTexture2D<float4>     outputImage;
// clang-format on

// BT.709 to BT.2020 color space conversion matrix
// This rotates from Rec.709 primaries to Rec.2020 primaries
static const float3x3 k709to2020 = float3x3(
    0.6274040f, 0.3292820f, 0.0433136f,
    0.0690970f, 0.9195400f, 0.0113612f,
    0.0163916f, 0.0880132f, 0.8955950f
);

// Apply ST.2084 (PQ) curve to normalized linear values
// Input: linear values normalized to [0,1] where 1.0 = 10000 nits
// Output: PQ-encoded values in [0,1]
float3 linearToST2084(float3 normalizedLinear)
{
    // ST.2084 constants
    const float m1 = 0.1593017578125f;   // 2610 / 16384
    const float m2 = 78.84375f;          // 2523 / 32
    const float c1 = 0.8359375f;         // 3424 / 4096
    const float c2 = 18.8515625f;        // 2413 / 128
    const float c3 = 18.6875f;           // 2392 / 128

    float3 Lm1 = pow(abs(normalizedLinear), float3(m1));
    float3 numerator = c1 + c2 * Lm1;
    float3 denominator = 1.0f + c3 * Lm1;
    
    return pow(numerator / denominator, float3(m2));
}

// Convert linear BT.709 HDR to HDR10 (BT.2020 + PQ)
float3 encodeHDR10(float3 linearBT709, float paperWhiteNits, float exposure)
{
    // Apply exposure
    float3 color = linearBT709 * exposure;
    
    // Rotate from BT.709 to BT.2020 color primaries
    float3 colorBT2020 = mul(k709to2020, color);
    
    // Normalize to PQ range (10000 nits max)
    // paperWhiteNits defines what linear 1.0 maps to in nits
    float3 normalizedNits = colorBT2020 * paperWhiteNits / 10000.0f;
    
    // Clamp to valid range before PQ encoding
    normalizedNits = saturate(normalizedNits);
    
    // Apply ST.2084 (PQ) curve
    return linearToST2084(normalizedNits);
}

[numthreads(16, 16, 1)]
[shader("compute")]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const int2 pixel = int2(dispatchThreadID.xy);
    
    if (pixel.x >= int(frameInfo.viewport.x) || pixel.y >= int(frameInfo.viewport.y))
        return;
    
    // Read linear HDR color (BT.709)
    float4 hdrColor = inputImage[pixel];
    
    // Encode to HDR10
    float3 hdr10 = encodeHDR10(
        hdrColor.rgb,
        frameInfo.hdrPaperWhiteNits,
        frameInfo.hdrExposure
    );
    
    outputImage[pixel] = float4(hdr10, hdrColor.a);
}
