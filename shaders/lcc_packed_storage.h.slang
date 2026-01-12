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

/*
 * LCC Packed Storage - GPU-side decompression for LCC (Lixel CyberColor) format
 * 
 * LCC Portable format stores each splat in 32 bytes:
 *   - Position: 3 × float32 (12 bytes) at offset 0
 *   - Color+Alpha: uint32 RGBA (4 bytes) at offset 12
 *   - Scale: 3 × uint16 (6 bytes) at offset 16 - quantized, requires min/max interpolation
 *   - Rotation: uint32 (4 bytes) at offset 22 - encoded quaternion with LUT
 *   - Padding: 6 bytes at offset 26
 * 
 * This approach uploads raw packed data to GPU and decompresses in shader,
 * matching the performance strategy of the LCC WebGL reference implementation.
 */

#ifndef _LCC_PACKED_STORAGE_H_
#define _LCC_PACKED_STORAGE_H_

// Note: This header is included from threedgs_particles_storage.h.slang which
// already includes shaderio.h, so BINDING_LCC_PACKED_BUFFER is defined.

// LCC packed data buffer - each splat is 32 bytes (8 × uint32)
[[vk::binding(BINDING_LCC_PACKED_BUFFER, 0)]] RWStructuredBuffer<uint> lccPackedBuffer;

// Constants for LCC decoding
static const float LCC_COLOR_NORM = 1.0 / 255.0;
static const float LCC_SCALE_NORM = 1.0 / 65535.0;
static const float LCC_ROT_NORM = 1.0 / 1023.0;
static const float LCC_SQRT_2 = 1.414213562373095;
static const float LCC_SQRT_2_INV = 0.7071067811865475;

// Quaternion reconstruction LUT (matches LCC spec)
// For each mode (0-3), defines which component slot gets which decoded value
// Mode indicates which quaternion component was dropped (largest magnitude)
static const int4 LCC_QUAT_LUT[4] = {
    int4(3, 0, 1, 2),  // Mode 0: w was largest, reconstruct w from xyz
    int4(0, 3, 1, 2),  // Mode 1: x was largest, reconstruct x from wyz
    int4(0, 1, 3, 2),  // Mode 2: y was largest, reconstruct y from wxz
    int4(0, 1, 2, 3)   // Mode 3: z was largest, reconstruct z from wxy
};

// Decode LCC packed position (3 × float32 at offset 0)
float3 lccDecodePosition(in uint splatIndex)
{
    uint baseOffset = splatIndex * 8;  // 32 bytes = 8 × uint32
    
    float x = asfloat(lccPackedBuffer[baseOffset + 0]);
    float y = asfloat(lccPackedBuffer[baseOffset + 1]);
    float z = asfloat(lccPackedBuffer[baseOffset + 2]);
    
    return float3(x, y, z);
}

// Decode LCC packed color+alpha (uint32 RGBA at offset 12)
float4 lccDecodeColor(in uint splatIndex)
{
    uint baseOffset = splatIndex * 8;
    uint colorVal = lccPackedBuffer[baseOffset + 3];  // offset 12 = uint32 index 3
    
    uint r = (colorVal) & 0xFF;
    uint g = (colorVal >> 8) & 0xFF;
    uint b = (colorVal >> 16) & 0xFF;
    uint a = (colorVal >> 24) & 0xFF;
    
    return float4(float(r), float(g), float(b), float(a)) * LCC_COLOR_NORM;
}

// Decode LCC packed scale (3 × uint16 at offset 16)
// Returns log-scale values compatible with existing shader pipeline (exp() is applied later)
float3 lccDecodeScale(in uint splatIndex, in float3 scaleMin, in float3 scaleMax)
{
    uint baseOffset = splatIndex * 8;
    
    // uint32 at offset 16 contains scale.x (low 16 bits) and scale.y (high 16 bits)
    uint scaleXY = lccPackedBuffer[baseOffset + 4];
    // uint32 at offset 20 contains scale.z (low 16 bits) and part of rotation (high 6 bits, but rotation starts at offset 22)
    uint scaleZRot = lccPackedBuffer[baseOffset + 5];
    
    // Extract uint16 scale values
    uint sx16 = scaleXY & 0xFFFF;
    uint sy16 = (scaleXY >> 16) & 0xFFFF;
    uint sz16 = scaleZRot & 0xFFFF;
    
    // Interpolate to actual scale range
    float3 t = float3(float(sx16), float(sy16), float(sz16)) * LCC_SCALE_NORM;
    float3 scale = lerp(scaleMin, scaleMax, t);
    
    // Convert to log-scale (existing shaders apply exp() to scale)
    return log(max(scale, float3(1e-7, 1e-7, 1e-7)));
}

// Decode LCC packed rotation (uint32 at offset 22)
// Returns normalized quaternion (w, x, y, z)
float4 lccDecodeRotation(in uint splatIndex)
{
    uint baseOffset = splatIndex * 8;
    
    // Rotation is at byte offset 22, which spans uint32 indices 5 and 6
    // offset 20: [scale.z (16 bits)] [rot bits 0-15]
    // offset 24: [rot bits 16-31] [padding]
    uint word5 = lccPackedBuffer[baseOffset + 5];
    uint word6 = lccPackedBuffer[baseOffset + 6];
    
    // Extract rotation uint32 from byte offset 22
    // Byte 22-25 spans: high 16 bits of word5, low 16 bits of word6
    uint rotVal = ((word5 >> 16) & 0xFFFF) | ((word6 & 0xFFFF) << 16);
    
    // Extract 3 × 10-bit components and 2-bit mode
    uint a = rotVal & 0x3FF;
    uint b = (rotVal >> 10) & 0x3FF;
    uint c = (rotVal >> 20) & 0x3FF;
    uint mode = (rotVal >> 30) & 0x3;
    
    // Normalize to [0, 1] then transform to [-0.707, 0.707]
    float3 abc = float3(float(a), float(b), float(c)) * LCC_ROT_NORM;
    abc = abc * LCC_SQRT_2 - LCC_SQRT_2_INV;
    
    // Compute the 4th component (the one that was dropped)
    float sumSquares = dot(abc, abc);
    float d = sqrt(max(0.0, 1.0 - sumSquares));
    
    // Reconstruct quaternion based on mode using LUT
    float4 q;
    int4 lut = LCC_QUAT_LUT[mode];
    
    // The LUT tells us: position i gets component lut[i]
    // Components are: 0=a, 1=b, 2=c, 3=d (reconstructed)
    float components[4] = {abc.x, abc.y, abc.z, d};
    
    q.x = components[lut.x];  // w
    q.y = components[lut.y];  // x  
    q.z = components[lut.z];  // y
    q.w = components[lut.w];  // z
    
    // Normalize quaternion
    return normalize(q);
}

// Convert LCC RGBA color to SH DC coefficient format
// LCC stores linear RGB in [0,1], convert to SH0 space
float3 lccColorToShDc(in float3 rgb)
{
    const float SH_C0 = 0.28209479177387814;
    return (rgb - 0.5) / SH_C0;
}

// Convert LCC alpha to logit opacity (inverse sigmoid)
// LCC stores alpha in [0,1], existing shaders expect logit space
float lccAlphaToOpacity(in float alpha)
{
    // Clamp to avoid infinity
    alpha = clamp(alpha, 0.001, 0.999);
    return log(alpha / (1.0 - alpha));
}

//////////////////////////////////////////////
// High-level fetch functions matching existing API

// Fetch center position for LCC packed storage
float3 fetchCenterLcc(in uint splatIndex)
{
    return lccDecodePosition(splatIndex);
}

// Fetch scale for LCC packed storage
float3 fetchScaleLcc(in uint splatIndex, in float3 scaleMin, in float3 scaleMax)
{
    return lccDecodeScale(splatIndex, scaleMin, scaleMax);
}

// Fetch rotation for LCC packed storage  
float4 fetchRotationLcc(in uint splatIndex)
{
    return lccDecodeRotation(splatIndex);
}

// Fetch color+opacity for LCC packed storage
// Returns float4(r, g, b, opacity) where opacity is in logit space
float4 fetchColorLcc(in uint splatIndex)
{
    float4 rgba = lccDecodeColor(splatIndex);
    float3 shDc = lccColorToShDc(rgba.rgb);
    float opacity = lccAlphaToOpacity(rgba.a);
    
    // Return in format compatible with existing fetchColor():
    // rgb = SH DC coefficient, a = logit opacity
    return float4(shDc.r * 0.28209479177387814 + 0.5,
                  shDc.g * 0.28209479177387814 + 0.5,
                  shDc.b * 0.28209479177387814 + 0.5,
                  opacity);
}

#endif // _LCC_PACKED_STORAGE_H_
