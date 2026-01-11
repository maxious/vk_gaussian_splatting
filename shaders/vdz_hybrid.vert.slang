/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless REQUIRED BY APPLICABLE LAW OR AGREED TO IN WRITING, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// Hybrid Vertex Shader: Combines divided mesh displacement with POM setup
// - Performs depth-based vertex displacement (coarse depth structure)
// - Computes tangent-space view direction for POM (fine detail)
// - Outputs data for fragment shader to apply POM ray-marching

#include "shaderio.h"

// clang-format off
[[vk::binding(BINDING_FRAME_INFO_UBO, 0)]] ConstantBuffer<FrameInfo> frameInfo;
[[vk::binding(BINDING_VDZ_DEPTH_TEXTURE, 0)]] Sampler2D<float4> vdzDepthTexture;
[[vk::push_constant]] ConstantBuffer<PushConstant> pcRaster;

// Helper function to unpack RGB-packed 16-bit depth to float
float unpackPackedDepth(float4 color)
{
    uint16_t uint16Val = (uint16_t(color.r * 255.0) << 8) | uint16_t(color.g * 255.0);
    float scale = (frameInfo.vdzZMax > frameInfo.vdzZMin) ? (frameInfo.vdzZMax - frameInfo.vdzZMin) / 65535.0 : 1.0;
    return float(uint16Val) * scale + frameInfo.vdzZMin;
}

struct VertexInput
{
  [[vk::location(0)]] float3 position : POSITION;  // xy in [0,1], z unused
  [[vk::location(1)]] float2 uv : TEXCOORD0;       // [0,1]
};

struct VertexOutput
{
  [[vk::location(0)]] float2 uv : TEXCOORD0;        // Original UV coordinates
  [[vk::location(1)]] float2 depthUV : TEXCOORD1;   // Depth texture UV (may differ in hybrid modes)
  [[vk::location(2)]] float3 viewDirTS : TEXCOORD2; // Tangent-space view direction for POM
  [[vk::location(3)]] float3 normalWS : TEXCOORD3;  // World-space normal (flat surface normal)
  float4 position : SV_Position;
};
// clang-format on

[shader("vertex")]
VertexOutput main(VertexInput input, uint viewIndex : SV_ViewID)
{
  VertexOutput output;
  
  float2 uv = input.uv;
  output.uv = uv;
  output.depthUV = uv;
  
  // Sample depth from VDZ texture for mesh displacement
  float4 depthColor = vdzDepthTexture.SampleLevel(uv, 0);
  float depth = unpackPackedDepth(depthColor);
  
  // Apply gamma correction (same as existing VDZ shader)
  depth = max(depth, 0.0);
  depth = pow(depth, frameInfo.vdzZGamma);
  
  // Apply scaling and clipping
  float zDepth = clamp(depth * frameInfo.vdzZScale, 0.0, frameInfo.vdzZMaxClip);
  float z = zDepth + frameInfo.vdzZBias;
  
  // Normalize depth to [0, 1] range for parallax calculation
  float minD = frameInfo.vdzZMin;
  float maxD = frameInfo.vdzZMax > minD ? frameInfo.vdzZMax : minD + 1.0;
  float normalizedDepth = clamp((depth - minD) / (maxD - minD), 0.0, 1.0);
  
  // Calculate mesh displacement (coarse depth structure)
  float2 parallaxOffset = float2(0.0, 0.0);
  float meshDisplacementZ = z;
  
  // Only apply mesh parallax displacement in hybrid mode
  if (frameInfo.vdzHybridMode == 2 && frameInfo.vdzHybridMeshStrength > 0.0 &&
      (abs(frameInfo.vdzParallaxOffset.x) > 0.0 || abs(frameInfo.vdzParallaxOffset.y) > 0.0))
  {
    float focusZ = frameInfo.vdzParallaxFocus;
    float depthDiff = normalizedDepth - focusZ;
    
    // Parallax displacement
    parallaxOffset = frameInfo.vdzParallaxOffset * frameInfo.vdzParallaxStrength * 
                     frameInfo.vdzHybridMeshStrength * depthDiff * float2(-1.0, 1.0);
    
    // Edge softening
    float edgeWidth = frameInfo.vdzParallaxEdgeSoftness;
    float2 edgeFactor = smoothstep(float2(0.0, 0.0), float2(edgeWidth, edgeWidth), uv) * 
                       smoothstep(float2(1.0, 1.0), float2(1.0 - edgeWidth, 1.0 - edgeWidth), uv);
    parallaxOffset *= edgeFactor.x * edgeFactor.y;
  }
  
  // Compute tangent-space view direction for POM
  // For a flat surface, tangent space is aligned with world space
  // T = (1, 0, 0), B = (0, 1, 0), N = (0, 0, 1)
  // So we can compute view direction in world space and use it directly
  
  // Calculate view direction from camera to vertex position
  float3 cameraPos = frameInfo.cameraPositionArray[viewIndex];
  
  // Build camera-aligned plane position (same as existing VDZ shader)
  float planeScale = frameInfo.vdzPlaneScale;
  float aspect = frameInfo.vdzAspect;
  
  float x = (0.5 - uv.x) * aspect * planeScale + parallaxOffset.x;
  float y = (0.5 - uv.y) * planeScale + parallaxOffset.y;
  
  // View-space position: camera looks down -Z
  float3 viewPosLocal = float3(x, y, -z);
  
  // Transform to world space if in world-space mode
  float3 worldPos;
  if (frameInfo.vdzWorldSpaceMode != 0)
  {
    float4 worldPos4 = mul(float4(viewPosLocal, 1.0), frameInfo.vdzModelMatrix);
    worldPos = worldPos4.xyz;
  }
  else
  {
    // In camera-attached mode, worldPos is just viewPosLocal transformed by view matrix inverse
    // For a flat surface directly in front of camera, viewPosLocal is already in view space
    // We need the view direction in "tangent space" which for our flat quad is aligned with the plane
    worldPos = viewPosLocal;
  }
  
  // View direction in tangent space (for POM)
  // For a flat quad facing the camera, tangent space is aligned with world space
  // View direction points from surface toward camera
  float3 viewDirTS = normalize(cameraPos - worldPos);
  
  // For POM, we need view direction in tangent space where Z points out of the surface
  // Our surface normal is (0, 0, 1) in view space, so viewDirTS is already suitable
  // We just need to ensure the Z component is positive (pointing toward viewer)
  output.viewDirTS = viewDirTS;
  output.normalWS = float3(0.0, 0.0, 1.0); // Flat surface normal
  
  // Transform to clip space
  float4 viewPos;
  if (frameInfo.vdzWorldSpaceMode != 0)
  {
    float4 worldPos4 = mul(float4(viewPosLocal, 1.0), frameInfo.vdzModelMatrix);
    viewPos = mul(worldPos4, mul(frameInfo.viewMatrix, frameInfo.projectionMatrix));
  }
  else
  {
    viewPos = mul(float4(viewPosLocal, 1.0), frameInfo.projectionMatrix);
  }
  
  output.position = viewPos;
  
  return output;
}
