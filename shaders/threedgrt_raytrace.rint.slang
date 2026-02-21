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
#include "threedgs_particles_storage.h.slang"
#include "threedgrt.h.slang"

// clang-format off
[[vk::binding(BINDING_FRAME_INFO_UBO, 0)]] ConstantBuffer<FrameInfo> frameInfo;
[[vk::push_constant]] ConstantBuffer<PushConstantRay> pcRay;
// clang-format on

bool particleDensityHitCustom(in float3   rayOrigin,
                              in float3   rayDirection,
                              in int    particleId,
                              in float  minHitDistance,
                              in float  maxHitDistance,
                              in float  maxParticleSquaredDistance,
                              out float hitDistance)
{
  float2 timeData = fetchTime(particleId);
  float t_scale = timeData.x; // JS Order: x = scale
  float t_center = timeData.y; // JS Order: y = center
  float dt = frameInfo.currentTime - t_center;

  if (frameInfo.temporalCulling != 0 && t_center > -5.0) {
      if (t_scale > 1e-5) {
          float v = dt / t_scale;
          if (v * v > 11.042922) {
              hitDistance = 0.0;
              return false;
          }
      } else {
          if (abs(dt) > 1e-5) {
              hitDistance = 0.0;
              return false;
          }
      }
  }

  float3 particlePosition;
  float3 particleScale;
  float3x3 particleInvRotation;

  fetchParticlePSR(frameInfo, particleId, particlePosition, particleScale, particleInvRotation);

  float3 canonicalRayOrigin;
  float3 canonicalRayDirection;
  particleCannonicalRay(rayOrigin, rayDirection, particlePosition, particleScale, particleInvRotation,
                        canonicalRayOrigin, canonicalRayDirection);

  hitDistance = particleRayDistance(canonicalRayOrigin, canonicalRayDirection, particleScale);

  return (hitDistance > minHitDistance) && (hitDistance < maxHitDistance)
         && (particleRayMinSquaredDistance(canonicalRayOrigin, canonicalRayDirection) < maxParticleSquaredDistance);
}

bool particleDensityHitInstance(in float3   canonicalRayOrigin,
                                in float3   canonicalUnormalizedRayDirection,
                                in float  minHitDistance,
                                in float  maxHitDistance,
                                in float  maxParticleSquaredDistance,
                                out float hitDistance)
{
  const float numerator   = -dot(canonicalRayOrigin, canonicalUnormalizedRayDirection);
  const float denominator = dot(canonicalUnormalizedRayDirection, canonicalUnormalizedRayDirection);
  hitDistance             = numerator / denominator;
  return (hitDistance > minHitDistance) && (hitDistance < maxHitDistance)
         && (particleRayMinSquaredDistance(canonicalRayOrigin, normalize(canonicalUnormalizedRayDirection)) < maxParticleSquaredDistance);
}

[shader("intersection")]
void main()
{
  float hitT;

#if RTX_USE_SPHERES
  // Blackwell native sphere primitives - hardware handles ray-sphere intersection
  // We still need intersection shader to compute the density t_closest (Gaussian center)
  // not the surface hit t that hardware provides
  
  // For sphere primitives, ObjectRayOrigin/Direction are already in unit sphere space
  // The hardware has already tested ray-sphere intersection at radius 3.0
  // We compute t_closest = point on ray closest to sphere center (Gaussian peak)
  if (particleDensityHitInstance(ObjectRayOrigin(), ObjectRayDirection(), RayTMin(), RayTCurrent(), 9.0F, hitT)) {
      BuiltInTriangleIntersectionAttributes attr;
      attr.barycentrics = float2(0.0, 0.0);
      ReportHit(hitT, 0, attr);
  }
#elif RTX_USE_INSTANCES
  // AABB mode with instances - testing unit particle transformed by instance
  if (particleDensityHitInstance(ObjectRayOrigin(), ObjectRayDirection(), RayTMin(), RayTCurrent(), 9.0F, hitT)) {
      BuiltInTriangleIntersectionAttributes attr;
      attr.barycentrics = float2(0.0, 0.0); // TODO: providing barycentrics will help rendering wireframe
      ReportHit(hitT, 0, attr);
  }
#else
  const int particleId = PrimitiveIndex();

  // Ray is already in model space (transformed before TraceRay in rgen shader)
  // Use WorldRayOrigin/Direction directly since the rgen shader now traces in model space
  const float3 modelRayOrigin = WorldRayOrigin();
  const float3 modelRayDirection = WorldRayDirection();

  if (particleDensityHitCustom(modelRayOrigin, modelRayDirection, particleId, RayTMin(), RayTCurrent(), 9.0F, hitT)) {
      BuiltInTriangleIntersectionAttributes attr;
      attr.barycentrics = float2(0.0, 0.0); // TODO: providing barycentrics will help rendering wireframe
      ReportHit(hitT, 0, attr);
  }
#endif
}