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

#include "nvshaders/random.h.slang"

#include "shaderio.h"
#include "cameras.h.slang"
#include "color.h.slang"
#include "wavefront.h.slang"

#include "threedgs_particles_storage.h.slang"
#include "threedgrt.h.slang"
#include "threedgrt_payload.h.slang"

// clang-format off
[[vk::binding(RTX_BINDING_TLAS_SPLATS, 1)]] RaytracingAccelerationStructure topLevelAS;
[[vk::binding(RTX_BINDING_OUTIMAGE, 1)]] RWTexture2D<float4> image;
[[vk::binding(RTX_BINDING_AUX1, 1)]] RWTexture2D<float4> imageAux;
[[vk::binding(RTX_BINDING_OUTDEPTH, 1)]] RWTexture2D<float> depthBuffer;
#if RTX_USE_MESHES
[[vk::binding(RTX_BINDING_TLAS_MESH, 1)]] RaytracingAccelerationStructure topLevelASMesh;
#endif
[[vk::binding(BINDING_FRAME_INFO_UBO, 0)]] ConstantBuffer<FrameInfo> frameInfo;
[[vk::push_constant]] ConstantBuffer<PushConstantRay> pcRay;
// TODO to be removed, just for debug readback
[[vk::binding(BINDING_INDIRECT_BUFFER, 0)]] RWStructuredBuffer<IndirectParams> indirectBuffer;
#if RTX_USE_MESHES
[[vk::binding(BINDING_MESH_DESCRIPTORS, 0)]] StructuredBuffer<ObjDesc> objDesc;
[[vk::binding(BINDING_LIGHT_SET, 0)]] StructuredBuffer<LightSource> lights;
#endif
#if defined(WITH_DLSS_RR) || defined(WITH_SPACE_WARP)
[[vk::binding(RTX_BINDING_DLSS_MOTION, 1)]] RWTexture2D<float2> dlssMotion;
#endif
#if RTX_USE_GLOBAL_KBUFFER
// GRTX optimization: Global SoA K-buffer for splat any-hit accumulation
// Structure-of-Arrays layout: buffer[slotIndex * numRays + rayIndex]
[[vk::binding(RTX_BINDING_KBUFFER_DIST, 1)]] RWStructuredBuffer<float> gKDist;
[[vk::binding(RTX_BINDING_KBUFFER_ID, 1)]] RWStructuredBuffer<int> gKId;
#endif

// clang-format on

double maxComponent(in double3 v)
{
  return max(v.x, max(v.y, v.z));
}

[shader("raygeneration")]
void main()
{
  float3 rayOrigin;
  float3 rayDirection;

  uint2 launchID   = DispatchRaysIndex().xy;
  uint2 launchSize = DispatchRaysDimensions().xy;
  
  // Determine eye index for stereo rendering
  // When multiviewEnabled, use X position to determine eye (SBS layout)
  // Otherwise use legacy viewportOffset for dual-pass rendering
  uint viewIndex = 0;
  uint2 outputPixel;
  uint2 rayPixel = launchID;  // Pixel coordinate for ray generation (per-eye local)
  uint2 rayViewportSize = launchSize;  // Viewport size for ray generation
  
#ifdef MULTIVIEW_ENABLED
  if(frameInfo.multiviewEnabled != 0)
  {
    // Single-pass SBS stereo: determine eye from X coordinate
    uint halfWidth = launchSize.x / 2;
    viewIndex = (launchID.x >= halfWidth) ? 1u : 0u;
    outputPixel = launchID;
    // Convert to per-eye local coordinate for ray generation
    rayPixel = uint2(launchID.x - viewIndex * halfWidth, launchID.y);
    rayViewportSize = uint2(halfWidth, launchSize.y);
  }
  else
#endif
  {
    outputPixel = launchID + uint2(pcRay.viewportOffset);
  }

  // Select appropriate view/projection matrices based on view index
  float4x4 viewMatrix;
  float4x4 viewInverseMatrix;
  float4x4 projMatrix;
  float4x4 projInverseMatrix;

#ifdef MULTIVIEW_ENABLED
  if(frameInfo.multiviewEnabled != 0)
  {
    viewMatrix = frameInfo.viewMatrixArray[viewIndex];
    viewInverseMatrix = frameInfo.viewInverseArray[viewIndex];
    projMatrix = frameInfo.projectionMatrixArray[viewIndex];
    projInverseMatrix = frameInfo.projInverseArray[viewIndex];
  }
  else
#endif
  {
    viewMatrix = frameInfo.viewMatrix;
    viewInverseMatrix = frameInfo.viewInverse;
    projMatrix = frameInfo.projectionMatrix;
    projInverseMatrix = frameInfo.projInverse;
  }

  // generate the ray for this pixel (using per-eye local coordinates)
#if CAMERA_TYPE == CAMERA_PINHOLE
  generatePinholeRay(rayPixel, float2(rayViewportSize), viewInverseMatrix, projInverseMatrix, rayOrigin, rayDirection);
#else
  if(!generateFisheyeRay(rayPixel, float2(rayViewportSize), frameInfo.fovRad, float2(0.0), viewInverseMatrix, rayOrigin, rayDirection))
  {
    image[outputPixel] = float4(0.0, 0.0, 0.0, 1.0);
    return;
  }
#endif

  // add Depth-of-Field perturbation to the ray
#if((PIPELINE != PIPELINE_HYBRID) && RTX_DOF_ENABLED)
  // Initialize the random number
  uint seed = xxhash32(uint3(launchID, frameInfo.frameSampleId));

  depthOfField(seed, frameInfo.focusDist, frameInfo.aperture, viewInverseMatrix, rayOrigin, rayDirection);
#endif

  // will use Any hit only
  uint rayFlags = RAY_FLAG_CULL_BACK_FACING_TRIANGLES | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER;
  // will use Closest hit only
  uint rayFlagsMesh = RAY_FLAG_CULL_BACK_FACING_TRIANGLES;

  float       tMax = 10000.0;
  const float epsT = 1e-9;

  int rayHitsCount = 0;

  float3  radiance      = float3(0.0f);
  double3 transmittance = float3(1.0f);

  bool meshHit = false;  // did we have a mesh hit by primary ray
  int  bounce  = 0;
  int  done    = 1;

  uint64_t clockStart = clockARB();

  // for debug feedback and alternative visu modes
  bool  closestParticleFound = false;
  float closestParticleDist  = PAYLOAD_INF_DISTANCE;
  int   closestParticleId    = PAYLOAD_INVALID_ID;

#if HYBRID_ENABLED
  float4 pixel;
  if(frameInfo.frameSampleId <= 0)  // no temporal or first frame
    pixel = image[outputPixel];
  else
    pixel = imageAux[outputPixel];
  radiance      = pixel.xyz;
  transmittance = float3(1.0 - pixel.w);
#endif

  // Create the payload
  HitPayload payload = {};

  // bounce loop
  while(true)
  {
    // each new bounce reinits Tmin
    float tMin               = 0.001;
    tMax                     = 10000.0;
    float rayLastHitDistance = max(0.0f, tMin - epsT);

    // trace the mesh with one closest hit if meshes are activated
    float meshHitDist = PAYLOAD_INF_DISTANCE;

#if RTX_USE_MESHES

    float3 meshHitWorldPos;
    float3 meshHitWorldNrm;
    int    meshHitObjId;
    int    meshHitMatId;

    payload.dist[0] = PAYLOAD_INF_DISTANCE;
    payload.id[0]   = PAYLOAD_INVALID_ID;
    payload.id[1]   = PAYLOAD_INVALID_ID;

// #define USE_SER
#ifdef USE_SER
    hitObjectNV hObj;
    hitObjectRecordEmptyNV(hObj);
    hitObjectTraceRayNV(hObj,                       // the hitObject instance
                        topLevelASMesh,             // acceleration structure
                        rayFlagsMesh,               // rayFlags
                        0xFF,                       // cullMask
                        0,                          // sbtRecordOffset
                        0,                          // sbtRecordStride
                        0,                          // missIndex
                        rayOrigin,                  // ray origin
                        rayLastHitDistance + epsT,  // ray min range
                        rayDirection,               // ray direction
                        tMax + epsT,                // ray max range
                        0                           // payload (location = 0)
    );
    reorderThreadNV(hObj);
    hitObjectExecuteShaderNV(hObj, 0);
#else
    RayDesc rayMesh;
    rayMesh.Origin    = rayOrigin;
    rayMesh.Direction = rayDirection;
    rayMesh.TMin      = rayLastHitDistance + epsT;
    rayMesh.TMax      = tMax + epsT;

    TraceRay(topLevelASMesh,  // acceleration structure
             rayFlagsMesh,    // rayFlags
             0xFF,            // cullMask
             0,               // sbtRecordOffset
             0,               // sbtRecordStride
             0,               // missIndex
             rayMesh,         // ray
             payload          // payload (location = 0)
    );
#endif
    meshHitDist = payload.dist[0];
    //
    if(meshHitDist < PAYLOAD_INF_DISTANCE)
    {
      tMax = meshHitDist;
      // collect info to compute shading and potential new ray later on
      meshHitObjId    = payload.id[0];
      meshHitMatId    = payload.id[1];
      meshHitWorldPos = float3(payload.dist[1], payload.dist[2], payload.dist[3]);
      meshHitWorldNrm = float3(payload.dist[4], payload.dist[5], payload.dist[6]);
      // Early return if meshDepthOnly pre-pass
      if(pcRay.meshDepthOnly)
      {
        const float4 clip = mul(mul(float4(meshHitWorldPos, 1.0), frameInfo.viewMatrix), frameInfo.projectionMatrix);
        const float3 ndc  = clip.xyz / clip.w;
        depthBuffer[outputPixel] = ndc.z;
        // clear image buffer
        if(frameInfo.frameSampleId <= 0)  // < 0 means temporal sampling off, 0 means first frame
          image[outputPixel] = float4(0.0, 0.0, 0.0, 1.0);
        else
          imageAux[outputPixel] = float4(0.0, 0.0, 0.0, 1.0);
        return;
      }
      //
      if(bounce == 0)
        meshHit = true;
    }
    else if(pcRay.meshDepthOnly)
    {
      // Early return if meshDepthOnly pre-pass
      depthBuffer[outputPixel] = 1.0;
      // clear image buffer
      if(frameInfo.frameSampleId <= 0)  // < 0 means temporal sampling off, 0 means first frame
        image[outputPixel] = float4(0.0, 0.0, 0.0, 1.0);
      else
        imageAux[outputPixel] = float4(0.0, 0.0, 0.0, 1.0);
      return;
    }
#endif

    // when hybrid is enabled, we trace the particles only for secondary rays
#if HYBRID_ENABLED
    if(bounce > 0)
    {
#endif
      // trace the gaussians with multi-pass any hit
      int outerIdx = 0;

      // The two following are to compute particleProcessHit with transformed splat set model
      const float3 splatSetModelRayOrigin = mul(float4(rayOrigin, 1.0), pcRay.modelMatrixInverse).xyz;
      // Since the ray direction should not be affected by translation,
      // uses the inverse of the rotation-scale part of the model matrix.
      const float3 splatSetModelRayDirection = normalize(mul(rayDirection, float3x3(pcRay.modelMatrixRotScaleInverse)));

#if RTX_USE_GLOBAL_KBUFFER
      // GRTX optimization: compute ray index for global K-buffer access
      const uint rayIndex = launchID.y * launchSize.x + launchID.x;
      const uint numRays = launchSize.x * launchSize.y;
      
      // Pass ray info to anyhit via minimal payload
      payload.rayIndex = rayIndex;
      payload.numRays = numRays;
#endif

      while(outerIdx < frameInfo.maxPasses && (rayLastHitDistance <= tMax)
            && (maxComponent(transmittance) > double(frameInfo.minTransmittance)))
      {
#if RTX_USE_GLOBAL_KBUFFER
        // Initialize global K-buffer for this ray
        [[unroll]]
        for(int i = 0; i < PAYLOAD_ARRAY_SIZE; ++i)
        {
          uint idx = getKBufferIndex(i, rayIndex, numRays);
          gKId[idx] = PAYLOAD_INVALID_ID;
          gKDist[idx] = PAYLOAD_INF_DISTANCE;
        }
#else
        // prepare the payload
        [[unroll]]
        for(int i = 0; i < PAYLOAD_ARRAY_SIZE; ++i)
        {
          payload.id[i]   = PAYLOAD_INVALID_ID;
          payload.dist[i] = PAYLOAD_INF_DISTANCE;
        }
#endif

        // trace the PAYLOAD_ARRAY_SIZE any hits
        // Use model-space ray since TLAS is built in model space with identity transform
        RayDesc raySplat;
        raySplat.Origin    = splatSetModelRayOrigin;
        raySplat.Direction = splatSetModelRayDirection;
        raySplat.TMin      = rayLastHitDistance + epsT;
        raySplat.TMax      = tMax + epsT;

        TraceRay(topLevelAS,  // acceleration structure
                 rayFlags,    // rayFlags
                 0xFF,        // cullMask
                 0,           // sbtRecordOffset
                 0,           // sbtRecordStride
                 0,           // missIndex
                 raySplat,    // ray
                 payload      // payload (location = 0)
        );

#if RTX_USE_GLOBAL_KBUFFER
        const int firstId = gKId[getKBufferIndex(0, rayIndex, numRays)];
#else
        const int firstId = payload.id[0];
#endif

        // no more hits found
        if(firstId == PAYLOAD_INVALID_ID)
        {
          break;
        }

        // evaluate the sorted hits
        [[unroll]]
        for(int i = 0; i < PAYLOAD_ARRAY_SIZE; ++i)
        {
#if RTX_USE_GLOBAL_KBUFFER
          uint idx = getKBufferIndex(i, rayIndex, numRays);
          const int   splatId = gKId[idx];
          const float dist    = gKDist[idx];
#else
          const int   splatId = payload.id[i];
          const float dist    = payload.dist[i];
#endif

          if((splatId != PAYLOAD_INVALID_ID) && (maxComponent(transmittance) > double(frameInfo.minTransmittance)))
          {
#if WIREFRAME
            // we render wireframe only for primary rays.
            if(bounce == 0)
            {
              const float2 uv        = payload.bary[i];
              const float3 bary      = float3(1.0 - uv.x - uv.y, uv.x, uv.y);
              const float  threshold = 0.02;
              if(bary.x < threshold || bary.y < threshold || bary.z < threshold)
              {
                radiance      = float3(1.0, 0.0, 0.0);  // wireframe color
                transmittance = float3(0.0);            // opaque
                break;
              }
            }
#endif  // end of display wireframe

            bool acceptedHit = particleProcessHit(frameInfo, splatSetModelRayOrigin, splatSetModelRayDirection, splatId,
                                                  transmittance, radiance);
            rayHitsCount += int(acceptedHit);

            // debug feedback and alternative visualizations
            if(!closestParticleFound && acceptedHit && outerIdx == 0)
            {
              closestParticleId    = splatId;
              closestParticleDist  = dist;
              closestParticleFound = true;
            }

            // we move on in any case
            rayLastHitDistance = max(rayLastHitDistance, dist);
          }
        }

        //
        outerIdx++;
      }

#if HYBRID_ENABLED
    }
#endif


#if RTX_USE_MESHES
    // process mesh shading if needed
    if((meshHitDist < PAYLOAD_INF_DISTANCE) && (maxComponent(transmittance) > double(frameInfo.minTransmittance)))
    {
      // retrieve the material for this hit
      ObjMaterial meshHitMaterial = objDesc[meshHitObjId].materialAddress[meshHitMatId];
      wavefrontComputeShading(lights, frameInfo.lightCount, meshHitWorldPos, meshHitWorldNrm, meshHitMaterial,
                              rayDirection, transmittance, radiance, done, rayOrigin, rayDirection);
    }
#endif

    // continue with secondary rays if needed (decided by wavefrontComputeShading)
    bounce++;
    if(done == 1 || bounce > frameInfo.rtxMaxBounces)
      break;

    // prepare ray for next iteration
    // new rayOrigin and direction was set by wavefrontComputeShading
    // Next iteration will stop if a reflective material isn't hit
    done = 1;

  }  // while(true)

  // debug feedback
  if(closestParticleFound && all(launchID == frameInfo.cursor))
  {
    indirectBuffer[0].particleID   = closestParticleId;
    indirectBuffer[0].particleDist = closestParticleDist;
  }

  //
  float3 fragRadiance = radiance;

  uint64_t clock = clockARB() - clockStart;

#if VISUALIZE == VISUALIZE_CLOCK
  fragRadiance = hsbToRgb(
      float3(lerp(0.65, 0.02, smoothstep(0.0, 1.0, frameInfo.multiplier * 0.1 * float(clock) / float(1 << 20))), 1.0, 1.));
#endif

#if VISUALIZE == VISUALIZE_DEPTH
  if(closestParticleFound)
  {
    fragRadiance = hsbToRgb(
        float3(lerp(0.65, 0.02, smoothstep(0.0, 1.0, frameInfo.multiplier * float(closestParticleDist) / 256.0F)), 1.0, 1.));
  }
#endif

#if VISUALIZE == VISUALIZE_RAYHITS
  fragRadiance = float3(frameInfo.multiplier * 10 * float(rayHitsCount) / 255, 0.0, 0.0);
#endif

#if VISUALIZE == VISUALIZE_FINAL
#if HYBRID_ENABLED
  if(meshHit)
  {
    if(frameInfo.frameSampleId <= 0)  // < 0 means temporal sampling off, 0 means first frame
      image[outputPixel] = float4(fragRadiance, 1.0);
    else
      imageAux[outputPixel] = float4(fragRadiance, 1.0);
  }
  return;
#else
  if(frameInfo.frameSampleId > 0)
  {
    // Do accumulation over time
    const float  a        = 1.0F / float(frameInfo.frameSampleId + 1);
    const float4 oldColor = image[outputPixel];
    fragRadiance          = lerp(oldColor, float4(fragRadiance, 1.0), a).xyz;
  }
#endif
#endif

// depending on previous build conditions this code might become unreachable (which is normal)
#pragma warning(disable: 41000)  // Disable warning warning 41000: unreachable code detected
  image[outputPixel] = float4(fragRadiance, 1.0);

#if defined(WITH_DLSS_RR) || defined(WITH_SPACE_WARP)
  // Motion vectors: compute screen-space motion from current and previous frame matrices

  // Since splat tracing now uses model-space rays, compute hit position in model space first,
  // then transform to world space for motion vector calculation.
  // Note: For XR locomotion, the model matrix changes frame-to-frame, but we don't track
  // prevModelMatrix, so motion vectors may be inaccurate during locomotion.
  // This is acceptable since DLSS-RR is typically disabled during rapid movement.
  const float3 splatSetModelRayOrigin = mul(float4(rayOrigin, 1.0), pcRay.modelMatrixInverse).xyz;
  const float3 splatSetModelRayDirection = normalize(mul(rayDirection, float3x3(pcRay.modelMatrixRotScaleInverse)));
  float3 modelPos = splatSetModelRayOrigin + splatSetModelRayDirection * closestParticleDist;
  float3 worldPos = mul(float4(modelPos, 1.0), pcRay.modelMatrix).xyz;
  
  float4 prevClip = mul(mul(float4(worldPos, 1.0), frameInfo.prevViewMatrix), frameInfo.prevProjectionMatrix);
  float2 prevNdc = prevClip.xy / prevClip.w;
  float4 currClip = mul(mul(float4(worldPos, 1.0), frameInfo.viewMatrix), frameInfo.projectionMatrix);
  float2 currNdc = currClip.xy / currClip.w;
  
  float2 motion;
  if (pcRay.useNdcMotion != 0)
  {
    // NDC space motion for Space Warp
    motion = (currNdc - prevNdc);
    // No jitter for Space Warp
  }
  else
  {
    // Screen space (pixel) motion for DLSS-RR
    motion = (currNdc - prevNdc) * frameInfo.viewport * 0.5;
    // Add jitter offset for temporal stability
    motion += frameInfo.dlssJitter;
  }
  
  dlssMotion[outputPixel] = closestParticleFound ? motion : float2(0.0);
  
#if defined(WITH_DLSS_RR) || defined(WITH_SPACE_WARP)
  // Linear depth: distance along the ray to the closest particle
  float linearDepth = closestParticleFound ? closestParticleDist : frameInfo.nearFar.y;
#ifdef WITH_DLSS_RR
  // dlssLinearDepth and dlssSpecHitDist are only bound when WITH_DLSS_RR is enabled
  // But they are not declared in this shader yet?
  // They are outputs: RTX_BINDING_DLSS_LINEAR_DEPTH and RTX_BINDING_DLSS_SPEC_HIT_DIST
#endif
  
#ifdef WITH_DLSS_RR
  // Specular hit distance: not applicable for diffuse Gaussian splats
#endif // WITH_DLSS_RR
#endif // WITH_DLSS_RR


#endif // WITH_DLSS_RR || WITH_SPACE_WARP
}
