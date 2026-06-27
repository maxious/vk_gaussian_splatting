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
 * Stochastic Gaussian Splatting - per-Gaussian accumulation kernel.
 *
 * One thread per Gaussian. Projects the splat to screen space, performs
 * stochastic acceptance (ST mode) or Gaussian Point Splatting (GPS mode),
 * and writes winning fragments via atomicMin on a 32-bit depth buffer.
 *
 * The 64-bit atomicMin on RWStructuredBuffer<uint64_t> is broken in
 * Slang/SPIR-V, so we use two separate 32-bit buffers:
 *   - depthBuffer: per-pixel depth for InterlockedMin
 *   - indexBuffer: per-pixel winning gaussian index (TOCTOU race acceptable
 *     for a stochastic renderer)
 *
 * Ported from the CUDA reference implementation (render.cu) from
 * https://github.com/fhahlbohm/stochastic-gaussian-splatting
 *
 * Portions derived from https://github.com/JorisAR/gaussian-point-splatting
 * Copyright (c) 2026, Joris Rijsdijk, licensed under the BSD 3-Clause License.
 * Modifications made by Florian Hahlbohm.
 */

#include "shaderio.h"
#include "stochasticgs.h.slang"
#include "threedgs_particles_storage.h.slang"

// clang-format off
[[vk::binding(BINDING_STOCHASTIC_DEPTH_BUFFER, 1)]]
RWStructuredBuffer<uint> depthBuffer;

[[vk::binding(BINDING_STOCHASTIC_INDEX_BUFFER, 1)]]
RWStructuredBuffer<uint> indexBuffer;

[[vk::binding(BINDING_FRAME_INFO_UBO, 0)]]
ConstantBuffer<FrameInfo>      frameInfo;

[[vk::push_constant]]
ConstantBuffer<PushConstant>    pcRaster;
// clang-format on

/*-------------------------------------------------------------------------------------------------
 * Configuration constants
 *-----------------------------------------------------------------------------------------------*/
static const uint STOCHASTIC_GS_WORKGROUP_SIZE       = 256u;
static const uint STOCHASTIC_GS_MAX_FRAGMENTS_PER_SPLAT = 32u;
static const uint STOCHASTIC_GS_N_SEQUENTIAL_THRESHOLD  = 4u;  // from CUDA reference (warp-level, informational)

/*-------------------------------------------------------------------------------------------------
 * ST mode (Stochastic Testing)
 *
 * For each pixel in the Gaussian's screen-space AABB, evaluate the Gaussian power,
 * convert to alpha, and perform a stochastic acceptance test. Accepted fragments
 * compete via 32-bit atomicMin on the depth buffer; the winner writes its
 * gaussian index to the index buffer.
 *-----------------------------------------------------------------------------------------------*/
void accumulateST(StochasticProjectedGaussian proj,
                  uint                          gaussian_idx,
                  uint                          frame,
                  uint                          fb_width,
                  uint                          fb_height,
                  uint                          depth_bits)
{
    // Clamp the screen-space AABB to the framebuffer bounds
    const int x_min = max(int(proj.screen_bounds.x), 0);
    const int x_max = min(int(proj.screen_bounds.y), int(fb_width) - 1);
    const int y_min = max(int(proj.screen_bounds.z), 0);
    const int y_max = min(int(proj.screen_bounds.w), int(fb_height) - 1);

    if(x_min > x_max || y_min > y_max)
        return;

    // PRNG state seeded per-Gaussian, per-frame
    uint2 state = make_seed(gaussian_idx, frame);

    for(int py = y_min; py <= y_max; ++py)
    {
        for(int px = x_min; px <= x_max; ++px)
        {
            // Offset from the splat center in pixel coordinates
            const float2 d = float2(float(px), float(py)) - proj.mean2d;

            // Gaussian power: -0.5 * d^T * Sigma^{-1} * d
            // conic = (cov_yy/det, -cov_xy/det, cov_xx/det)
            const float power = -0.5f * (proj.conic.x * d.x * d.x
                                       + 2.0f * proj.conic.y * d.x * d.y
                                       + proj.conic.z * d.y * d.y);

            // Skip pixels outside the power threshold (alpha < min_alpha_threshold)
            // power = -0.5 * mahalanobis² (always ≤ 0). Skip when |power| > power_threshold
            if(-power > proj.power_threshold)
                continue;

            // Alpha = opacity * exp(power)
            const float alpha = proj.opacity * exp(power);

            // Stochastic acceptance test
            const float2 rnd = pcg2d(state);
            if(rnd.x < alpha)
            {
                // Two-pass 32-bit atomic approach
                // 1. InterlockedMin on 32-bit depth buffer
                // 2. If we won (our depth was smaller), write our gaussian index to index buffer
                const uint pixel_id = uint(py) * fb_width + uint(px);
                uint oldDepth;
                InterlockedMin(depthBuffer[pixel_id], depth_bits, oldDepth);
                if(oldDepth > depth_bits)
                {
                    indexBuffer[pixel_id] = gaussian_idx;
                }
            }
        }
    }
}

/*-------------------------------------------------------------------------------------------------
 * GPS mode (Gaussian Point Splatting)
 *
 * Compute the expected number of fragments from the splat's importance (area * dilog(opacity)),
 * then sample that many points from the 2D Gaussian distribution using an importance-corrected
 * Box-Muller transform. Each sampled point competes via 32-bit atomicMin on the depth buffer.
 *-----------------------------------------------------------------------------------------------*/
void accumulateGPS(StochasticProjectedGaussian proj,
                   uint                          gaussian_idx,
                   uint                          frame,
                   uint                          fb_width,
                   uint                          fb_height,
                   uint                          depth_bits)
{
    // Importance = expected fragment count = 2*pi*sqrt(det(cov2d)) * dilog(opacity)
    const float det_cov    = max(proj.determinant, 0.0f);
    const float importance = STOCHASTIC_GS_TWO_PI * sqrt(det_cov) * dilog(proj.opacity);
    const uint  num_samples = min(uint(importance), STOCHASTIC_GS_MAX_FRAGMENTS_PER_SPLAT);

    if(num_samples == 0u)
        return;

    // Cholesky decomposition L of cov2d (lower triangular):
    //   cov2d = [[cov_xx, cov_xy], [cov_xy, cov_yy]]
    //   L = [[L00, 0], [L10, L11]]
    const float a   = proj.cov2d.x;  // cov_xx
    const float b   = proj.cov2d.y;  // cov_xy
    const float c   = proj.cov2d.z;  // cov_yy
    const float L00 = sqrt(max(a, 1.0e-20f));
    const float L10 = b / L00;
    const float L11 = sqrt(max(c - L10 * L10, 1.0e-20f));

    // Importance-corrected sampling parameters
    const float alpha_rcp    = 1.0f / proj.opacity;
    const float alpha_dilog = dilog(proj.opacity);

    // PRNG state seeded per-Gaussian, per-frame
    uint2 state = make_seed(gaussian_idx, frame);

    for(uint s = 0u; s < num_samples; ++s)
    {
        // Draw two uniform floats in [0, 1)
        const float2 u = pcg2d(state);

        // Importance-corrected Box-Muller: returns a 2D sample from the
        // standard normal, weighted by the opacity importance distribution.
        const float2 nd = corrected_box_muller(u.x, u.y, alpha_rcp, alpha_dilog);

        // Transform from standard normal to pixel coordinates:
        //   point = mean2d + L * nd
        const float2 point = proj.mean2d + float2(L00 * nd.x, L10 * nd.x + L11 * nd.y);

        // Snap to pixel center and check bounds
        const int px = int(point.x);
        const int py = int(point.y);
        if(px < 0 || px >= int(fb_width) || py < 0 || py >= int(fb_height))
            continue;

        const uint pixel_id = uint(py) * fb_width + uint(px);
        // Two-pass 32-bit atomic approach:
        uint oldDepth;
        InterlockedMin(depthBuffer[pixel_id], depth_bits, oldDepth);
        if(oldDepth > depth_bits)
        {
            indexBuffer[pixel_id] = gaussian_idx;
        }
    }
}

/*-------------------------------------------------------------------------------------------------
 * Main compute entry point.
 * Dispatch: ceil(splatCount / 256) groups in X, 1 in Y, 1 in Z.
 *-----------------------------------------------------------------------------------------------*/
[numthreads(256, 1, 1)]
[shader("compute")]
void main(uint3 id : SV_DispatchThreadID)
{
    const uint gaussian_idx = id.x;
    if(gaussian_idx >= uint(frameInfo.splatCount))
        return;

    // ---- Fetch Gaussian attributes from particle storage ----
    const float3 mean3d_obj  = fetchCenter(gaussian_idx);
    const float3 raw_scale   = fetchScale(gaussian_idx);
    const float4 raw_rotation = fetchRotation(gaussian_idx);
    const float4 color        = fetchColor(gaussian_idx);
    // fetchColor stores SIGMOID-activated opacity (applied on CPU during upload).
    // project_gaussian() expects RAW logit opacity and applies sigmoid internally.
    // Reverse the activation to avoid double-activation.
    const float activated_opacity = clamp(color.w, 1e-7f, 1.0f - 1e-7f);
    const float raw_opacity  = -log(1.0f / activated_opacity - 1.0f);

    // ---- Transform splat center to world space via the model matrix ----
    // Row-major convention: vector * matrix (matching threedgs.h.slang)
    const float3 mean3d = mul(float4(mean3d_obj, 1.0f), pcRaster.modelMatrix).xyz;

    // ---- Camera / projection parameters (SSAA-scaled) ----
    const float ssf      = float(frameInfo.stochasticSupersamplingFactor);
    const float width    = float(frameInfo.stochasticWidth);
    const float height   = float(frameInfo.stochasticHeight);
    const float focal_x   = frameInfo.focal.x * ssf;
    const float focal_y   = frameInfo.focal.y * ssf;
    const float center_x  = width * 0.5f + frameInfo.stereoShift.x * ssf;
    const float center_y  = height * 0.5f + frameInfo.stereoShift.y * ssf;

    // ---- Project the Gaussian to screen space ----
    StochasticProjectedGaussian proj;
    const bool visible = project_gaussian(
        mean3d,
        raw_scale,
        raw_rotation,
        raw_opacity,
        frameInfo.viewMatrix,
        width, height,
        focal_x, focal_y,
        center_x, center_y,
        frameInfo.nearFar.x, frameInfo.nearFar.y,
        false,  // proper_antialiasing (MipSplatting AA)
        ssf,    // supersampling_factor
        proj);

    if(!visible)
        return;

    // ---- Depth bits for 32-bit atomicMin ----
    // For positive view-space depths, the raw float bit pattern sorts correctly
    // under atomicMin (smaller depth = smaller uint bit pattern).
    const uint depth_bits = asuint(proj.depth);

    const uint fb_width  = uint(frameInfo.stochasticWidth);
    const uint fb_height = uint(frameInfo.stochasticHeight);
    const uint frame     = uint(frameInfo.stochasticFrameCounter);

    // ---- Dispatch to the selected accumulation mode ----
    if(frameInfo.stochasticUseGps != 0)
    {
        accumulateGPS(proj, gaussian_idx, frame, fb_width, fb_height, depth_bits);
    }
    else
    {
        accumulateST(proj, gaussian_idx, frame, fb_width, fb_height, depth_bits);
    }
}
