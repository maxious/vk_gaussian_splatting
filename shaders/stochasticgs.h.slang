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
 * Shared helper functions for the stochastic Gaussian Splatting rendering pipeline.
 * Ported from the CUDA reference implementation (render.cu, math_utils.h, gs_utils.h)
 * from https://github.com/fhahlbohm/stochastic-gaussian-splatting
 *
 * Portions derived from https://github.com/JorisAR/gaussian-point-splatting
 * Copyright (c) 2026, Joris Rijsdijk, licensed under the BSD 3-Clause License.
 * Modifications made by Florian Hahlbohm.
 */

#ifndef STOCHASTICGS_H_SLANG
#define STOCHASTICGS_H_SLANG

#include "nvshaders/random.h.slang"  // for xxhash32 seeding
#include "threedgs.h.slang"          // for threedgsCovarianceProjection (covariance projection math)

/*-------------------------------------------------------------------------------------------------
 * Configuration constants (from gs_utils.h - Faster-GS inherited config)
 *-----------------------------------------------------------------------------------------------*/
static const float STOCHASTIC_GS_TWO_PI              = 2.0 * 3.14159265358979323846;
static const float STOCHASTIC_GS_DILATION            = 0.3;
static const float STOCHASTIC_GS_DILATION_PROPER_AA   = 0.1;
static const float STOCHASTIC_GS_MIN_COV2D_DET        = 1e-6;
static const float STOCHASTIC_GS_MIN_ALPHA_THRESH_RCP = 255.0;
static const float STOCHASTIC_GS_MIN_ALPHA_THRESHOLD  = 1.0 / STOCHASTIC_GS_MIN_ALPHA_THRESH_RCP;  // 0.00392156862
static const float STOCHASTIC_GS_MAX_POWER_THRESHOLD  = 5.54126354516;  // ln(min_alpha_threshold_rcp)

/*-------------------------------------------------------------------------------------------------
 * Spherical harmonics constants (from gs_utils.h)
 *-----------------------------------------------------------------------------------------------*/
static const float SH_C0  = 0.28209479177387814f;
// SH_C1 may also be defined by threedgs_particles_storage.h.slang (same value,
// different trailing digits). Guard against redefinition when both headers are
// included in the same translation unit (e.g. stochasticgs_resolve.comp.slang).
#ifndef SH_C1_DEFINED
#define SH_C1_DEFINED
static const float SH_C1  = 0.48860251190291987f;
#endif
static const float SH_C2a = 1.0925484305920792f;
static const float SH_C2b = 0.94617469575755997f;
static const float SH_C2c = 0.31539156525251999f;
static const float SH_C2d = 0.54627421529603959f;
static const float SH_C3a = 0.59004358992664352f;
static const float SH_C3b = 1.7701307697799304f;
static const float SH_C3c = 2.8906114426405538f;
static const float SH_C3d = 0.45704579946446572f;
static const float SH_C3e = 2.2852289973223288f;
static const float SH_C3f = 1.865881662950577f;
static const float SH_C3g = 1.1195289977703462f;
static const float SH_C3h = 1.4453057213202769f;

/*-------------------------------------------------------------------------------------------------
 * 64-bit fragment packing / unpacking
 *
 * The framebuffer stores (depth_key, gaussian_idx) packed into a single uint64_t per pixel.
 * The depth_key is the raw bit pattern of the float depth, enabling atomicMin-based depth sorting.
 *-----------------------------------------------------------------------------------------------*/
uint64_t pack_fragment(uint depth_bits, uint gaussian_idx)
{
    return (uint64_t(depth_bits) << 32) | uint64_t(gaussian_idx);
}

uint2 unpack_fragment(uint64_t frag)
{
    return uint2(uint(frag >> 32), uint(frag & 0xFFFFFFFFu));
}

/*-------------------------------------------------------------------------------------------------
 * Integer hash (from math_utils.h)
 *-----------------------------------------------------------------------------------------------*/
uint stochastic_hash32(uint x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

/*-------------------------------------------------------------------------------------------------
 * make_seed - generate a 2D PRNG seed from a Gaussian index and frame number (from math_utils.h)
 *-----------------------------------------------------------------------------------------------*/
uint2 make_seed(uint idx, uint frame)
{
    uint h1 = stochastic_hash32(idx ^ (frame * 0x9E3779B9u));
    uint h2 = stochastic_hash32((idx * 0xBB67AE85u) ^ frame);
    h1 ^= (h2 * 0x3C6EF372u);
    h2 ^= (h1 * 0xA54FF53Au);
    return uint2(h1, h2);
}

/*-------------------------------------------------------------------------------------------------
 * pcg2d_uint - BIT-EXACT port from CUDA math_utils.h
 *
 * Uses LCG constants 1664525u and 1013904223u (Numerical Recipes LCG).
 * This is a 2D PCG hash that scrambles both components of the state vector.
 *-----------------------------------------------------------------------------------------------*/
uint2 pcg2d_uint(inout uint2 state)
{
    state.x = state.x * 1664525u + 1013904223u;
    state.y = state.y * 1664525u + 1013904223u;
    state.x += state.y * 1664525u;
    state.y += state.x * 1664525u;
    state.x ^= state.x >> 16u;
    state.y ^= state.y >> 16u;
    state.x += state.y * 1664525u;
    state.y += state.x * 1664525u;
    state.x ^= state.x >> 16u;
    state.y ^= state.y >> 16u;
    return state;
}

/*-------------------------------------------------------------------------------------------------
 * pcg2d - generate two uniform floats in [0, 1) from the PRNG state (from math_utils.h)
 *-----------------------------------------------------------------------------------------------*/
float2 pcg2d(inout uint2 seed)
{
    seed = pcg2d_uint(seed);
    return float2(float(seed.x), float(seed.y)) * 2.3283064365387e-10f;
}

/*-------------------------------------------------------------------------------------------------
 * pcg_hash - single-state PCG hash (from math_utils.h)
 *
 * Uses magic numbers 747796405u, 2891336453u, 277803737u (PCG32 LCG constants).
 * This is the same algorithm as nvshaders/random.h.slang `pcg()` but provided here
 * as a stateless hash for the stochastic GS `rand()` function.
 *-----------------------------------------------------------------------------------------------*/
uint pcg_hash(uint seed)
{
    const uint state = seed * 747796405u + 2891336453u;
    const uint word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

/*-------------------------------------------------------------------------------------------------
 * stochastic_rand - per-pixel random float in [0, 1) (from math_utils.h `rand`)
 *
 * Renamed to avoid collision with nvshaders/random.h.slang `rand()`.
 * Uses pcg_hash with a composite seed from frame, instance, and pixel coordinates.
 *-----------------------------------------------------------------------------------------------*/
float stochastic_rand(uint frame, uint instance, uint x, uint y)
{
    const uint seed = instance ^ (frame * 1973u) ^ (x * 9277u) ^ (y * 2699u);
    return asfloat(0x3f800000u | (pcg_hash(seed) >> 9u)) - 1.0f;
}

/*-------------------------------------------------------------------------------------------------
 * dilog - polynomial approximation of the dilogarithm Li_2(x) (from math_utils.h)
 *
 * Degree-9 polynomial (8 coefficients in Horner form, then multiplied by x).
 * The final term adds s * log(s) where s = 1 - x, for the analytic continuation.
 *-----------------------------------------------------------------------------------------------*/
float dilog(float x)
{
    float y = -0.952623943f;
    y = fma(y, x, 2.995873777f);
    y = fma(y, x, -3.762115247f);
    y = fma(y, x, 2.334605281f);
    y = fma(y, x, -0.773941317f);
    y = fma(y, x, 5.97506396e-02f);
    y = fma(y, x, -0.256617940f);
    y = fma(y, x, 2.000061866f);
    y *= x;
    const float s = 1.0f - x;
    if(s > 0.0f)
        y += s * log(max(s, 1.0e-37f));
    return y;
}

/*-------------------------------------------------------------------------------------------------
 * inv_dilog - inverse polynomial approximation of the dilogarithm (from math_utils.h)
 *
 * Degree-10 polynomial in t = x / (pi^2/6), evaluated in Horner form.
 *-----------------------------------------------------------------------------------------------*/
float inv_dilog(float x)
{
    const float t = min(x / 1.6449340668482264f, 1.0f);
    float y = -1.27463503e+01f;
    y = fma(y, t, 5.88993459e+01f);
    y = fma(y, t, -1.16025780e+02f);
    y = fma(y, t, 1.26945827e+02f);
    y = fma(y, t, -8.43108826e+01f);
    y = fma(y, t, 3.48799862e+01f);
    y = fma(y, t, -8.89606235e+00f);
    y = fma(y, t, 1.38640936e+00f);
    y = fma(y, t, -7.80640876e-01f);
    y = fma(y, t, 1.64841888e+00f);
    y = fma(y, t, -2.82836687e-05f);
    return y;
}

/*-------------------------------------------------------------------------------------------------
 * box_muller - standard Box-Muller transform for Gaussian sampling (from math_utils.h)
 *
 * Maps two uniform floats to a 2D sample from the standard normal distribution.
 *-----------------------------------------------------------------------------------------------*/
float2 box_muller(float u1, float u2)
{
    const float R     = sqrt(max(-2.0f * log(max(min(u1, 1.0f), 1.0e-37f)), 0.0f));
    const float theta = STOCHASTIC_GS_TWO_PI * u2;
    float s, c;
    sincos(theta, s, c);
    return float2(R * c, R * s);
}

/*-------------------------------------------------------------------------------------------------
 * corrected_box_muller - importance-corrected Box-Muller transform (from math_utils.h)
 *
 * Uses the inverse dilogarithm to map the uniform sample u1 through the opacity-weighted
 * importance distribution before applying the Box-Muller transform.
 *-----------------------------------------------------------------------------------------------*/
float2 corrected_box_muller(float u1, float u2, float alpha_rcp, float alpha_dilog)
{
    const float a     = alpha_rcp * inv_dilog((1.0f - u1) * alpha_dilog);
    const float R     = sqrt(max(-2.0f * log(max(min(a, 1.0f), 1.0e-37f)), 0.0f));
    const float theta = STOCHASTIC_GS_TWO_PI * u2;
    float s, c;
    sincos(theta, s, c);
    return float2(R * c, R * s);
}

/*-------------------------------------------------------------------------------------------------
 * poisson - Poisson-distributed sample count via Cornish-Fisher expansion (from math_utils.h)
 *
 * Generates a Poisson(lambda) random variate using a normal approximation with
 * higher-order correction terms (Cornish-Fisher quantile expansion).
 *-----------------------------------------------------------------------------------------------*/
uint poisson(inout uint2 state, float lambda)
{
    const float2 u  = pcg2d(state);
    const float  w  = box_muller(u.x, u.y).x;
    const float  w2 = w * w;
    const float  w3 = w2 * w;
    const float  w4 = w2 * w2;
    const float  s     = sqrt(lambda);
    const float  s_rcp = 1.0f / s;
    const float  l_rcp = 1.0f / lambda;
    float kf = lambda + s * w + (w2 - 1.0f) / 6.0f;
    kf += s_rcp * (-(1.0f / 36.0f) * w - (1.0f / 72.0f) * w3);
    kf += l_rcp * (-(8.0f / 405.0f) + (7.0f / 810.0f) * w2 + (1.0f / 270.0f) * w4);
    return uint(round(max(kf, 0.0f)));
}

/*-------------------------------------------------------------------------------------------------
 * 3D covariance computation from scale and rotation (ported from gs_utils.h project_gaussian)
 *
 * Computes the 3D covariance matrix: Sigma = R * diag(exp(2*scale)) * R^T / norm_sq
 * where R is the rotation matrix from the (possibly unnormalized) quaternion.
 *
 * The quaternion is stored as (r, i, j, k) = (w, x, y, z).
 *-----------------------------------------------------------------------------------------------*/
float3x3 stochasticgs_covariance3d(float3 raw_scale, float4 raw_rotation)
{
    const float3 variance = exp(2.0f * raw_scale);

    const float r = raw_rotation.x, i = raw_rotation.y, j = raw_rotation.z, k = raw_rotation.w;
    const float ii = i * i, jj = j * j, kk = k * k;
    const float ij = i * j, ik = i * k, jk = j * k;
    const float ri = r * i, rj = r * j, rk = r * k;
    const float norm_sq     = r * r + ii + jj + kk;
    const float norm_sq_rcp = 1.0f / norm_sq;

    // Rotation matrix R (normalized by norm_sq)
    float3x3 R = float3x3(
        1.0f - 2.0f * (jj + kk) * norm_sq_rcp, 2.0f * (ij - rk) * norm_sq_rcp, 2.0f * (ik + rj) * norm_sq_rcp,
        2.0f * (ij + rk) * norm_sq_rcp, 1.0f - 2.0f * (ii + kk) * norm_sq_rcp, 2.0f * (jk - ri) * norm_sq_rcp,
        2.0f * (ik - rj) * norm_sq_rcp, 2.0f * (jk + ri) * norm_sq_rcp, 1.0f - 2.0f * (ii + jj) * norm_sq_rcp);

    // R * diag(variance)
    float3x3 RSS = float3x3(
        R[0][0] * variance.x, R[0][1] * variance.y, R[0][2] * variance.z,
        R[1][0] * variance.x, R[1][1] * variance.y, R[1][2] * variance.z,
        R[2][0] * variance.x, R[2][1] * variance.y, R[2][2] * variance.z);

    // cov3d = RSS * R^T (upper-triangular portion in CUDA, full matrix here)
    float3x3 cov3d = mul(RSS, transpose(R));
    return cov3d;
}

/*-------------------------------------------------------------------------------------------------
 * Projected Gaussian result struct
 *-----------------------------------------------------------------------------------------------*/
struct StochasticProjectedGaussian
{
    float  depth;            // View-space depth of the splat center
    float  opacity;          // Sigmoid-activated opacity (possibly AA-compensated)
    float  determinant;      // Determinant of the 2D covariance (after dilation)
    float  power_threshold;  // ln(opacity * 255) - cutoff power for alpha threshold
    float2 mean2d;           // 2D splat center in screen (pixel) coordinates
    float3 conic;            // Inverse 2D covariance: (cov_zz/det, -cov_yz/det, cov_xx/det)
    float3 cov2d;            // 2D covariance: (cov_xx, cov_xy, cov_yy) after dilation
    float4 screen_bounds;    // Screen-space bounding box: (x_min, x_max, y_min, y_max)
};

/*-------------------------------------------------------------------------------------------------
 * project_gaussian - load, transform, and project a Gaussian splat (from gs_utils.h)
 *
 * This function wraps threedgs.h.slang's threedgsCovarianceProjection for the 2D covariance
 * projection math, while porting the stochastic-GS-specific logic (cov3d from scale/rotation,
 * EWA clipping, dilation, antialiasing, screen bounds) from the CUDA reference.
 *
 * No CUDA warp intrinsics (ballot, shfl, __fns, __popc) are used - this is a per-thread function.
 * Returns false if the Gaussian should be culled (behind near plane, too transparent, etc.).
 *
 * Parameters:
 *   mean3d         - 3D world-space center of the Gaussian
 *   raw_scale      - Log-scale parameters (3 floats)
 *   raw_rotation   - Rotation quaternion (r, i, j, k) = (w, x, y, z)
 *   raw_opacity    - Raw opacity (pre-sigmoid)
 *   w2c            - World-to-camera (view) matrix (4x4, row-vector convention matching threedgs.h.slang)
 *   width, height  - Framebuffer dimensions in pixels
 *   focal_x/y      - Focal lengths in pixels
 *   center_x/y     - Principal point in pixels
 *   near_plane     - Near clipping distance
 *   far_plane      - Far clipping distance
 *   proper_antialiasing - Whether to use MipSplatting-style alpha compensation
 *   supersampling_factor - SSAA factor (affects dilation kernel size)
 *   result         - Output struct with projected Gaussian parameters
 *-----------------------------------------------------------------------------------------------*/
bool project_gaussian(
    float3 mean3d,
    float3 raw_scale,
    float4 raw_rotation,
    float raw_opacity,
    float4x4 w2c,
    float width,
    float height,
    float focal_x,
    float focal_y,
    float center_x,
    float center_y,
    float near_plane,
    float far_plane,
    bool proper_antialiasing,
    float supersampling_factor,
    out StochasticProjectedGaussian result)
{
    // Transform splat center to view space
    const float4 viewCenter = mul(float4(mean3d, 1.0f), w2c);
    result.depth = viewCenter.z;

    // Z culling
    if(result.depth < near_plane || result.depth > far_plane)
        return false;

    // Load and activate opacity
    result.opacity = 1.0f / (1.0f + exp(-raw_opacity));
    if(result.opacity < STOCHASTIC_GS_MIN_ALPHA_THRESHOLD)
        return false;

    // Compute 3D covariance from scale and rotation
    const float3 raw_scale_local = raw_scale;
    const float4 raw_rot_local   = raw_rotation;
    const float  r = raw_rot_local.x, i = raw_rot_local.y, j = raw_rot_local.z, k = raw_rot_local.w;
    const float ii = i * i, jj = j * j, kk = k * k;
    const float ij = i * j, ik = i * k, jk = j * k;
    const float ri = r * i, rj = r * j, rk = r * k;
    const float norm_sq = r * r + ii + jj + kk;
    if(norm_sq < 1e-8f)
        return false;
    const float norm_sq_rcp = 1.0f / norm_sq;

    const float3 variance = exp(2.0f * raw_scale_local);

    float3x3 R = float3x3(
        1.0f - 2.0f * (jj + kk) * norm_sq_rcp, 2.0f * (ij - rk) * norm_sq_rcp, 2.0f * (ik + rj) * norm_sq_rcp,
        2.0f * (ij + rk) * norm_sq_rcp, 1.0f - 2.0f * (ii + kk) * norm_sq_rcp, 2.0f * (jk - ri) * norm_sq_rcp,
        2.0f * (ik - rj) * norm_sq_rcp, 2.0f * (jk + ri) * norm_sq_rcp, 1.0f - 2.0f * (ii + jj) * norm_sq_rcp);

    float3x3 RSS = float3x3(
        R[0][0] * variance.x, R[0][1] * variance.y, R[0][2] * variance.z,
        R[1][0] * variance.x, R[1][1] * variance.y, R[1][2] * variance.z,
        R[2][0] * variance.x, R[2][1] * variance.y, R[2][2] * variance.z);

    float3x3 cov3d = mul(RSS, transpose(R));

    // Compute normalized image coordinates (perspective division)
    const float x = viewCenter.x / result.depth;
    const float y = viewCenter.y / result.depth;

    // EWA splatting: clip the normalized coordinates to a slightly expanded screen region
    // This prevents extreme Jacobian values at the edges of the field of view
    const float clip_left   = (-0.15f * width - center_x) / focal_x;
    const float clip_right  = (1.15f * width - center_x) / focal_x;
    const float clip_top    = (-0.15f * height - center_y) / focal_y;
    const float clip_bottom = (1.15f * height - center_y) / focal_y;
    const float x_clipped   = clamp(x, clip_left, clip_right);
    const float y_clipped   = clamp(y, clip_top, clip_bottom);

    // Build the clipped view-space center for the covariance projection.
    // threedgsCovarianceProjection expects view-space coordinates (before perspective division),
    // so we multiply the clipped normalized coordinates back by depth.
    const float4 clippedViewCenter = float4(x_clipped * result.depth, y_clipped * result.depth, result.depth, 1.0f);

    // Reuse threedgs.h.slang for the 2D covariance projection (EWA splatting)
    // Returns float3(cov_xx, cov_xy, cov_yy)
    result.cov2d = threedgsCovarianceProjection(cov3d, clippedViewCenter, float2(focal_x, focal_y), w2c);

    // Apply dilation kernel (low-pass filter to avoid aliasing)
    const float determinant_raw = result.cov2d.x * result.cov2d.z - result.cov2d.y * result.cov2d.y;
    const float kernel_size = (proper_antialiasing ? STOCHASTIC_GS_DILATION_PROPER_AA : STOCHASTIC_GS_DILATION)
                              * supersampling_factor * supersampling_factor;
    result.cov2d.x += kernel_size;
    result.cov2d.z += kernel_size;
    result.determinant = result.cov2d.x * result.cov2d.z - result.cov2d.y * result.cov2d.y;
    if(result.determinant < STOCHASTIC_GS_MIN_COV2D_DET)
        return false;

    // Conic (inverse covariance) for evaluating the Gaussian power
    result.conic = float3(
        result.cov2d.z / result.determinant,
        -result.cov2d.y / result.determinant,
        result.cov2d.x / result.determinant);

    // MipSplatting alpha compensation
    if(proper_antialiasing)
    {
        result.opacity *= sqrt(max(determinant_raw / result.determinant, 0.0f));
        if(result.opacity < STOCHASTIC_GS_MIN_ALPHA_THRESHOLD)
            return false;
    }

    // 2D mean in screen (pixel) coordinates
    result.mean2d = float2(x * focal_x + center_x, y * focal_y + center_y);

    // Compute screen-space bounding box from the power threshold
    result.power_threshold = log(result.opacity * STOCHASTIC_GS_MIN_ALPHA_THRESH_RCP);
    const float cutoff_factor = 2.0f * result.power_threshold;
    const float extent_x = sqrt(result.cov2d.x * cutoff_factor);
    const float extent_y = sqrt(result.cov2d.z * cutoff_factor);
    result.screen_bounds = float4(
        result.mean2d.x - extent_x,  // x_min
        result.mean2d.x + extent_x,  // x_max
        result.mean2d.y - extent_y,  // y_min
        result.mean2d.y + extent_y); // y_max

    // Cull if bounding box is entirely off-screen
    if(result.screen_bounds.x >= width || result.screen_bounds.y < 0.0f
       || result.screen_bounds.z >= height || result.screen_bounds.w < 0.0f)
        return false;

    return true;
}

/*-------------------------------------------------------------------------------------------------
 * eval_sh_at - spherical harmonics evaluation at degree 0/1/2/3 (from gs_utils.h eval_sh)
 *
 * Evaluates the SH-encoded view-dependent color for a Gaussian splat.
 * The coefficients array contains `total_sh_bases` float3 values (one per SH basis function,
 * each storing RGB). For degree 3, this is 16 coefficients (48 floats total).
 *
 * Parameters:
 *   coefficients  - Array of SH coefficients (up to 15 float3 values for degree 3)
 *   direction     - Normalized direction from camera to splat center
 *   total_sh_bases - Number of SH basis functions (1, 4, 9, or 16 for degree 0-3)
 *
 * Returns the evaluated RGB color, clamped to [0, +inf).
 *-----------------------------------------------------------------------------------------------*/
float3 eval_sh_at(float3 coefficients[16], float3 direction, uint total_sh_bases)
{
    const float x = direction.x, y = direction.y, z = direction.z;

    // Degree 0 (DC component)
    float3 color = 0.5f + SH_C0 * coefficients[0];

    if(total_sh_bases > 1u)
    {
        // Degree 1
        color = color - SH_C1 * y * coefficients[1]
                      + SH_C1 * z * coefficients[2]
                      - SH_C1 * x * coefficients[3];

        if(total_sh_bases > 4u)
        {
            const float xx = x * x, yy = y * y, zz = z * z;
            const float xy = x * y, xz = x * z, yz = y * z;
            // Degree 2
            color = color + SH_C2a * xy * coefficients[4]
                          - SH_C2a * yz * coefficients[5]
                          + (SH_C2b * zz - SH_C2c) * coefficients[6]
                          - SH_C2a * xz * coefficients[7]
                          + SH_C2d * (xx - yy) * coefficients[8];

            if(total_sh_bases > 9u)
            {
                // Degree 3
                color = color + y * (SH_C3a * yy - SH_C3b * xx) * coefficients[9]
                              + SH_C3c * xy * z * coefficients[10]
                              + y * (SH_C3d - SH_C3e * zz) * coefficients[11]
                              + z * (SH_C3f * zz - SH_C3g) * coefficients[12]
                              + x * (SH_C3d - SH_C3e * zz) * coefficients[13]
                              + SH_C3h * z * (xx - yy) * coefficients[14]
                              + x * (SH_C3b * yy - SH_C3a * xx) * coefficients[15];
            }
        }
    }
    return max(color, 0.0f);
}

#endif  // STOCHASTICGS_H_SLANG