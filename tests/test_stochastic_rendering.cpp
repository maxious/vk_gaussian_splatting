/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Unit tests for stochastic Gaussian Splatting rendering helpers.
 * These helper functions are ported from shaders/stochasticgs.h.slang
 * for testability without requiring GPU hardware.
 */

#include "doctest.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ios>
#include <vector>

/*-------------------------------------------------------------------------------------------------
 * Helper functions ported from shaders/stochasticgs.h.slang
 *-----------------------------------------------------------------------------------------------*/

// 64-bit fragment packing / unpacking
inline uint64_t pack_fragment(uint32_t depth_bits, uint32_t gaussian_idx)
{
    return (static_cast<uint64_t>(depth_bits) << 32) | static_cast<uint64_t>(gaussian_idx);
}

inline void unpack_fragment(uint64_t frag, uint32_t& depth_bits, uint32_t& gaussian_idx)
{
    depth_bits = static_cast<uint32_t>(frag >> 32);
    gaussian_idx = static_cast<uint32_t>(frag & 0xFFFFFFFFu);
}

// Integer hash (stochastic_hash32 from math_utils.h)
inline uint32_t stochastic_hash32(uint32_t x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// make_seed - generate a 2D PRNG seed from Gaussian index and frame number
inline void make_seed(uint32_t idx, uint32_t frame, uint32_t& seed_x, uint32_t& seed_y)
{
    uint32_t h1 = stochastic_hash32(idx ^ (frame * 0x9E3779B9u));
    uint32_t h2 = stochastic_hash32((idx * 0xBB67AE85u) ^ frame);
    h1 ^= (h2 * 0x3C6EF372u);
    h2 ^= (h1 * 0xA54FF53Au);
    seed_x = h1;
    seed_y = h2;
}

// pcg2d_uint - 2D PCG hash (BIT-EXACT port from CUDA math_utils.h)
inline void pcg2d_uint(uint32_t& state_x, uint32_t& state_y)
{
    state_x = state_x * 1664525u + 1013904223u;
    state_y = state_y * 1664525u + 1013904223u;
    state_x += state_y * 1664525u;
    state_y += state_x * 1664525u;
    state_x ^= state_x >> 16u;
    state_y ^= state_y >> 16u;
    state_x += state_y * 1664525u;
    state_y += state_x * 1664525u;
    state_x ^= state_x >> 16u;
    state_y ^= state_y >> 16u;
}

// dilog - polynomial approximation of the dilogarithm Li_2(x) (degree-9 in Horner form)
inline float dilog(float x)
{
    float y = -0.952623943f;
    y = std::fma(y, x, 2.995873777f);
    y = std::fma(y, x, -3.762115247f);
    y = std::fma(y, x, 2.334605281f);
    y = std::fma(y, x, -0.773941317f);
    y = std::fma(y, x, 5.97506396e-02f);
    y = std::fma(y, x, -0.256617940f);
    y = std::fma(y, x, 2.000061866f);
    y *= x;
    const float s = 1.0f - x;
    if(s > 0.0f)
        y += s * std::log(std::max(s, 1.0e-37f));
    return y;
}

// inv_dilog - inverse polynomial approximation of the dilogarithm (degree-10 in Horner form)
inline float inv_dilog(float x)
{
    const float t = std::min(x / 1.6449340668482264f, 1.0f);
    float y = -1.27463503e+01f;
    y = std::fma(y, t, 5.88993459e+01f);
    y = std::fma(y, t, -1.16025780e+02f);
    y = std::fma(y, t, 1.26945827e+02f);
    y = std::fma(y, t, -8.43108826e+01f);
    y = std::fma(y, t, 3.48799862e+01f);
    y = std::fma(y, t, -8.89606235e+00f);
    y = std::fma(y, t, 1.38640936e+00f);
    y = std::fma(y, t, -7.80640876e-01f);
    y = std::fma(y, t, 1.64841888e+00f);
    y = std::fma(y, t, -2.82836687e-05f);
    return y;
}

// SH constant (from stochasticgs.h.slang)
static const float SH_C0 = 0.28209479177387814f;

// Convert float depth to its raw uint32 bit representation
// For positive floats, this mapping is monotonic (IEEE 754),
// enabling atomicMin-based depth sorting in the framebuffer.
inline uint32_t float_depth_to_bits(float depth)
{
    return std::bit_cast<uint32_t>(depth);
}

// eval_sh_at degree 0 - DC component only
// The 3DGS representation stores colors as SH coefficients with a 0.5 offset.
// Degree 0 evaluation: color = 0.5 + SH_C0 * f_dc
// where f_dc is (R, G, B) stored in coefficients[0].
inline void eval_sh_at_deg0(float dc_r, float dc_g, float dc_b,
                            float& out_r, float& out_g, float& out_b)
{
    out_r = std::max(0.5f + SH_C0 * dc_r, 0.0f);
    out_g = std::max(0.5f + SH_C0 * dc_g, 0.0f);
    out_b = std::max(0.5f + SH_C0 * dc_b, 0.0f);
}

// Clear sentinel for the framebuffer: all bits set to 1
// unpack_fragment yields depth=0xFFFFFFFF, idx=0xFFFFFFFF ("no fragment")
static constexpr uint64_t FRAMEBUFFER_CLEAR_VALUE = 0xFFFFFFFFFFFFFFFFull;

/*-------------------------------------------------------------------------------------------------
 * Test cases
 *-----------------------------------------------------------------------------------------------*/

TEST_SUITE("Stochastic Rendering Helpers")
{
    TEST_CASE("PRNG pcg2d_uint determinism")
    {
        // Call pcg2d_uint with known input state and assert output matches hardcoded values
        uint32_t s1_x = 0u, s1_y = 0u;
        pcg2d_uint(s1_x, s1_y);
        CHECK(s1_x == 417608103u);
        CHECK(s1_y == 90043601u);

        // Verify same input → same output (deterministic)
        uint32_t s2_x = 0u, s2_y = 0u;
        pcg2d_uint(s2_x, s2_y);
        CHECK(s2_x == 417608103u);
        CHECK(s2_y == 90043601u);

        // Verify different input → different output
        uint32_t s3_x = 12345u, s3_y = 67890u;
        pcg2d_uint(s3_x, s3_y);
        CHECK(s3_x == 3432654312u);
        CHECK(s3_y == 2676821219u);

        // Verify next iteration differs from first
        uint32_t s4_x = 0u, s4_y = 0u;
        pcg2d_uint(s4_x, s4_y); // first call
        pcg2d_uint(s4_x, s4_y); // second call (state has advanced)
        // After two iterations from (0,0), outputs should differ from the first call
        CHECK(s4_x != 417608103u);
        CHECK(s4_y != 90043601u);
    }

    TEST_CASE("64-bit packing/unpacking roundtrip")
    {
        // Test various (depth_bits, gaussian_idx) pairs roundtrip correctly
        struct TestPair { uint32_t depth; uint32_t idx; };
        TestPair pairs[] = {
            {0x00000000u, 0x00000000u},
            {0x12345678u, 0x9ABCDEF0u},
            {0xFFFFFFFFu, 0xFFFFFFFFu},
            {0x3DCCCCCDu, 0u},         // depth 0.1f, gaussian 0
            {0x3F000000u, 1u},         // depth 0.5f, gaussian 1
            {0x3F800000u, 42u},        // depth 1.0f, gaussian 42
        };

        for(const auto& p : pairs)
        {
            uint64_t packed = pack_fragment(p.depth, p.idx);
            uint32_t out_depth, out_idx;
            unpack_fragment(packed, out_depth, out_idx);
            INFO("depth=", std::hex, p.depth, " idx=", p.idx);
            CHECK(out_depth == p.depth);
            CHECK(out_idx == p.idx);
        }
    }

    TEST_CASE("dilog roundtrip")
    {
        // For x in [0.001, 0.999], inv_dilog(dilog(x)) ≈ x
        // The polynomial approximation has ~4e-5 worst-case error near endpoints.
        float test_values[] = {0.001f, 0.01f, 0.1f, 0.3f, 0.5f, 0.7f, 0.9f, 0.99f, 0.999f};
        for(float x : test_values)
        {
            float d = dilog(x);
            float result = inv_dilog(d);
            float diff = std::abs(result - x);
            INFO("x=", x, " dilog(x)=", d, " roundtrip=", result, " diff=", diff);
            CHECK(diff < 5e-4f);
            // For central values, the approximation is tighter
            if(x >= 0.1f && x <= 0.9f)
                CHECK(diff < 1e-4f);
        }
    }

    TEST_CASE("make_seed determinism")
    {
        // Same (idx, frame) → same seed
        uint32_t s1_x = 0, s1_y = 0;
        make_seed(42u, 7u, s1_x, s1_y);
        CHECK(s1_x == 1846015972u);
        CHECK(s1_y == 2498442166u);

        uint32_t s2_x = 0, s2_y = 0;
        make_seed(42u, 7u, s2_x, s2_y);
        CHECK(s2_x == 1846015972u);
        CHECK(s2_y == 2498442166u);

        // Different (idx, frame) → different seed
        uint32_t s3_x = 0, s3_y = 0;
        make_seed(43u, 7u, s3_x, s3_y);
        CHECK(s3_x == 1979888410u);
        CHECK(s3_y == 4029602536u);

        // Different frame also gives different seed
        uint32_t s4_x = 0, s4_y = 0;
        make_seed(42u, 8u, s4_x, s4_y);
        bool differs = (s4_x != s1_x || s4_y != s1_y);
        CHECK(differs);
    }

    TEST_CASE("Framebuffer clear value")
    {
        // 0xFFFFFFFFFFFFFFFF is the correct sentinel for "no fragment"
        uint64_t clear = FRAMEBUFFER_CLEAR_VALUE;
        CHECK(clear == 0xFFFFFFFFFFFFFFFFull);

        // Verify unpack yields sentinel values
        uint32_t depth, idx;
        unpack_fragment(clear, depth, idx);
        CHECK(depth == 0xFFFFFFFFu);
        CHECK(idx == 0xFFFFFFFFu);

        // Ensure the clear value is distinct from any valid packed fragment
        uint64_t valid = pack_fragment(float_depth_to_bits(0.1f), 0u);
        CHECK(valid != clear);
        CHECK(valid < clear); // any positive depth is < 0xFFFFFFFF in uint
    }

    TEST_CASE("Empty scene")
    {
        // 0 Gaussians → all framebuffer entries remain cleared
        constexpr size_t W = 4, H = 4;
        std::vector<uint64_t> framebuffer(W * H, FRAMEBUFFER_CLEAR_VALUE);

        // No Gaussians processed → all pixels stay at clear value
        for(size_t i = 0; i < framebuffer.size(); ++i)
        {
            CHECK(framebuffer[i] == FRAMEBUFFER_CLEAR_VALUE);
        }

        // Each pixel's unpacked values should be sentinel
        for(size_t i = 0; i < framebuffer.size(); ++i)
        {
            uint32_t depth, idx;
            unpack_fragment(framebuffer[i], depth, idx);
            CHECK(depth == 0xFFFFFFFFu);
            CHECK(idx == 0xFFFFFFFFu);
        }
    }

    TEST_CASE("Single Gaussian at center")
    {
        // Simulate rendering 1 Gaussian at depth 0.3 into center pixel
        constexpr size_t W = 4, H = 4;
        std::vector<uint64_t> framebuffer(W * H, FRAMEBUFFER_CLEAR_VALUE);

        // Pack a fragment for Gaussian 0 at depth 0.3 into pixel (2, 2)
        uint32_t depth_bits = float_depth_to_bits(0.3f);
        uint32_t gaussian_idx = 0u;
        uint64_t frag = pack_fragment(depth_bits, gaussian_idx);

        size_t center_pixel = 2 * W + 2; // row 2, col 2
        framebuffer[center_pixel] = frag;

        // Center pixel should have a non-clear value
        CHECK(framebuffer[center_pixel] != FRAMEBUFFER_CLEAR_VALUE);

        // Verify unpacked values
        uint32_t out_depth, out_idx;
        unpack_fragment(framebuffer[center_pixel], out_depth, out_idx);
        CHECK(out_depth == depth_bits);
        CHECK(out_idx == gaussian_idx);

        // Other pixels should remain cleared
        for(size_t i = 0; i < framebuffer.size(); ++i)
        {
            if(i == center_pixel) continue;
            CHECK(framebuffer[i] == FRAMEBUFFER_CLEAR_VALUE);
        }
    }

    TEST_CASE("Two Gaussians at different depths")
    {
        // 2 Gaussians at depths 0.1 and 0.5 → atomicMin should pick depth 0.1 (closer)
        // The upper 32 bits of the packed uint64 encode the float depth's raw bit pattern.
        // For positive IEEE 754 floats, smaller float → smaller uint32 → smaller uint64.

        uint32_t depth_01 = float_depth_to_bits(0.1f); // 0x3DCCCCCD
        uint32_t depth_05 = float_depth_to_bits(0.5f); // 0x3F000000

        // Verify that the closer depth has a smaller uint representation
        CHECK(depth_01 < depth_05);

        uint64_t frag_near  = pack_fragment(depth_01, 0u);
        uint64_t frag_far   = pack_fragment(depth_05, 1u);

        // atomicMin semantics: min(frag_near, frag_far) should pick frag_near
        uint64_t min_frag = std::min(frag_near, frag_far);
        CHECK(min_frag == frag_near);

        // Verify the correct Gaussian index is stored
        uint32_t out_depth, out_idx;
        unpack_fragment(min_frag, out_depth, out_idx);
        CHECK(out_depth == depth_01);
        CHECK(out_idx == 0u); // Gaussian 0 is the closer one

        // Simulate a full framebuffer scenario: both fragments compete for same pixel
        constexpr size_t W = 2, H = 2;
        std::vector<uint64_t> framebuffer(W * H, FRAMEBUFFER_CLEAR_VALUE);

        // Pixel (0,0): first the far fragment arrives, then the near one
        size_t pixel = 0;
        framebuffer[pixel] = frag_far;
        // atomicMin would replace frag_far with frag_near
        framebuffer[pixel] = std::min(framebuffer[pixel], frag_near);
        CHECK(framebuffer[pixel] == frag_near);

        unpack_fragment(framebuffer[pixel], out_depth, out_idx);
        CHECK(out_depth == depth_01);
        CHECK(out_idx == 0u);

        // Reverse order: near first, then far → near should remain
        framebuffer[pixel] = frag_near;
        framebuffer[pixel] = std::min(framebuffer[pixel], frag_far);
        CHECK(framebuffer[pixel] == frag_near);
    }

    TEST_CASE("SH evaluation at degree 0")
    {
        // SH degree 0 = DC component only
        // Formula: color = 0.5 + SH_C0 * f_dc
        // where f_dc is the (R, G, B) DC coefficient

        // Test with known DC values
        float r, g, b;
        eval_sh_at_deg0(0.2f, 0.5f, 0.8f, r, g, b);

        // 0.5 + 0.28209479177... * 0.2 = 0.5 + 0.0564189584 = 0.5564189584
        // 0.5 + 0.28209479177... * 0.5 = 0.5 + 0.1410473959 = 0.6410473959
        // 0.5 + 0.28209479177... * 0.8 = 0.5 + 0.2256758334 = 0.7256758334
        CHECK(std::abs(r - 0.5564189584f) < 1e-6f);
        CHECK(std::abs(g - 0.6410473959f) < 1e-6f);
        CHECK(std::abs(b - 0.7256758334f) < 1e-6f);

        // Test DC=0: color should be exactly 0.5 per channel
        eval_sh_at_deg0(0.0f, 0.0f, 0.0f, r, g, b);
        CHECK(std::abs(r - 0.5f) < 1e-6f);
        CHECK(std::abs(g - 0.5f) < 1e-6f);
        CHECK(std::abs(b - 0.5f) < 1e-6f);

        // Verify colors are clamped to >= 0
        // SH_C0 * (-2.0) < -0.5, so color would be negative without clamping
        eval_sh_at_deg0(-2.0f, -2.0f, -2.0f, r, g, b);
        CHECK(r >= 0.0f);
        CHECK(g >= 0.0f);
        CHECK(b >= 0.0f);
    }
}
