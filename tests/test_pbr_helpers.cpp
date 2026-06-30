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
 * Unit tests for PBR math helpers ported from nvpro_core2 Slang reference.
 * These functions are CPU ports of the Slang shaders used in the TRON PBR
 * pipeline: GGX BRDF microfacet math, MIS weighting, hemisphere sampling,
 * and AgX tone mapping.
 *
 * Reference files:
 *   nvpro_core2/nvshaders/pbr_ggx_microfacet.h.slang  — D_GGX, smith_shadow_or_mask, schlickFresnel
 *   nvpro_core2/nvshaders/bsdf_functions.h.slang       — bsdfEvaluateSimple (Cook-Torrance)
 *   nvpro_core2/nvshaders/functions.h.slang            — cosineSampleHemisphere, powerHeuristic
 *   nvpro_core2/nvshaders/tonemap_functions.h.slang    — tonemapAgX
 */

#include "doctest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

/*-------------------------------------------------------------------------------------------------
 * 3-vector helper (float3 equivalent)
 *-----------------------------------------------------------------------------------------------*/
struct Vec3
{
  float x, y, z;

  Vec3() : x(0.f), y(0.f), z(0.f) {}
  Vec3(float s) : x(s), y(s), z(s) {}
  Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

  Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
  Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
  Vec3 operator*(const Vec3& o) const { return {x * o.x, y * o.y, z * o.z}; }
  Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
  Vec3 operator/(const Vec3& o) const { return {x / o.x, y / o.y, z / o.z}; }
  Vec3 operator/(float s) const { return {x / s, y / s, z / s}; }
  Vec3 operator-() const { return {-x, -y, -z}; }

  float length() const { return std::sqrt(x * x + y * y + z * z); }
  float dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
  Vec3 cross(const Vec3& o) const
  {
    return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
  }
};

inline Vec3 normalize(const Vec3& v)
{
  float len = v.length();
  return (len > 0.f) ? v / len : Vec3(0.f, 0.f, 0.f);
}

inline Vec3 max3(const Vec3& a, float s) { return {std::max(a.x, s), std::max(a.y, s), std::max(a.z, s)}; }

inline Vec3 min3(const Vec3& a, float s) { return {std::min(a.x, s), std::min(a.y, s), std::min(a.z, s)}; }

inline Vec3 clamp3(const Vec3& v, float lo, float hi)
{
  return {std::clamp(v.x, lo, hi), std::clamp(v.y, lo, hi), std::clamp(v.z, lo, hi)};
}

inline Vec3 log2Vec(const Vec3& v) { return {std::log2(v.x), std::log2(v.y), std::log2(v.z)}; }

inline Vec3 absVec(const Vec3& v) { return {std::fabs(v.x), std::fabs(v.y), std::fabs(v.z)}; }

// Column-major 3x3 matrix (matching Slang float3x3 convention)
// Slang float3x3(a,b,c, d,e,f, g,h,i) → col0=(a,b,c), col1=(d,e,f), col2=(g,h,i)
struct Mat3
{
  // Stored column-major: m[col][row]
  float m[9];

  Mat3(float m00, float m01, float m02,  // row 0
       float m10, float m11, float m12,  // row 1
       float m20, float m21, float m22)  // row 2
  {
    // Column 0
    m[0] = m00;
    m[1] = m01;
    m[2] = m02;
    // Column 1
    m[3] = m10;
    m[4] = m11;
    m[5] = m12;
    // Column 2
    m[6] = m20;
    m[7] = m21;
    m[8] = m22;
  }
};

// mul(v_row, M_column_major) — replicate Slang's mul(float3, float3x3)
inline Vec3 mulRowVec(const Vec3& v, const Mat3& M)
{
  return {
      v.x * M.m[0] + v.y * M.m[3] + v.z * M.m[6],  // row 0 of matrix
      v.x * M.m[1] + v.y * M.m[4] + v.z * M.m[7],  // row 1 of matrix
      v.x * M.m[2] + v.y * M.m[5] + v.z * M.m[8],  // row 2 of matrix
  };
}

/*-------------------------------------------------------------------------------------------------
 * Constants
 *-----------------------------------------------------------------------------------------------*/
static constexpr float M_PI_F  = 3.14159265358979323846f;
static constexpr float M_1_PI_F = 1.0f / M_PI_F;

/*-------------------------------------------------------------------------------------------------
 * Test 1: GGX Distribution (D)
 * Ported from nvpro_core2/nvshaders/pbr_ggx_microfacet.h.slang: D_GGX() (lines 484-489)
 *
 *   alphaRoughnessSq = alpha * alpha
 *   f = (NdotH * NdotH) * (alphaRoughnessSq - 1.0) + 1.0
 *   return alphaRoughnessSq / (M_PI * f * f)
 *-----------------------------------------------------------------------------------------------*/
static float D_GGX(float NdotH, float alphaRoughness)
{
  float a2 = alphaRoughness * alphaRoughness;
  float f  = (NdotH * NdotH) * (a2 - 1.0f) + 1.0f;
  return a2 / (M_PI_F * f * f);
}

/*-------------------------------------------------------------------------------------------------
 * Test 2: Smith Geometry (G)
 * Ported from nvpro_core2/nvshaders/pbr_ggx_microfacet.h.slang: smith_shadow_or_mask() (lines 132-144)
 *
 * For a local-space direction k = (T·dir, B·dir, N·dir) and isotropic roughness alpha:
 *   G1 = 2.0 / (1.0 + sqrt(1.0 + alpha^2 * (T^2 + B^2) / N^2))
 * When direction has no tangent component (T=B=0, N=NdotV), this reduces to G1=1 (no shadowing).
 * For a general direction with T = sqrt(1 - NdotV^2), the formula simplifies to:
 *   G1 = 2 * NdotV / (NdotV + sqrt(alpha^2 + (1 - alpha^2) * NdotV^2))
 * G  = G1(NdotV) * G1(NdotL)
 *-----------------------------------------------------------------------------------------------*/
static float G1_Smith(float NdotX, float alpha)
{
  // Equivalent to smith_shadow_or_mask for isotropic roughness
  // when the direction has no bitangent component (k.y = 0)
  float a2    = alpha * alpha;
  float denom = NdotX + std::sqrt(a2 + (1.0f - a2) * NdotX * NdotX);
  return 2.0f * NdotX / denom;
}

static float G_Smith(float NdotV, float NdotL, float alpha)
{
  return G1_Smith(NdotV, alpha) * G1_Smith(NdotL, alpha);
}

/*-------------------------------------------------------------------------------------------------
 * Test 3: Schlick Fresnel (F)
 * Ported from nvpro_core2/nvshaders/pbr_ggx_microfacet.h.slang: schlickFresnel() (lines 36-39)
 *
 *   F = F0 + (F90 - F0) * pow(1 - VdotH, 5.0)
 *-----------------------------------------------------------------------------------------------*/
static float schlickFresnel(float F0, float F90, float VdotH)
{
  return F0 + (F90 - F0) * std::pow(1.0f - VdotH, 5.0f);
}

/*-------------------------------------------------------------------------------------------------
 * Test 4: Cook-Torrance specular BRDF
 * Ported from nvpro_core2/nvshaders/bsdf_functions.h.slang: bsdfEvaluateSimple() (lines 1269-1310)
 *
 * The standard Cook-Torrance form (Eq. 20 in Walter et al. 2007):
 *   spec = D * F * G / (4 * NdotV * NdotL)
 *
 * Note: The nvpro_core2 implementation uses 4*NdotV*NdotH in the denominator
 * instead of 4*NdotV*NdotL because the PDF normalization works through the
 * half-vector Jacobian. Both formulations are numerically equivalent when
 * VdotH == LdotH (which holds for reflection). We test with the textbook form
 * for clarity and note the equivalence.
 *-----------------------------------------------------------------------------------------------*/
static float cookTorranceSpecular(float NdotV,
                                  float NdotL,
                                  float NdotH,
                                  float VdotH,
                                  float alpha,
                                  float F0)
{
  float D = D_GGX(NdotH, alpha);
  float G = G_Smith(NdotV, NdotL, alpha);
  float F = schlickFresnel(F0, 1.0f, VdotH);
  return D * F * G / (4.0f * NdotV * NdotL);
}

/*-------------------------------------------------------------------------------------------------
 * Test 5: MIS Power Heuristic (Balance Heuristic, beta=2)
 * Ported from nvpro_core2/nvshaders/functions.h.slang: powerHeuristic() (lines 202-206)
 *
 *   w_A = pdfA^2 / (pdfA^2 + pdfB^2)
 *   w_B = pdfB^2 / (pdfA^2 + pdfB^2)  →  w_A + w_B = 1
 *-----------------------------------------------------------------------------------------------*/
static float misPowerHeuristic(float pdfA, float pdfB)
{
  float t = pdfA * pdfA;
  return t / (pdfB * pdfB + t);
}

/*-------------------------------------------------------------------------------------------------
 * Test 6: Cosine-Weighted Hemisphere Sampling
 * Ported from nvpro_core2/nvshaders/functions.h.slang: cosineSampleHemisphere() (lines 151-160)
 *
 *   r   = sqrt(r1)
 *   phi = 2 * pi * r2
 *   dir = (r * cos(phi), r * sin(phi), sqrt(1 - r1))
 *
 * PDF = cos(theta) / pi = dir.z / pi
 *-----------------------------------------------------------------------------------------------*/
static Vec3 cosineSampleHemisphere(float r1, float r2)
{
  float r   = std::sqrt(r1);
  float phi = 2.0f * M_PI_F * r2;
  return {r * std::cos(phi), r * std::sin(phi), std::sqrt(std::max(0.0f, 1.0f - r1))};
}

/*-------------------------------------------------------------------------------------------------
 * Test 7: Environment Map Direction Validation
 *
 * Verifies that sampled directions are unit-length and within the upper hemisphere.
 * Uses both cosine-weighted and uniform hemisphere samples.
 *-----------------------------------------------------------------------------------------------*/
static Vec3 uniformSampleHemisphere(float r1, float r2)
{
  float cosTheta = r1;
  float sinTheta = std::sqrt(std::max(0.0f, 1.0f - cosTheta * cosTheta));
  float phi      = 2.0f * M_PI_F * r2;
  return {sinTheta * std::cos(phi), sinTheta * std::sin(phi), cosTheta};
}

/*-------------------------------------------------------------------------------------------------
 * Test 8: AgX Tone Mapping
 * Ported from nvpro_core2/nvshaders/tonemap_functions.h.slang: tonemapAgX() (lines 149-179)
 *
 * Note: The function is named tonemapAgX, not agxToneMap.
 * Signature: float3 tonemapAgX(float3 color) — no exposure parameter.
 *-----------------------------------------------------------------------------------------------*/
static Vec3 tonemapAgX(Vec3 color)
{
  // 1. Input transform (column-major Slang float3x3)
  const Mat3 agx_mat(
      0.842479062253094f, 0.0423282422610123f, 0.0423756549057051f,    // row 0
      0.0784335999999992f, 0.878468636469772f, 0.0784336f,              // row 1
      0.0792237451477643f, 0.0791661274605434f, 0.879142973793104f      // row 2
  );
  color = mulRowVec(color, agx_mat);

  // 2. Log2 space encoding
  const float min_ev = -12.47393f;
  const float max_ev = 4.026069f;
  color              = clamp3(log2Vec(color), min_ev, max_ev);
  color              = (color - Vec3(min_ev)) / (max_ev - min_ev);

  // 3. 6th-order sigmoid approximation via Horner's method
  // Polynomial: 15.5*x^6 - 40.14*x^5 + 31.96*x^4 - 6.868*x^3 + 0.4298*x^2 + 0.1191*x - 0.0023
  Vec3 v = color * 15.5f - Vec3(40.14f);
  v      = color * v + Vec3(31.96f);
  v      = color * v - Vec3(6.868f);
  v      = color * v + Vec3(0.4298f);
  v      = color * v + Vec3(0.1191f);
  v      = color * v - Vec3(0.0023f);

  // 4. Output transform
  const Mat3 agx_mat_inv(
      1.19687900512017f, -0.0528968517574562f, -0.0529716355144438f,    // row 0
      -0.0980208811401368f, 1.15190312990417f, -0.0980434501171241f,    // row 1
      -0.0990297440797205f, -0.0989611768448433f, 1.15107367264116f     // row 2
  );
  return mulRowVec(v, agx_mat_inv);
}

/*-------------------------------------------------------------------------------------------------
 * Test cases
 *-----------------------------------------------------------------------------------------------*/

TEST_SUITE("PBR Helpers")
{
  TEST_CASE("GGX Distribution (D)")
  {
    // D_GGX(NdotH, alpha) from pbr_ggx_microfacet.h.slang (Khronos NDF)
    //
    // Test value table:
    //   NdotH  | alpha | expected D (hand-computed)
    //   -------|-------|----------------------------
    //   0.9    | 0.1   | 0.08111147
    //   0.5    | 0.5   | 0.120539
    //   0.1    | 0.9   | 0.258811
    //
    // Verify: D >= 0, D peaks near NdotH=1 for smooth surfaces (low alpha)
    // and spreads out for rough surfaces (high alpha).

    // Smooth surface: narrow peak
    float d_smooth = D_GGX(0.9f, 0.1f);
    CHECK(d_smooth == doctest::Approx(0.08111147f).epsilon(1e-5f));

    // Medium roughness
    float d_medium = D_GGX(0.5f, 0.5f);
    CHECK(d_medium == doctest::Approx(0.120539f).epsilon(1e-5f));

    // Rough surface: broad distribution
    float d_rough = D_GGX(0.1f, 0.9f);
    CHECK(d_rough == doctest::Approx(0.258811f).epsilon(1e-5f));

    // Edge cases
    // NdotH=1.0 (mirror): D = a^2 / (pi * a^4) = 1/(pi * a^2)
    float d_mirror = D_GGX(1.0f, 0.2f);
    float expected = 1.0f / (M_PI_F * 0.2f * 0.2f);  // = 1/(pi*0.04) ≈ 7.9577
    CHECK(d_mirror == doctest::Approx(expected).epsilon(1e-5f));

    // NdotH → 0 (grazing): D → a^2 / pi
    float d_grazing = D_GGX(1e-6f, 0.5f);
    float a2         = 0.25f;
    expected         = a2 / M_PI_F;  // ≈ 0.07958
    CHECK(d_grazing == doctest::Approx(expected).epsilon(1e-2f));  // looser for near-zero

    // NdotH=0: exactly a^2 / pi
    float d_zero = D_GGX(0.0f, 0.3f);
    expected     = 0.09f / M_PI_F;
    CHECK(d_zero == doctest::Approx(expected).epsilon(1e-5f));

    // D should be non-negative for all valid inputs
    CHECK(D_GGX(0.5f, 0.05f) >= 0.0f);
    CHECK(D_GGX(0.999f, 0.999f) >= 0.0f);
  }

  TEST_CASE("Smith Geometry (G)")
  {
    // G1_Smith(NdotX, alpha) based on smith_shadow_or_mask() from pbr_ggx_microfacet.h.slang
    //
    // For isotropic roughness with direction k=(T,0,N) in local space,
    // the formula reduces to the standard GGX Smith G1:
    //   G1 = 2 * NdotX / (NdotX + sqrt(alpha^2 + (1-alpha^2) * NdotX^2))
    //
    // G_Smith = G1(NdotV) * G1(NdotL)

    // --- G1 tests ---
    // NdotV=1.0: G1 should be 1.0 for any alpha (looking straight at surface)
    CHECK(G1_Smith(1.0f, 0.1f) == doctest::Approx(1.0f).epsilon(1e-6f));
    CHECK(G1_Smith(1.0f, 0.5f) == doctest::Approx(1.0f).epsilon(1e-6f));
    CHECK(G1_Smith(1.0f, 0.9f) == doctest::Approx(1.0f).epsilon(1e-6f));

    // NdotV=0.8, alpha=0.1: G1 ≈ 0.9986
    float g1_a = G1_Smith(0.8f, 0.1f);
    CHECK(g1_a == doctest::Approx(0.998596f).epsilon(1e-4f));

    // NdotV=0.3, alpha=0.5: G1 ≈ 0.6949
    float g1_b = G1_Smith(0.3f, 0.5f);
    CHECK(g1_b == doctest::Approx(0.69487f).epsilon(1e-4f));

    // NdotV=0.2, alpha=0.9: G1 ≈ 0.3622 (rough surface, grazing angle)
    float g1_c = G1_Smith(0.2f, 0.9f);
    CHECK(g1_c == doctest::Approx(0.36225f).epsilon(1e-4f));

    // NdotV → 0+: G1 should approach 0 (totally shadowed at grazing)
    float g1_grazing = G1_Smith(1e-6f, 0.5f);
    CHECK(g1_grazing < 0.01f);  // near zero at extreme grazing

    // --- Combined G tests ---
    // G = G1(0.8) * G1(0.7), alpha=0.1
    float g_combined = G_Smith(0.8f, 0.7f, 0.1f);
    // G1(0.8, 0.1)=0.998596, G1(0.7, 0.1)=0.99741, product=0.99590
    CHECK(g_combined == doctest::Approx(0.99590f).epsilon(1e-4f));

    // G = G1(0.3) * G1(0.2), alpha=0.5
    float g_combined2 = G_Smith(0.3f, 0.2f, 0.5f);
    // G1(0.3, 0.5)=0.69487, G1(0.2, 0.5)=0.54858, product=0.38119
    CHECK(g_combined2 == doctest::Approx(0.38119f).epsilon(1e-4f));

    // G must be in [0, 1]
    CHECK(g_combined >= 0.0f);
    CHECK(g_combined <= 1.0f);
    CHECK(g_combined2 >= 0.0f);
    CHECK(g_combined2 <= 1.0f);
  }

  TEST_CASE("Schlick Fresnel (F)")
  {
    // schlickFresnel(F0, F90, VdotH) from pbr_ggx_microfacet.h.slang
    //   F = F0 + (F90 - F0) * (1 - VdotH)^5

    // VdotH = 1.0 (head-on): F = F0
    CHECK(schlickFresnel(0.04f, 1.0f, 1.0f) == doctest::Approx(0.04f).epsilon(1e-6f));
    CHECK(schlickFresnel(0.9f, 1.0f, 1.0f) == doctest::Approx(0.9f).epsilon(1e-6f));

    // VdotH = 0.5 (intermediate angle): (1 - 0.5)^5 = 0.03125
    float f_mid = schlickFresnel(0.04f, 1.0f, 0.5f);
    // F = 0.04 + (1.0 - 0.04) * 0.03125 = 0.04 + 0.96 * 0.03125 = 0.04 + 0.03 = 0.07
    CHECK(f_mid == doctest::Approx(0.07f).epsilon(1e-6f));

    // Metallic F0=0.9, VdotH=0.5
    float f_metal = schlickFresnel(0.9f, 1.0f, 0.5f);
    // F = 0.9 + (1.0 - 0.9) * 0.03125 = 0.9 + 0.1 * 0.03125 = 0.9 + 0.003125 = 0.903125
    CHECK(f_metal == doctest::Approx(0.903125f).epsilon(1e-6f));

    // VdotH = 0.0 (grazing): F = F90 = 1.0
    CHECK(schlickFresnel(0.04f, 1.0f, 0.0f) == doctest::Approx(1.0f).epsilon(1e-6f));
    CHECK(schlickFresnel(0.5f, 1.0f, 0.0f) == doctest::Approx(1.0f).epsilon(1e-6f));

    // Monotonicity: as VdotH decreases (more grazing), F increases
    CHECK(schlickFresnel(0.04f, 1.0f, 0.6f) <= schlickFresnel(0.04f, 1.0f, 0.3f));
    CHECK(schlickFresnel(0.04f, 1.0f, 0.95f) <= schlickFresnel(0.04f, 1.0f, 0.5f));

    // F should be in [F0, F90] for VdotH ∈ [0, 1]
    float f_varied = schlickFresnel(0.04f, 1.0f, 0.25f);
    CHECK(f_varied >= 0.04f);
    CHECK(f_varied <= 1.0f);
  }

  TEST_CASE("Cook-Torrance specular BRDF")
  {
    // Full Cook-Torrance: spec = D * F * G / (4 * NdotV * NdotL)
    //
    // Test configuration: VL symmetric about the normal (mirror reflection)
    // V at 30° from normal, L at -30° from normal: H = N, NdotH = 1.0
    // N=(0,0,1), V=(sin30°,0,cos30°), L=(-sin30°,0,cos30°)
    float NdotV = std::cos(30.0f * M_PI_F / 180.0f);  // 0.8660254
    float NdotL = NdotV;                                // symmetric
    float NdotH = 1.0f;                                 // H = N (mirror reflect)
    float VdotH = NdotV;                                // V·H = V·N = cos30
    float alpha  = 0.2f;
    float F0     = 0.04f;

    float spec = cookTorranceSpecular(NdotV, NdotL, NdotH, VdotH, alpha, F0);

    // Verify spec is non-negative
    CHECK(spec >= 0.0f);

    // Compute expected components and verify they match:
    float D = D_GGX(NdotH, alpha);                           // NdotH=1 → D = 1/(pi * alpha^2)
    CHECK(D == doctest::Approx(1.0f / (M_PI_F * alpha * alpha)).epsilon(1e-5f));
    float G = G_Smith(NdotV, NdotL, alpha);
    float F = schlickFresnel(F0, 1.0f, VdotH);
    float expected = D * F * G / (4.0f * NdotV * NdotL);
    CHECK(spec == doctest::Approx(expected).epsilon(1e-5f));

    // Rough surface (alpha=0.9): should be lower specular at peak
    // For mirror reflection (H=N, NdotH=1), D = 1/(pi * alpha^2) decreases as alpha increases
    float spec_rough =
        cookTorranceSpecular(NdotV, NdotL, NdotH, VdotH, 0.9f, F0);
    CHECK(spec_rough < spec);

    // Smooth surface (alpha=0.01): sharp peak → much higher specular
    float spec_smooth =
        cookTorranceSpecular(NdotV, NdotL, NdotH, VdotH, 0.01f, F0);
    CHECK(spec_smooth > spec);
    CHECK(spec_smooth > spec_rough);

    // Intermediate roughness between smooth and rough
    float spec_medium =
        cookTorranceSpecular(NdotV, NdotL, NdotH, VdotH, 0.3f, F0);
    CHECK(spec_medium < spec_smooth);
    CHECK(spec_medium > spec_rough);

    // Metallic case (F0=0.9): higher reflectance than dielectric
    float spec_metal =
        cookTorranceSpecular(NdotV, NdotL, NdotH, VdotH, alpha, 0.9f);
    CHECK(spec_metal > spec);
  }

  TEST_CASE("MIS Weight (Balance Heuristic)")
  {
    // misPowerHeuristic(pdfA, pdfB) from functions.h.slang: powerHeuristic()
    //   w_A = pdfA^2 / (pdfA^2 + pdfB^2)
    //   w_B = pdfB^2 / (pdfA^2 + pdfB^2)

    // Equal PDFs: weight should be 0.5 each
    float w_eq = misPowerHeuristic(0.5f, 0.5f);
    CHECK(w_eq == doctest::Approx(0.5f).epsilon(1e-6f));

    // pdfA dominates (0.9 vs 0.1): w_A = 0.81/(0.01+0.81) = 0.81/0.82 = 0.9878
    float w_dom = misPowerHeuristic(0.9f, 0.1f);
    CHECK(w_dom == doctest::Approx(0.987805f).epsilon(1e-5f));

    // pdfB dominates: w_A = 0.01/(0.25+0.01) = 0.01/0.26 = 0.03846
    float w_sub = misPowerHeuristic(0.1f, 0.5f);
    CHECK(w_sub == doctest::Approx(0.0384615f).epsilon(1e-5f));

    // Verify weights of A and B sum to 1
    float wA = misPowerHeuristic(0.3f, 0.7f);
    float wB = misPowerHeuristic(0.7f, 0.3f);  // swapped args = B's perspective
    CHECK((wA + wB) == doctest::Approx(1.0f).epsilon(1e-6f));

    // More test pairs — weights must always sum to 1
    struct PDFPair
    {
      float a, b;
    };
    PDFPair pairs[] = {
        {0.1f, 0.9f},
        {0.25f, 0.75f},
        {0.333f, 0.667f},
        {0.6f, 0.4f},
        {0.99f, 0.01f},
    };
    for(auto p : pairs)
    {
      wA = misPowerHeuristic(p.a, p.b);
      wB = misPowerHeuristic(p.b, p.a);
      INFO("pdfA=", p.a, " pdfB=", p.b, " wA=", wA, " wB=", wB);
      CHECK((wA + wB) == doctest::Approx(1.0f).epsilon(1e-6f));
    }

    // Edge case: pdfA=0 gives weight 0
    float w_zero = misPowerHeuristic(0.0f, 0.5f);
    CHECK(w_zero == doctest::Approx(0.0f).epsilon(1e-6f));

    // pdfB=0 gives weight 1 (pdfA dominates exclusively)
    float w_one = misPowerHeuristic(0.5f, 0.0f);
    CHECK(w_one == doctest::Approx(1.0f).epsilon(1e-6f));

    // Both zero: 0/0 is NaN. Implementation returns 0/(0+0)=NaN.
    // The expectation depends on the implementation; nvpro_core2 doesn't guard against 0/0.
    // We skip this pathological case.
    // Instead, verify both close to zero gives ~0.5
    float w_tiny = misPowerHeuristic(1e-10f, 1e-10f);
    CHECK(w_tiny == doctest::Approx(0.5f).epsilon(1e-3f));
  }

  TEST_CASE("Cosine-weighted hemisphere sampling")
  {
    // cosineSampleHemisphere(r1, r2) from functions.h.slang
    // Samples direction in upper hemisphere (z >= 0) with cos(θ)/π PDF.

    // Test: r1 → [0,1) controls radius (and thus cosθ), r2 → [0,1) controls azimuth
    struct SamplePoint
    {
      float r1, r2;
    };

    SamplePoint samples[] = {
        {0.25f, 0.0f},   // phi=0: dir along +X in XY plane
        {0.5f, 0.25f},   // phi=π/2: dir along +Y
        {0.75f, 0.5f},   // phi=π: dir along -X
        {1.0f, 0.75f},   // phi=3π/2: dir along -Y, r=1, z=0
        {0.0f, 0.33f},   // r=0: dir along +Z only
    };

    for(auto sp : samples)
    {
      Vec3 dir = cosineSampleHemisphere(sp.r1, sp.r2);
      float len = dir.length();
      INFO("r1=", sp.r1, " r2=", sp.r2, " dir=(", dir.x, ",", dir.y, ",", dir.z, ") len=", len);
      // Direction must be unit length
      CHECK(len == doctest::Approx(1.0f).epsilon(1e-5f));
      // Direction must be in upper hemisphere
      CHECK(dir.z >= 0.0f);
    }

    // Verify r1 → 0 produces direction along +Z
    Vec3 dir_up = cosineSampleHemisphere(0.0f, 0.5f);
    CHECK(dir_up.z == doctest::Approx(1.0f).epsilon(1e-5f));
    CHECK(dir_up.x == doctest::Approx(0.0f).epsilon(1e-5f));
    CHECK(dir_up.y == doctest::Approx(0.0f).epsilon(1e-5f));

    // Verify r1 → 1 produces direction on XY plane (z ≈ 0)
    Vec3 dir_flat = cosineSampleHemisphere(1.0f, 0.0f);
    CHECK(dir_flat.z == doctest::Approx(0.0f).epsilon(1e-5f));
    CHECK(dir_flat.x == doctest::Approx(1.0f).epsilon(1e-5f));  // cos(φ)=cos(0)=1, sin(φ)=sin(0)=0

    // PDF = cos(θ)/π = dir.z / π
    // For r1=0.5: z = sqrt(1-0.5) = sqrt(0.5) = 0.7071, PDF = 0.7071/π = 0.22508
    Vec3 dir_half = cosineSampleHemisphere(0.5f, 0.5f);
    float pdf      = dir_half.z * M_1_PI_F;
    CHECK(pdf == doctest::Approx(0.225079f).epsilon(1e-4f));

    // Statistical check: average of many samples should have z > 0 (all in upper hemisphere)
    int count = 1000;
    float sum_z = 0.0f;
    for(int i = 0; i < count; ++i)
    {
      float r1 = static_cast<float>(i) / static_cast<float>(count);
      float r2 = (r1 * 0.618033988749895f);  // golden ratio for pseudo-random
      r2       = r2 - std::floor(r2);
      Vec3 d   = cosineSampleHemisphere(r1, r2);
      CHECK(d.z >= 0.0f);
      sum_z += d.z;
    }
    // Average z for cosine hemisphere should be ~2/3 (0.6667)
    float avg_z = sum_z / static_cast<float>(count);
    CHECK(avg_z > 0.6f);
    CHECK(avg_z < 0.75f);
  }

  TEST_CASE("Environment map direction validation")
  {
    // Verifies sampled directions are valid for environment map lookups.
    // Uses both cosine-weighted and uniform hemisphere sampling.

    // Uniform hemisphere: PDF = 1/(2π)
    {
      Vec3 d = uniformSampleHemisphere(0.5f, 0.25f);
      float len = d.length();
      CHECK(len == doctest::Approx(1.0f).epsilon(1e-5f));
      CHECK(d.z >= 0.0f);
    }

    // All uniform hemisphere samples should be unit vectors in upper hemisphere
    for(int i = 0; i < 100; ++i)
    {
      float r1 = static_cast<float>(i) / 100.0f;
      float r2 = r1 * 0.75487767f;
      r2       = r2 - std::floor(r2);
      Vec3 d   = uniformSampleHemisphere(r1, r2);
      float len = d.length();
      INFO("i=", i, " d=(", d.x, ",", d.y, ",", d.z, ") len=", len);
      CHECK(len == doctest::Approx(1.0f).epsilon(1e-4f));
      CHECK(d.z >= 0.0f);
    }

    // Cosine-weighted hemisphere samples are also valid directions
    for(int i = 0; i < 100; ++i)
    {
      float r1 = static_cast<float>(i) / 100.0f;
      float r2 = r1 * 0.75487767f;
      r2       = r2 - std::floor(r2);
      Vec3 d   = cosineSampleHemisphere(r1, r2);
      float len = d.length();
      INFO("i=", i, " r1=", r1, " d=(", d.x, ",", d.y, ",", d.z, ") len=", len);
      CHECK(len == doctest::Approx(1.0f).epsilon(1e-4f));
      CHECK(d.z >= 0.0f);
    }

    // Direction vectors can be converted to spherical coordinates for env map lookup
    // theta = acos(z), phi = atan2(y, x) — these should always be valid
    Vec3 test_dirs[] = {
        {0.0f, 0.0f, 1.0f},   // straight up (north pole)
        {1.0f, 0.0f, 0.0f},   // equator, +X
        {0.0f, 1.0f, 0.0f},   // equator, +Y
        {0.6f, 0.8f, 0.0f},   // equator, off-axis
        {0.3f, 0.4f, 0.5f},   // mid-latitude, unit after normalize
    };
    for(auto d : test_dirs)
    {
      d = normalize(d);  // ensure unit length
      float theta = std::acos(std::clamp(d.z, -1.0f, 1.0f));
      float phi   = std::atan2(d.y, d.x);
      // theta should be in [0, pi] for upper hemisphere: [0, pi/2]
      CHECK(theta >= 0.0f);
      CHECK(theta <= M_PI_F);
      // Spherical → Cartesian and back should recover the direction
      Vec3 recovered = {
          std::sin(theta) * std::cos(phi),
          std::sin(theta) * std::sin(phi),
          std::cos(theta),
      };
      CHECK(recovered.x == doctest::Approx(d.x).epsilon(1e-4f));
      CHECK(recovered.y == doctest::Approx(d.y).epsilon(1e-4f));
      CHECK(recovered.z == doctest::Approx(d.z).epsilon(1e-4f));
    }
  }

  TEST_CASE("AgX tone mapping")
  {
    // tonemapAgX(color) from tonemap_functions.h.slang (Benjamin Wrensch's AgX approximation)
    //
    // Steps:
    //   1. Color space transform via agx_mat
    //   2. log2 encoding + clamp to [min_ev, max_ev]
    //   3. Normalize to [0, 1]
    //   4. 6th-order sigmoid polynomial
    //   5. Output transform via agx_mat_inv
    // Output is in sRGB space (no additional gamma correction needed).

    // --- Pure black (0,0,0) ---
    // log2(0) → -inf, clamped to min_ev = -12.47393
    // After normalization: (-12.47393 + 12.47393) / 16.5 = 0
    // Sigmoid at 0: 15.5*0^6 + ... + 0.1191*0 - 0.0023 = -0.0023
    // Output transform of vec3(-0.0023) gives small negative values
    Vec3 black = tonemapAgX(Vec3(0.0f));
    // Near-black input should produce near-black output (may be slightly negative)
    CHECK(black.x < 0.01f);
    CHECK(black.y < 0.01f);
    CHECK(black.z < 0.01f);

    // --- 18% gray (0.18, 0.18, 0.18) ---
    // This is the reference mid-gray. AgX maps it to approximately 0.5 in sRGB.
    Vec3 mid_gray = tonemapAgX(Vec3(0.18f));
    CHECK(mid_gray.x == doctest::Approx(0.497f).epsilon(1e-2f));
    CHECK(mid_gray.y == doctest::Approx(0.497f).epsilon(1e-2f));
    CHECK(mid_gray.z == doctest::Approx(0.497f).epsilon(1e-2f));
    // Neutral input must produce neutral output (hue preserved)
    CHECK(mid_gray.x == doctest::Approx(mid_gray.y).epsilon(1e-3f));
    CHECK(mid_gray.y == doctest::Approx(mid_gray.z).epsilon(1e-3f));

    // --- Pure white (1.0, 1.0, 1.0) ---
    Vec3 white = tonemapAgX(Vec3(1.0f));
    // Should be bright but not clipped: expected ~0.79
    CHECK(white.x == doctest::Approx(0.787f).epsilon(2e-2f));
    CHECK(white.y == doctest::Approx(0.787f).epsilon(2e-2f));
    CHECK(white.z == doctest::Approx(0.787f).epsilon(2e-2f));
    // Neutral input → neutral output
    CHECK(white.x == doctest::Approx(white.y).epsilon(1e-2f));
    CHECK(white.y == doctest::Approx(white.z).epsilon(1e-2f));

    // --- HDR value (10.0, 10.0, 10.0) ---
    // Very bright HDR should compress, not clip to 1.0
    Vec3 hdr = tonemapAgX(Vec3(10.0f));
    CHECK(hdr.x > 0.9f);    // bright but not clipped
    CHECK(hdr.x < 1.5f);    // should be around 1.0-1.2
    CHECK(hdr.y > 0.9f);
    CHECK(hdr.y < 1.5f);
    CHECK(hdr.z > 0.9f);
    CHECK(hdr.z < 1.5f);
    // Neutral input → neutral output
    CHECK(hdr.x == doctest::Approx(hdr.y).epsilon(5e-2f));
    CHECK(hdr.y == doctest::Approx(hdr.z).epsilon(5e-2f));

    // --- Monotonicity ---
    // Brighter input must produce brighter or equal output
    Vec3 c1 = tonemapAgX(Vec3(0.1f));
    Vec3 c2 = tonemapAgX(Vec3(0.5f));
    Vec3 c3 = tonemapAgX(Vec3(2.0f));
    CHECK(c1.x <= c2.x);
    CHECK(c1.y <= c2.y);
    CHECK(c1.z <= c2.z);
    CHECK(c2.x <= c3.x);
    CHECK(c2.y <= c3.y);
    CHECK(c2.z <= c3.z);

    // --- Hue preservation for colored input ---
    // Same hue, different luminance → output should have similar chromaticity
    Vec3 red    = tonemapAgX(Vec3(1.0f, 0.1f, 0.1f));
    Vec3 bright_red = tonemapAgX(Vec3(5.0f, 0.5f, 0.5f));
    // Red channel should dominate
    CHECK(red.x > red.y);
    CHECK(red.x > red.z);
    CHECK(bright_red.x > bright_red.y);
    CHECK(bright_red.x > bright_red.z);
    // Brighter input → brighter output
    CHECK(bright_red.x > red.x);

    // --- Extreme value doesn't explode ---
    Vec3 extreme = tonemapAgX(Vec3(100.0f));
    CHECK(extreme.x < 2.0f);
    CHECK(extreme.y < 2.0f);
    CHECK(extreme.z < 2.0f);
    // Should still be neutral for neutral input
    CHECK(extreme.x == doctest::Approx(extreme.y).epsilon(1e-2f));
    CHECK(extreme.y == doctest::Approx(extreme.z).epsilon(1e-2f));

    // --- Per-channel consistency ---
    // Pure red, green, blue should map through correctly
    Vec3 red_only = tonemapAgX(Vec3(1.0f, 0.0f, 0.0f));
    Vec3 green_only = tonemapAgX(Vec3(0.0f, 1.0f, 0.0f));
    Vec3 blue_only  = tonemapAgX(Vec3(0.0f, 0.0f, 1.0f));
    CHECK(red_only.x > red_only.y);
    CHECK(red_only.x > red_only.z);
    CHECK(green_only.y > green_only.x);
    CHECK(green_only.y > green_only.z);
    CHECK(blue_only.z > blue_only.x);
    CHECK(blue_only.z > blue_only.y);
  }
}
