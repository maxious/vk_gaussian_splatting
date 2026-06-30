/*
 * TRON Pipeline — Cook-Torrance PBR Shading
 *
 * Implements physically-based shading using the split-sum approximation
 * (Karis 2013, "Real Shading in Unreal Engine 4"):
 *   - Specular IBL:  prefilteredEnvMap * (F0 * LUT.r + LUT.g)
 *   - Diffuse IBL:   Lambert * basecolor * (1 - metallic)
 *
 * This is a self-contained, deterministic PBR eval — no random sampling
 * required. Designed for rasterization-style forward shading.
 */

#ifndef PBR_SHADING_H_SLANG
#define PBR_SHADING_H_SLANG

#include "color.h.slang"

// --- Constants ---
// Match the value used in nvpro_core2 nvshaders/functions.h.slang
// Use preprocessor define so the #ifndef guard works when this header is
// included alongside nvpro_core2's constants.h.slang (which also defines M_PI).
#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// Default mip count for a 256x256x6 cubemap: floor(log2(256)) = 8 levels (0..7).
// Pre-filtered mip chain typically goes one level higher, so MAX_MIP_LEVEL = 9.0F covers
// the full range from rough (highest mip) to mirror (mip 0).
// Override this in the calling translation unit if your prefilter has a different count.
#ifndef MAX_MIP_LEVEL
#define MAX_MIP_LEVEL 9.0F
#endif

// --- GGX Normal Distribution Function (D) ---
// Trowbridge-Reitz / GGX NDF; roughness in [0,1].
float D_GGX(float NdotH, float roughness)
{
    float a  = roughness * roughness;
    float a2 = a * a;
    float f  = NdotH * NdotH * (a2 - 1.0F) + 1.0F;
    return a2 / (M_PI * f * f);
}

// --- Smith Geometry / Shadowing-Masking (G) ---
// Height-correlated Smith with the GGX formulation. G1 is a clamped version
// of the height-correlated form (Karasev / Hammon); the product G1(NdotV) * G1(NdotL)
// is the standard uncorrelated Smith G_Smith used by the Simple BSDF path.
float G1_GGX(float NdotV, float roughness)
{
    float a  = roughness * roughness;
    float a2 = a * a;
    float denom = NdotV + sqrt(a2 + (1.0F - a2) * NdotV * NdotV);
    return (2.0F * NdotV) / denom;
}

float G_Smith(float NdotV, float NdotL, float roughness)
{
    return G1_GGX(NdotV, roughness) * G1_GGX(NdotL, roughness);
}

// --- Schlick Fresnel (F) ---
// F0 is the reflectance at normal incidence (dielectric 0.04 or basecolor for metals).
// VdotH is the cosine of the angle between view and half vector.
float3 schlickFresnel(float3 F0, float VdotH)
{
    return F0 + (1.0F - F0) * pow(1.0F - VdotH, 5.0F);
}

// --- Cook-Torrance Specular BRDF ---
// Per-direction eval (requires precomputed NdotV/NdotL/NdotH/VdotH).
// Returns the ratio f_spec / (4 * NdotV * NdotL) ready to be weighted by NdotL.
float3 cookTorranceSpecular(float NdotV, float NdotL, float NdotH, float VdotH, float roughness, float3 F0)
{
    float  D = D_GGX(NdotH, roughness);
    float  G = G_Smith(NdotV, NdotL, roughness);
    float3 F = schlickFresnel(F0, VdotH);
    return (D * G * F) / max(4.0F * NdotV * NdotL, 1e-6F);
}

// --- Main PBR evaluation (split-sum IBL) ---
// Combines specular IBL (prefiltered envmap) + diffuse (Lambert * basecolor).
// Uses the Unreal Engine 4 split-sum approximation (Karis 2013).
//
// Parameters:
//   basecolor        : linear RGB albedo
//   roughness        : [0,1], perceptually-linear
//   metallic         : [0,1]
//   N                : world-space normal (unit)
//   V                : world-space view direction (unit, points from surface to camera)
//   prefilteredEnvMap: roughness-prefiltered cubemap (mip 0 = mirror, MAX_MIP_LEVEL = rough)
//   envSampler       : sampler for the envmap (clamp + linear)
//   brdfLUT          : 2D RG LUT (R = scale, G = bias), lookup by (NdotV, roughness)
//   lutSampler       : sampler for the BRDF LUT (clamp + linear)
//   envRotation      : Y-axis rotation in radians applied to the reflected direction
//   exposure         : linear exposure multiplier applied at the end
float3 evaluatePBR(
    float3 basecolor,
    float roughness,
    float metallic,
    float3 N,
    float3 V,
    TextureCube prefilteredEnvMap,
    SamplerState envSampler,
    Texture2D brdfLUT,
    SamplerState lutSampler,
    float envRotation,
    float exposure)
{
    float NdotV = max(dot(N, V), 0.0F);

    // Reflect view direction around the normal; this is the direction we sample the envmap in.
    float3 R = reflect(-V, N);

    // Apply envmap rotation (Y-axis) to the reflection vector.
    // Convention: positive envRotation rotates the dome clockwise when viewed from +Y.
    float cosR = cos(envRotation);
    float sinR = sin(envRotation);
    float3 Rrot = float3(R.x * cosR - R.z * sinR, R.y, R.x * sinR + R.z * cosR);

    // F0: dielectric (0.04) blended with basecolor for metals.
    float3 F0 = lerp(float3(0.04F, 0.04F, 0.04F), basecolor, metallic);

    // Sample the roughness-prefiltered envmap at a mip level that matches surface roughness.
    // Unreal's convention: mip index = roughness * maxMip (linear, no sqrt).
    float NdotVClamped = max(NdotV, 0.001F);
    float mipLevel     = roughness * MAX_MIP_LEVEL;
    float3 prefilteredColor = prefilteredEnvMap.SampleLevel(envSampler, Rrot, mipLevel).rgb;

    // Sample the BRDF LUT: (NdotV, roughness) -> (scale, bias).
    // The .r channel is the Fresnel-scale integral; .g is the Fresnel-bias integral.
    float2 brdf = brdfLUT.SampleLevel(lutSampler, float2(NdotVClamped, roughness), 0.0F).rg;

    // Split-sum specular: integrate BRDF * cosTheta over the hemisphere.
    // The (F0 * scale + bias) form absorbs the Schlick F into the preintegrated LUT.
    float3 specularColor = prefilteredColor * (F0 * brdf.r + brdf.g);

    // Diffuse term: Lambert * (1 - metallic) * (1 - F0_at_normal) * basecolor.
    // Real PBR also pre-filters diffuse IBL; this simplified form uses a constant
    // ambient term, which is acceptable for a forward-shaded rasterization pass.
    float3 kS = F0;
    float3 kD = (1.0F - kS) * (1.0F - metallic);
    float3 diffuseColor = kD * basecolor * (1.0F / M_PI);

    // Combine diffuse + specular, then apply linear exposure.
    return (diffuseColor + specularColor) * exposure;
}

#endif // PBR_SHADING_H_SLANG
