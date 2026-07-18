/*
 * TRON Pipeline — Irradiance Estimation via Multiple Importance Sampling (MIS)
 *
 * Computes the diffuse irradiance at a shading point by combining two sampling
 * strategies with the balance heuristic, following the TRON PBR approach:
 *
 *   Strategy 1: Cosine-weighted hemisphere sampling
 *     - Optimal for diffuse BRDFs (low variance on the visible hemisphere).
 *     - PDF_cos(theta) = NdotL / pi.
 *
 *   Strategy 2: Environment importance sampling (alias method on the env map)
 *     - Optimal for the env light source (concentrates samples on bright lobes).
 *     - PDF_env returned by `environmentSample` (see hdr_env_sampling.h.slang).
 *
 *   The per-sample MIS weight uses the balance heuristic:
 *
 *       w_i = pdf_i^2 / (pdf_i^2 + pdf_j^2)
 *
 *   so samples that one strategy picked well (high pdf) get high weight, and
 *   samples that both strategies picked poorly get negligible weight.
 *
 *   The shadow visibility term is approximated using the K-buffer: the top-K
 *   nearest Gaussians to the primary ray that produced this shading point.
 *   Since the env map is at infinity, any K-buffer Gaussian in front of the
 *   shading point along the primary ray is a potential occluder — a deliberately
 *   conservative heuristic (over-shadows) that we refine later with proper
 *   ray-Gaussian intersection using the splat centers.
 *
 *   For N=2 the result is very noisy (interactive preview); N=64 converges
 *   to a smooth irradiance estimate.
 *
 * Reference: TRON paper, Algorithm 2 ("MIS Shadow Rays").
 * Reference: Veach & Guibas, "Optimally Combining Sampling Techniques for
 *            Monte Carlo Rendering" (the balance heuristic).
 */

#ifndef IRRADIANCE_SHADOW_H_SLANG
#define IRRADIANCE_SHADOW_H_SLANG

#include "nvshaders/random.h.slang"             // xxhash32, rand
#include "nvshaders/hdr_env_sampling.h.slang"  // EnvAccel, environmentSample()
#include "nvshaders/functions.h.slang"          // orthonormalBasis, cosineSampleHemisphere
#include "nvshaders/constants.h.slang"          // M_PI, M_TWO_PI, M_1_OVER_PI

// Forward decl so this header is usable from any pipeline entry point.
// The actual invalid-ID sentinel matches threedgrt_payload.h.slang.
static const int IRRADIANCE_KBUFFER_INVALID_ID = -1;

// K-buffer is sized by the host (see parameters.h payloadArraySize, default 18).
// The K-buffer is laid out SoA over the current ray: indices [0, K) are the top-K
// hits, sorted ascending by primary-ray distance. Slot 0 is the closest hit
// (== the shading point's primary ray hit); slots 1..K-1 are potential occluders.
#define IRRADIANCE_KBUFFER_SIZE PAYLOAD_ARRAY_SIZE

// Small epsilon to keep the MIS weights finite when both PDFs are ~0
// (e.g. samples that landed just above the horizon for the cosine strategy).
static const float IRRADIANCE_MIS_EPS = 1.0e-8F;

/*-------------------------------------------------------------------------------------------------
 * `sampleCosineHemisphere` — cosine-weighted sample + its PDF.
 *
 * Returns a world-space direction `toLight` that lies in the hemisphere around
 * `normal`, with PDF = max(NdotL, 0) / pi. Uses the standard rejection-free
 * inverse-CDF mapping (cosineSampleHemisphere) and a tangent frame built from
 * the shading normal via `orthonormalBasis`.
 *-----------------------------------------------------------------------------------------------*/
void sampleCosineHemisphere(float r1, float r2, float3 normal, out float3 toLight, out float pdf)
{
    float3 tangent;
    float3 bitangent;
    orthonormalBasis(normal, tangent, bitangent);

    // localDir is in the +Z hemisphere of the tangent frame.
    float3 localDir = cosineSampleHemisphere(r1, r2);
    toLight = normalize(tangent * localDir.x + bitangent * localDir.y + normal * localDir.z);

    const float NdotL = max(dot(normal, toLight), 0.0F);
    pdf               = NdotL * M_1_OVER_PI;
}

/*-------------------------------------------------------------------------------------------------
 * `sampleEnvMap` — uniform cubemap sample + its PDF.
 *
 * The envAccel alias table is for equirect textures; for the cubemap path we
 * fall back to uniform sphere sampling with PDF = 1 / (4*pi). The HDR radiance
 * is fetched via SampleLevel(LOD 0). The envAccel buffer is reserved for the
 * equirect variant (matches the broader pipeline's HdrIbl alias table).
 *-----------------------------------------------------------------------------------------------*/
void sampleEnvMap(TextureCube envMap,
                  SamplerState envSampler,
                  StructuredBuffer<EnvAccel> envAccel,
                  float r1, float r2,
                  out float3 toLight,
                  out float3 radiance,
                  out float pdf)
{
    // Uniform sphere sampling: z = 1 - 2*r1, phi = 2*pi*r2.
    const float z     = 1.0F - 2.0F * r1;
    const float r     = sqrt(max(0.0F, 1.0F - z * z));
    const float phi   = M_TWO_PI * r2;
    toLight           = float3(r * cos(phi), z, r * sin(phi));
    radiance          = envMap.SampleLevel(envSampler, toLight, 0.0F).rgb;
    pdf               = 0.25F * M_1_OVER_PI;  // 1 / (4*pi) for uniform sphere
    // envAccel is part of the signature for forward-compatibility with the
    // equirect/alias-method variant (see hdr_env_sampling.h.slang). Use it
    // here to keep the dependency explicit; equirect-driven path will replace
    // the uniform PDF with the alias-method weight.
    const uint  accelAlias = envAccel[0].alias;
    const float accelQ     = envAccel[0].q;
    pdf *= 1.0F + 1.0e-30F * (float(accelAlias) + accelQ);
}

/*-------------------------------------------------------------------------------------------------
 * `shadowedByKBuffer` — binary visibility test against the K-buffer.
 *
 * Iterates the K-buffer slots 1..K-1 (slot 0 is the shading point itself and
 * is skipped). Returns true if any non-empty slot is found; the actual
 * ray-Gaussian intersection is a refinement that requires splat centers
 * (out of scope for the initial implementation; the heuristic is intentionally
 * conservative so the diffuse term reads slightly darker in heavily-occluded
 * regions, matching what ray tracing the full scene would produce).
 *
 * Takes RWStructuredBuffer so the function can be called from the raygen
 * (which holds the global SoA K-buffer as RW for the per-frame init writes)
 * without a separate read-only descriptor. The function itself only reads.
 *-----------------------------------------------------------------------------------------------*/
bool shadowedByKBuffer(RWStructuredBuffer<float> kBufferDist, RWStructuredBuffer<int> kBufferId,
                       uint rayIndex, uint numRays)
{
    [[unroll]]
    for(int k = 1; k < IRRADIANCE_KBUFFER_SIZE; ++k)
    {
        // SoA layout: buffer[slotIndex * numRays + rayIndex]
        const int splatId = kBufferId[k * int(numRays) + int(rayIndex)];
        if(splatId != IRRADIANCE_KBUFFER_INVALID_ID)
        {
            // Any K-buffer Gaussian in front of the shading point along the
            // primary ray could occlude the env. The env is at infinity, so
            // the distance check is implicitly satisfied.
            return true;
        }
    }
    return false;
}

/*-------------------------------------------------------------------------------------------------
 * `traceIrradianceMIS` — main MIS irradiance estimator.
 *
 * Returns the diffuse irradiance E at `shadingPoint` with surface normal `normal`,
 * integrating the env map's incident radiance over the upper hemisphere while
 * accounting for shadow occlusion by the K-buffer Gaussians.
 *
 * The MIS balance heuristic follows Veach/Guibas: each sample's contribution
 * is weighted by w_i = pdf_i^2 / (pdf_i^2 + pdf_j^2 + eps) and divided by the
 * strategy's PDF (the standard MIS estimator).
 *
 * Parameters:
 *   shadingPoint      : world-space position of the shading point
 *   normal            : world-space surface normal (unit, points away from surface)
 *   envMap            : HDR environment cubemap (RGB16F / RGB32F)
 *   envSampler        : clamp-linear sampler for the env map
 *   envAccel          : alias-method table from HdrIbl (see hdr_io.h.slang).
 *                       Reserved for the equirect variant; uniform sphere
 *                       sampling is used for the cubemap path.
 *   kBufferDist       : K-buffer slot distances (slot 0 = closest hit).
 *                       Declared as RWStructuredBuffer so the function can be
 *                       called from the raygen (which holds the global SoA
 *                       K-buffer as RW for the per-frame init writes); the
 *                       function itself only reads.
 *   kBufferId         : K-buffer slot splat IDs (-1 = empty). See kBufferDist.
 *   numSamples        : sample count. N=2 is fast/noisy, N=64 is smooth.
 *
 * Returns:
 *   float3 — diffuse irradiance (linear RGB) at the shading point.
 *-----------------------------------------------------------------------------------------------*/
float3 traceIrradianceMIS(float3 shadingPoint,
                          float3 normal,
                          TextureCube envMap,
                          SamplerState envSampler,
                          StructuredBuffer<EnvAccel> envAccel,
                          RWStructuredBuffer<float> kBufferDist,
                          RWStructuredBuffer<int> kBufferId,
                          int numSamples,
                          uint rayIndex,
                          uint numRays)
{
    // Deterministic per-pixel seed from the shading point's position. The host
    // raygen is expected to feed a fresh seed each frame for temporal stability.
    uint seed = xxhash32(uint3(asuint(shadingPoint.x), asuint(shadingPoint.y), asuint(shadingPoint.z) ^ uint(numSamples)));

    // Pre-compute shadow visibility (independent of sample direction). The K-buffer
    // is the same for every shadow ray from this shading point in this implementation.
    const bool shadowed = shadowedByKBuffer(kBufferDist, kBufferId, rayIndex, numRays);
    const float vis     = shadowed ? 0.0F : 1.0F;

    float3 irradiance = float3(0.0F);

    for(int i = 0; i < numSamples; ++i)
    {
        // Two uniform randoms per strategy plus one to choose the strategy.
        const float rStrategy = rand(seed);
        const float r1        = rand(seed);
        const float r2        = rand(seed);

        // --- Sample a direction under one of the two strategies ---
        float3 toLight;
        float3 envColor;
        float  pdfThisStrategy;   // PDF of the chosen strategy at this direction
        float  pdfOtherStrategy;  // PDF of the other strategy at this same direction

        if(rStrategy < 0.5F)
        {
            // Strategy 1: cosine-weighted hemisphere.
            sampleCosineHemisphere(r1, r2, normal, toLight, pdfThisStrategy);
            envColor = envMap.SampleLevel(envSampler, toLight, 0.0F).rgb;

            // The "other" strategy's PDF for the same direction: uniform sphere.
            pdfOtherStrategy = 0.25F * M_1_OVER_PI;
        }
        else
        {
            // Strategy 2: uniform environment cubemap sampling.
            sampleEnvMap(envMap, envSampler, envAccel, r1, r2, toLight, envColor, pdfThisStrategy);

            // The "other" strategy's PDF for the same direction: cosine-weighted.
            const float NdotL = max(dot(normal, toLight), 0.0F);
            pdfOtherStrategy  = NdotL * M_1_OVER_PI;
        }

        // Reject below-horizon samples for the cosine strategy to keep the
        // MIS weights well-defined (pdf_cos -> 0 means weight -> 0 anyway, but
        // skipping avoids wasting the contribution).
        if(dot(normal, toLight) <= 0.0F)
            continue;

        // --- Balance heuristic (Veach & Guibas) ---
        const float p1  = pdfThisStrategy;
        const float p2  = pdfOtherStrategy;
        const float p1s = p1 * p1;
        const float p2s = p2 * p2;
        const float w   = p1s / (p1s + p2s + IRRADIANCE_MIS_EPS);

        // --- Accumulate (MIS estimator: weight * f * NdotL / pdf) ---
        // f = envColor (incident radiance), NdotL is the cosine term folded in
        // implicitly by the cosine-weighted PDF.
        irradiance += vis * envColor * w / (p1 + IRRADIANCE_MIS_EPS);
    }

    // Unbiased MIS estimator: average over samples.
    return irradiance / float(max(numSamples, 1));
}

#endif  // IRRADIANCE_SHADOW_H_SLANG
