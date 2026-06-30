# TRON PBR: Physically-Based Ray Tracing for 3D Gaussians

![TRON PBR rendering with environment map lighting](tron_pbr_rendering.png)

We extend the 3D Gaussian Ray Tracing (3DGRT) pipeline with a deferred physically-based shading pass that replaces the SH-based front-to-back compositing with a split-sum image-based lighting (IBL) model using environment maps. This is based on the TRON paper approach for deferred PBR compositing of volumetric Gaussian splats.

## Table of Contents

1. [Principle](#principle)
2. [Deferred G-Buffer Accumulation](#deferred-g-buffer-accumulation)
3. [Split-Sum PBR Evaluation](#split-sum-pbr-evaluation)
4. [MIS Irradiance Estimation](#mis-irradiance-estimation)
5. [AgX Tone Mapping](#agx-tone-mapping)
6. [PLY Material Property Format](#ply-material-property-format)
7. [Command-Line Arguments](#command-line-arguments)
8. [UI Controls](#ui-controls)
9. [Limitations](#limitations)
10. [References](#references)

## Principle

The standard 3DGRT ray tracing pipeline computes radiance by compositing spherical harmonic (SH) coefficients in a front-to-back order using the any-hit shader. The TRON PBR pipeline replaces this with a deferred approach:

1. **G-buffer pass**: Accumulate per-particle material attributes (basecolor, roughness, metallic, normal) into a deferred G-buffer using front-to-back Over compositing, stopping at the median depth (alpha < 0.5).
2. **Shading pass**: Evaluate the Cook-Torrance BRDF against a prefiltered environment map at the median-depth shading point using the split-sum approximation (Karis 2013).
3. **Irradiance pass (optional)**: Compute diffuse irradiance via multiple importance sampling (MIS) with the balance heuristic (Veach 1998), combining cosine-weighted hemisphere sampling and environment importance sampling, with shadow visibility approximated from the ray's K-buffer.
4. **Tone mapping (optional)**: Convert HDR linear output to sRGB display space using the AgX tone mapping operator.

This approach allows Gaussian splats to carry physically meaningful material parameters (basecolor, roughness, metallic) instead of view-dependent SH coefficients, enabling realistic relighting under arbitrary HDR environment maps.

![PBR pipeline data flow: G-buffer accumulation -> split-sum shading -> MIS irradiance -> tone mapping](pbr_pipeline_flow.png)

## Deferred G-Buffer Accumulation

Instead of blending SH-based radiance directly into the final pixel color, the PBR path accumulates material properties into a `GBufferData` struct:

```
position:    float3  (world-space shading point)
normal:      float3  (world-space, lerped from particle rotation)
basecolor:   float3  (linear RGB albedo)
roughness:   float    (perceptual roughness, [0,1])
metallic:    float    (metalness, [0,1])
alpha_acc:   float    (accumulated opacity, starts at 1.0)
```

The accumulation uses front-to-back Over compositing:

```c
gbuf.normal    = lerp(gbuf.normal, particleNormal, alpha);
gbuf.basecolor = lerp(gbuf.basecolor, particleBasecolor, alpha);
gbuf.roughness = lerp(gbuf.roughness, particleRoughness, alpha);
gbuf.metallic  = lerp(gbuf.metallic, particleMetallic, alpha);
gbuf.alpha_acc *= (1.0 - particleAlpha);
```

The loop terminates once `alpha_acc` drops below 0.5 (median depth). This ensures the shading point is computed at the surface where the Gaussian density is highest, rather than at the first encountered particle.

![G-buffer visualization showing albedo, normal, roughness, and PBR output channels](pbr_gbuffer_channels.png)

## Split-Sum PBR Evaluation

At the median-depth shading point, the Cook-Torrance BRDF is evaluated using the split-sum approximation (Karis 2013). The key insight of split-sum is that the IBL integral can be factored into two independent parts:

1. **Prefiltered environment map**: A convolution of the HDR envmap with the GGX distribution at varying roughness levels, stored as a cubemap mip chain. Sampled at mip level `roughness * maxMipLevel`.

2. **BRDF integration LUT**: A 2D lookup table parameterized by `(NdotV, roughness)` that stores the Fresnel scale (`R`) and bias (`G`) integrals. This encodes the environment-independent part of the split-sum.

The final specular color is:

```c
float3 specular = prefilteredColor * (F0 * brdfLUT.r + brdfLUT.g);
```

The diffuse term uses a simplified Lambertian model:

```c
float3 diffuse = (1.0 - metallic) * basecolor * (1.0 / PI);
```

The reflection vector is rotated by `envMapRotation` (Y-axis rotation in radians) to allow reorienting the environment without re-baking. The final output is multiplied by `envMapExposure` for brightness control.

Key source files:

- `shaders/pbr_shading.h.slang` -- `evaluatePBR()`, `cookTorranceSpecular()`, `D_GGX()`, `G1_GGX()`, `schlickFresnel()`
- `shaders/threedgrt_raytrace.rgen.slang` -- G-buffer loop, PBR evaluation dispatch (lines 492-558)
- `shaders/threedgrt.h.slang` -- `GBufferData` struct, `particleProcessHitGbuffer()`

## MIS Irradiance Estimation

When `--irradianceEnabled 1` is set, an additional Monte Carlo pass estimates the diffuse irradiance at the shading point. This uses the balance heuristic MIS (Veach & Guibas 1998) to combine two sampling strategies:

| Strategy | PDF | Optimal For |
|----------|-----|-------------|
| Cosine-weighted hemisphere | `max(NdotL, 0) / pi` | Diffuse BRDF (low variance on visible hemisphere) |
| Uniform environment sampling | `1 / (4 * pi)` | Bright envmap lobes (alias method) |

For each of the `numSamples` (default 16), a strategy is chosen uniformly at random. The MIS weight for each sample is:

```
w_i = pdf_i^2 / (pdf_i^2 + pdf_j^2 + eps)
```

This is the balance heuristic in power form (Veach 1998, Section 9.3), which gives high weight to samples that one strategy was likely to produce and low weight to improbable samples.

Shadow visibility is approximated using the K-buffer from the primary ray: any K-buffer Gaussian in front of the shading point along the primary ray is considered a potential occluder. This is a deliberately conservative heuristic (it over-shadows) that can be refined with proper ray-Gaussian intersection tests.

Key source file:

- `shaders/irradiance_shadow.h.slang` -- `traceIrradianceMIS()`, `sampleCosineHemisphere()`, `sampleEnvMap()`, `shadowedByKBuffer()`

## AgX Tone Mapping

When `--toneMapEnabled 1` is set, the final HDR linear color passes through the AgX tone mapping operator before display. AgX is a filmic tone mapper developed by Troy Sobotka that preserves hue consistency across the dynamic range while compressing HDR values into [0, 1] for SDR displays.

The implementation follows Benjamin Wrensch's minimal AgX approximation (ported from Three.js):

1. Apply the AgX input transform matrix (AP0 to LogC encoding)
2. Clamp to the [-12.47, 4.03] EV range and normalize
3. Apply a 6th-order sigmoid via Horner's method
4. Apply the inverse AgX output transform matrix
5. The result is in sRGB display space (gamma is baked in)

Key source file:

- `shaders/tonemap.h.slang` -- `applyToneMapping()`, `tonemapAgX()`

## PLY Material Property Format

PLY files can carry per-particle PBR material data as optional vertex properties:

| Property | Type | Description | Default |
|----------|------|-------------|---------|
| `basecolor_0` | float | Linear red albedo | SH DC coefficient R |
| `basecolor_1` | float | Linear green albedo | SH DC coefficient G |
| `basecolor_2` | float | Linear blue albedo | SH DC coefficient B |
| `roughness` | float | Perceptual roughness, [0, 1] | 0.5 |
| `metallic` | float | Metalness, [0, 1] | 0.0 |

The properties are optional. When missing, defaults are applied per-splat:

- **Basecolor**: Falls back to the SH DC coefficient (the per-splat `f_dc_0/1/2`). This means existing 3DGS models without material data still render with a plausible diffuse color under the environment map.
- **Roughness**: Defaults to 0.5 (moderately rough).
- **Metallic**: Defaults to 0.0 (pure dielectric).

All material values are stored and processed in linear space. The PBR pipeline does not require additional gamma correction; AgX tone mapping (when enabled) produces the final sRGB output.

The property names are parsed in `src/splat_loader_fast.cpp` (lines 172-179), and the default fallback logic is at lines 358-401.

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--envmap <path>` | string | (none) | Path to HDR environment map file (.hdr or .exr) |
| `--envmapRotation <float>` | float | 0.0 | Yaw rotation of the environment map in radians (0 to 2pi) |
| `--envmapExposure <float>` | float | 1.0 | Linear exposure multiplier applied to envmap samples |
| `--pbrEnabled <0/1>` | int | 0 | Enable PBR shading pass (requires pipeline 2) |
| `--irradianceEnabled <0/1>` | int | 0 | Enable MIS irradiance estimation with shadow rays |
| `--toneMapEnabled <0/1>` | int | 0 | Enable AgX HDR-to-sRGB tone mapping |

Example:

```bash
./vk_viewer --inputFile scene.ply --pipeline 2 \\
  --pbrEnabled 1 --irradianceEnabled 0 --toneMapEnabled 1 \\
  --envmap studio_small_07_4k.hdr --envmapRotation 1.57 --envmapExposure 0.8
```

The CLI arguments are registered in `src/parameters.cpp` (lines 131-142) and wired to the shader's `FrameInfo` uniform in `src/vk_viewer.cpp` (lines 229-238).

## UI Controls

The PBR settings are exposed in the **Properties** panel under the **Renderer** section as a **PBR Settings** collapsible header:

| Control | Type | Description |
|---------|------|-------------|
| **Envmap** | Load/Clear button | Select an .hdr or .exr file for the HDR environment map |
| **PBR shading** | Checkbox | Enable the deferred PBR evaluation pass |
| **Irradiance** | Checkbox | Enable diffuse MIS irradiance estimation (disabled when PBR off) |
| **Tone mapping** | Checkbox | Enable AgX HDR-to-sRGB tone mapping |
| **Rotation** | Slider, 0..2pi | Y-axis rotation of the environment cubemap |
| **Exposure** | Slider, 0.1..10 | Linear exposure multiplier |

The UI is implemented in `src/vk_viewer_ui_renderer.cpp` (`guiDrawPbrSettings()`, lines 667-744).

**Important**: Loading or clearing an envmap triggers a shader recompilation (the HDR pipeline shaders are recompiled on the fly). When `--pbrEnabled 1` is set via CLI and a valid envmap was loaded, the scene re-renders immediately with PBR shading on frame 0. There is no temporal accumulation, so the first frame already shows the final result.

## Limitations

- **3DGRT pipeline only**: PBR mode requires `--pipeline 2` (3D Gaussian Ray Tracing). It does not work with rasterization (pipeline 0/1), 3DGUT (pipeline 4), hybrid (pipeline 3/5), or stochastic GS (pipeline 6) pipelines.

- **No temporal accumulation**: Each frame is shaded independently. There is no temporal sample accumulation or denoising for the PBR evaluation (unlike the standard SH path which uses frame accumulation for depth of field). This means higher sample counts for MIS irradiance are required for smooth results, which impacts performance.

- **Conservative shadow heuristic**: The K-buffer shadow test in the MIS irradiance pass over-shadows. Particles in the K-buffer that happened to be in front of the shading point along the primary ray direction are treated as occluders even if they do not block the environment sample direction. This is a heuristic simplification; proper ray-Gaussian intersection tests would be more accurate.

- **No secondary PBR bounces**: PBR evaluation occurs at the primary surface only. Reflections, refractions, and indirect lighting bounces (which the base 3DGRT pipeline supports for mesh materials) do not benefit from PBR shading on the secondary ray path.

- **Fixed envmap sampling resolution**: The MIS irradiance estimator uses N=16 samples (hardcoded in the shader) regardless of scene complexity. Environments with high-frequency lighting (sharp shadows, small bright sources) require more samples for convergence.

- **Surface normal from Gaussian orientation**: The shading normal is interpolated from the Gaussian particle's orientation (the third row of `particleInvRotation`). There are no per-particle normal maps or bump mapping. The median-depth interpolation of normals across splats produces a smooth surface normal that approximates the underlying scene geometry.

- **GGX-only BRDF**: The specular BRDF uses a single-lobe GGX microfacets model. There is no clearcoat, sheen, anisotropy, subsurface scattering, or emissive term.

## References

- **TRON Paper**: "TRON: Physically-Based Rendering of Volumetric Gaussian Splats." Describes the deferred G-buffer accumulation, split-sum PBR evaluation, and MIS shadow ray approach used in this implementation. See Algorithm 2 ("MIS Shadow Rays").

- **[[Karis2013](https://cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf)]** "Real Shading in Unreal Engine 4." Brian Karis. SIGGRAPH 2013. Introduces the split-sum approximation for IBL: factoring the BRDF integral into a prefiltered environment map lookup and a 2D BRDF integration LUT.

- **[[Veach1998](https://graphics.stanford.edu/papers/veach_thesis/)]** "Robust Monte Carlo Methods for Light Transport Simulation." Eric Veach. PhD Thesis, Stanford University, 1998. Chapter 9 describes the balance heuristic for combining multiple sampling strategies in MIS.

- **AgX Tone Mapping**: Troy Sobotka's AgX filmic tone mapping curve, as implemented in [Filament](https://github.com/google/filament) and ported to [Three.js](https://github.com/mrdoob/three.js/) by Benjamin Wrensch. See `shaders/tonemap.h.slang` for the minimal implementation.

- **[[Cook1982](https://dl.acm.org/doi/10.1145/357290.357293)]** "A Reflectance Model for Computer Graphics." Robert L. Cook and Kenneth E. Torrance. ACM TOG 1982. The Cook-Torrance microfacets BRDF model with the Trowbridge-Reitz (GGX) normal distribution function.

For general 3D Gaussian Splatting and 3DGRT references, see the main [References](../readme.md#References) section.

## Continue Reading

1. [VK3DGRT: Efficient 3D Gaussian Ray Tracing (3DGRT) using Vulkan RTX](ray_tracing_3d_gaussians.md)
2. [VK3DGHR: 3D Gaussian Hybrid Rendering Using Vulkan RTX and Rasterization](hybrid_rendering_3d_gaussians.md)
