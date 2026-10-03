# OpenDLSS-NR (vendored)

This directory is a **vendored, patched copy** of [maanHimself/OpenDLSS-NR](https://github.com/maanHimself/OpenDLSS-NR),
an MIT-licensed Vulkan reimplementation of NVIDIA's DLSS 5 Neural Rendering network.

| | |
| --- | --- |
| Upstream | `https://github.com/maanHimself/OpenDLSS-NR.git` |
| Pinned commit | `9d08f4184bbcb9d858e2fb7a7834ec0837a9d2f1` (`chore: fix network description`) |
| Licence | MIT (`LICENSE`); see also `NOTICE` |

## Why vendored instead of a submodule

Upstream targets Windows + MSVC and its PTX route depends on `VK_NV_cuda_kernel_launch`, whose types live in
Vulkan's beta headers. Building it on Linux requires source changes, so a pristine submodule is not possible;
the patches below are small and are kept here rather than in a private fork.

## What is included

`src/`, `shaders/`, `LICENSE`, `NOTICE`, the text documentation, and the NR integration reference from `demo/`
(`nr_pass.h`, `nr_pass.cpp` and `demo/shaders/`). `nr_pass.cpp` is the upstream integration: it is the template
for the viewer's wrapper (context adoption, image ownership, descriptor sets, history parity, the pre-recorded
secondary command buffers and the exact barrier/command order). `demo/shaders/nr_common.glsl` holds
`roundF16`/`truncateHalf`/`NrParams`/`historySample`, which the viewer's own passes reuse.

Excluded, because we neither build nor need them: the rest of `demo/` (Filament), `ports/` (the WebGPU/three.js
port), `scripts/` (PowerShell tooling and the Python PTX generators), `third_party/`, `docs/images/` and the
top-level `README.md`.

## Patches

Local changes, all in `src/`:

1. **`vk_context.h`: add `#include <cstring>`.** `SpecConstants::addFloat` calls `memcpy`; MSVC's headers pull
   `<cstring>` in transitively, GCC's do not.

2. **`vk_context.h`/`.cpp`: make the PTX surface compile without `VK_NV_cuda_kernel_launch`.** That extension's
   `VkCudaModuleNV`/`VkCudaFunctionNV` are declared only in `vulkan_beta.h`, which needs
   `VK_ENABLE_BETA_EXTENSIONS`. A new `NR_HAS_CUDA_LAUNCH` feature macro (default: derived from the headers,
   overridable with `-DNR_HAS_CUDA_LAUNCH=0`) guards the real implementations and substitutes opaque
   pointer-to-incomplete-type handles otherwise. The `Context::cudaLaunch*` entry points then throw, because a
   throw means a caller reached a PTX path the predicates should have disabled.

3. **`kernels.cpp`: force the GLSL route when the PTX route is compiled out.** `ptxGemmEnabled`,
   `ptxQkvEnabled`, `ptxGemmVEnabled`, `ptxBlock32Enabled`, `ptxFfnEnabled`, `ptxGlobalAttentionEnabled` and
   `chainEnabled` return `false` unconditionally under `!NR_HAS_CUDA_LAUNCH`. This makes routing a compile-time
   property rather than a function of `setenv` ordering: upstream's `fusedBlock32Ptx` calls
   `ptxKernel(entry + ".ptx", entry)` without probing for the file and throws when no PTX is present.

4. **`kernels.h`: give `Kernels::Chain` an explicit default constructor.** Its members had default member
   initializers, and GCC will not synthesize a default constructor from those for a default argument used inside
   the enclosing class; `expertFfnPtx(...)` and `qkvAttention(...)` default `const Chain& = Chain()`.

5. **`nr_graph.cpp`: reorder the `Graph` constructor's member-init list.** It did not follow the members'
   declaration order (`routes_` is declared first, well before `context_`/`model_`/`kernels_`/`geometry_`/
   `options_`), which GCC reports as `-Wreorder`. Behaviour is unchanged: `routesFromEnvironment()` only reads
   the environment and depends on none of the other members.

Patches 1, 4 and 5 are pure portability fixes. Patches 2 and 3 are the Linux build deliberately selecting the
GLSL/SPIR-V route; the upstream PTX path remains intact for a Windows build that defines
`NR_HAS_CUDA_LAUNCH=1`.

## Building

`cmake/BuildOpenDLSSNR.cmake` compiles `src/*.cpp` except the `dlss5vk` CLI (`main.cpp`) into a static library
`opendlss_nr`, and compiles the 13 `shaders/*.comp` files to SPIR-V next to the runtime binary with
`glslangValidator -V --target-env vulkan1.3 -I<shaders>`. No PTX is generated or required.

## Model weights

Upstream ships no weights. `nr::Model` reads a model directory (`manifest.json` + packed E4M3 stages); the
loader requires exactly the 71-block network. See `docs/weights.md` for the layout and
`tools/extract_dlssnr_model.py` in this repository for producing such a directory from an
`nvngx_dlssnr.dll`. The weights are NVIDIA's property and are not distributed with this project.
