# TRON-Style PBR Ray Tracing Pipeline for vk_viewer

> **🔄 SESSION CONTINUATION (2026-06-30)**: This is the ACTIVE plan. Replaces the abandoned `free-splatter-integration` plan (19/128 tasks done, archived to `archive/free-splatter-integration.ABANDONED-2026-06-30.md`).
>
> **Orchestrator Action Required**: The `boulder.json` file still references the abandoned plan. The user must either:
> 1. Manually edit `.sisyphus/boulder.json` to point `active_plan` at this file and `plan_name` to `tron-pbr-pipeline`, OR
> 2. Re-run `/start-work` to pick up this plan as the new active boulder (it will auto-detect this as the only non-archived plan)
>
> **Note**: Prometheus (the planning agent) cannot modify `boulder.json` directly. This plan is ready for execution; the orchestrator (Sisyphus/atlas) will begin Wave 1 (Phase 0: nvpro_core2 update) on next session.

## TL;DR

> **Quick Summary**: Add ray-traced PBR relighting to vk_viewer's 3DGRT pipeline, based on the TRON paper (arXiv:2606.11314). Per-particle material attributes (basecolor/roughness/metallic) + deferred G-buffer + Cook-Torrance split-sum PBR shading against HDR environment maps + irradiance any-hit shadow rays.
>
> **Deliverables**:
> - Updated `nvpro_core2` submodule to latest upstream main
> - Per-particle material attributes (basecolor, roughness, metallic) in PLY files
> - HDR environment map loading + IBL prefiltering (BRDF LUT, diffuse, glossy)
> - Deferred G-buffer compositing in 3DGRT raygen shader
> - Cook-Torrance split-sum PBR shading pass
> - Irradiance any-hit shadow rays with MIS
> - AgX tone mapping
> - ImGui envmap control panel + CLI args
> - CPU unit tests for PBR math + functional screenshot tests
>
> **Estimated Effort**: Large (8-10 phases, ~40-50 tasks)
> **Parallel Execution**: YES (5 waves)
> **Critical Path**: Phase 0 update → Material storage → IBL infrastructure → G-buffer compositing → PBR shading → UI/Tests

---

## Context

### Original Request
The user extracted the TRON paper (arXiv:2606.11314) and asked to implement the ray-traced PBR pipeline in vk_viewer before the source code is released. The neural renderer portion is blocked on trained Cosmos model weights, so only the ray-traced PBR pipeline (Phases 1-8) is in scope.

### Interview Summary

**Key Decisions**:
- **PLY format**: Extend with optional `basecolor_0/1/2`, `roughness`, `metallic` float properties
- **Material defaults**: basecolor = SH DC color, roughness = 0.5, metallic = 0.0 (when properties missing)
- **Default envmap**: Ship CC0 HDR envmap (Poly Haven studio_small_07_4k) in `_downloaded_resources/`
- **Pipeline scope**: 3DGRT only (`--pipeline 2`)
- **Test strategy**: Functional screenshot tests + CPU math unit tests
- **nvpro_core2 update**: Update to latest upstream main BEFORE TRON work (Phase 0)
- **API strategy**: Adapt to any new API after update

**Research Findings**:
- vk_viewer has 3DGRT ray tracing foundation (any-hit K-buffer, bounce loop, mesh compositing)
- nvpro_core2 provides complete unused IBL stack: `HdrIbl`, `HdrEnvDome`, `bsdfEvaluateSimple`, `environmentSample`
- 5 DLSS-RR G-buffer slots exist but are dead bindings (DIFFUSE_ALBEDO, SPECULAR_ALBEDO, NORMAL_ROUGH, LINEAR_DEPTH, SPEC_HIT_DIST)
- Current 3DGRT does forward SH compositing (not deferred)
- Mesh shading uses Blinn-Phong (not Cook-Torrance) in `wavefront.h.slang`
- Shadow miss shader exists at `threedgrt_raytrace_shadow.rmiss.slang` but is empty
- PLY loader uses `splat_loader_fast.cpp` with property name → offset mapping
- Test infrastructure: `unit_tests` (doctest, CPU-only, runs in CI) + functional screenshot tests (manual/local)

### Metis Review (Gap Analysis)

**Critical Assumptions Validated**:
- HdrIbl/HdrEnvDome are in `nvpro_core2/nvvk/` and `nvpro_core2/nvshaders_host/` (confirmed)
- nvpro_core2 already linked (CMakeLists.txt:379)
- Particle storage pattern: `has_time_data` bool + optional vectors (confirmed)

**Critical Risks Identified**:
1. **K-buffer for shadow ray visibility** — K-buffer answers "which splat is closest?" not "is ray occluded?". Need to decide approach (a/b/c/d) before Phase 5
2. **5 G-buffer slots repurposing** — May conflict with DLSS-RR semantics; need explicit guard
3. **Volumetric blending direction** — Must use front-to-back Over operator, not back-to-front
4. **SH DC as basecolor fallback** — Heuristic, not physically correct; document limitation
5. **Default envmap size** — 4K is ~50MB; consider 1K version for repo size
6. **HDR framebuffer** — May need R16G16B16A16_SFLOAT intermediate target

**Guardrails Set**:
- PLY-only material extension (SoG/KSR/etc. get defaults)
- 3DGRT pipeline only (other pipelines untouched)
- No DLSS-RR integration
- No mesh PBR changes (Blinn-Phong stays)
- Single envmap, no light sources
- PBR at primary surface only
- No temporal accumulation
- No new file formats
- No material editor UI
- No normal maps, emissive, subsurface, anisotropic
- No new acceleration structures (use existing K-buffer or limit to top-K splats)

---

## Work Objectives

### Core Objective
Add ray-traced PBR relighting capabilities to vk_viewer's 3DGRT pipeline using HDR environment maps, enabling dynamic scene relighting, material editing, and improved visual quality for 3DGS scenes.

### Concrete Deliverables
- Updated `nvpro_core2` submodule to latest upstream main
- 6 new PLY property fields (basecolor_0/1/2, roughness, metallic) with backward-compatible defaults
- 3 new GPU buffer/texture types for material attributes
- 4 new shader files: `pbr_shading.h.slang`, `irradiance_shadow.h.slang`, `tonemap.h.slang`, `pbr_defines.h.slang`
- 3 new CLI args: `--envmap <path>`, `--envmapRotation <float>`, `--envmapExposure <float>`
- 1 new ImGui panel: "PBR Environment" with envmap selection, rotation, exposure, PBR/legacy toggle
- 1 default HDR envmap file in `_downloaded_resources/`
- 1 new CPU unit test file: `tests/test_pbr_helpers.cpp`
- 1 new functional screenshot test: `tests/run_pbr_screenshot_test.sh`
- 1 new doc page: `doc/pbr_ray_tracing.md`
- Updated `AGENTS.md` with new options

### Definition of Done
- [ ] `nvpro_core2` updated to latest upstream main, build passes with no errors
- [ ] All existing tests pass (`ctest --output-on-failure`)
- [ ] Default HDR envmap downloaded and verified
- [ ] PLY files with new material properties load correctly
- [ ] PLY files without material properties load with defaults (backward compat)
- [ ] PBR pipeline produces visible relighting when envmap is swapped
- [ ] PBR pipeline matches expected Cook-Torrance behavior (validated by CPU unit tests)
- [ ] ImGui envmap panel controls all PBR parameters
- [ ] CLI args override ImGui settings
- [ ] Screenshot test produces non-black 800x600 PNG
- [ ] Build succeeds in Debug + Release
- [ ] No memory leaks (Vulkan validation layers clean)
- [ ] Performance: ≥1 FPS at 800x600 on RTX 3080+ (target; not blocker)

### Must Have
- Phase 0: nvpro_core2 update + build verification
- Per-particle material attributes with backward-compat defaults
- HDR envmap loading (HdrIbl)
- IBL prefiltering (HdrEnvDome → BRDF LUT, diffuse, glossy)
- Deferred G-buffer compositing (position, normal, basecolor, roughness, metallic)
- Cook-Torrance split-sum PBR shading (using `bsdfEvaluateSimple` or custom)
- MIS shadow rays for irradiance (using top-K splats from K-buffer)
- AgX tone mapping
- ImGui envmap control panel
- CLI args (--envmap, --envmapRotation, --envmapExposure)
- Default HDR envmap shipped in repo
- CPU unit tests for BRDF + MIS math
- Functional screenshot test

### Must NOT Have (Guardrails)
- ❌ Neural renderer integration (Cosmos 0.6B — blocked on weights)
- ❌ Prior-informed reconstruction (DiffusionRenderer — separate pipeline)
- ❌ Envmap optimization (Algorithm 2 — only for training data)
- ❌ 3DGS rasterization pipeline changes
- ❌ 3DGUT pipeline changes
- ❌ Mesh PBR upgrade (Blinn-Phong stays)
- ❌ DLSS-RR integration
- ❌ Multiple envmaps / light sources
- ❌ Material editor UI
- ❌ Normal maps, emissive, subsurface, anisotropic
- ❌ New acceleration structures (BVH, BLAS for shadow rays)
- ❌ Temporal accumulation for PBR
- ❌ New file formats (.mat, JSON material files)
- ❌ Throwing exceptions (use LOGE + return false)

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** — All verification is agent-executed.

### Test Decision
- **Infrastructure exists**: YES (doctest + functional screenshot tests)
- **Automated tests**: YES (CPU unit tests run in CI; screenshot tests are manual/local)
- **Framework**: doctest (CPU) + bash + Python PIL (screenshot)
- **TDD orientation**: CPU math helpers ported from Slang to C++ first, then tested, then back to Slang

### QA Policy
Every implementation task MUST include agent-executed QA scenarios:
- **C++ code**: Build with `cmake --build build --config Debug` and `--config Release`, no warnings
- **Shaders**: Compile via Slang, no errors, no warnings (validation layers enabled in Debug)
- **PLY loading**: Load test PLY files with/without material properties, verify defaults
- **GPU resources**: Vulkan validation layers clean, no resource leaks
- **Rendering**: Screenshot test produces valid PNG, non-black, correct dimensions
- **CPU math**: doctest unit tests pass with known values (BRDF, MIS, gamma)

### Evidence Capture
- Build logs → `.sisyphus/evidence/task-{N}-build-{config}.log`
- Screenshot test outputs → `.sisyphus/evidence/task-{N}-screenshot.png`
- Unit test outputs → `.sisyphus/evidence/task-{N}-unittest.log`
- Vulkan validation output → `.sisyphus/evidence/task-{N}-validation.log`

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Phase 0 — Foundation, sequential, all other phases blocked):
├── Task 1: Update nvpro_core2 submodule to latest main
├── Task 2: Build and fix any API breaks in existing vk_viewer code
└── Task 3: Audit HdrIbl/HdrEnvDome/bsdfEvaluateSimple APIs for changes

Wave 2 (CPU data model — no GPU changes, all parallel):
├── Task 4: Extend SplatSet with material vectors (splat_set.h)
├── Task 5: Extend PLY parser for new properties (splat_loader_fast.cpp)
├── Task 6: Add CPU unit test for PLY material parsing (test_ply_material.cpp)
└── Task 7: Add CPU unit test for BRDF math helpers (test_pbr_helpers.cpp)

Wave 3 (GPU data + IBL infrastructure — all parallel, depends on Wave 1+2):
├── Task 8: Add material GPU buffers/textures to SplatSetVk
├── Task 9: Add material fetch functions in threedgs_particles_storage.h.slang
├── Task 10: Instantiate HdrIbl + HdrEnvDome in vk_viewer
├── Task 11: Download + verify default HDR envmap
└── Task 12: Add IBL bindings to descriptor set + shaderio.h

Wave 4 (PBR shading — depends on Wave 3):
├── Task 13: Add G-buffer output bindings to 3DGRT
├── Task 14: Implement particleProcessHitGbuffer with front-to-back Over
├── Task 15: Implement Cook-Torrance PBR shader (pbr_shading.h.slang)
├── Task 16: Wire PBR shader to use G-buffer + IBL
├── Task 17: Implement MIS shadow ray pass (irradiance_shadow.h.slang)
├── Task 18: Wire shadow ray pass to use top-K K-buffer splats
└── Task 19: Implement AgX tone mapping (tonemap.h.slang)

Wave 5 (Composition + UI + CLI + Tests — depends on Wave 4):
├── Task 20: Compose PBR + irradiance in final output
├── Task 21: Add --envmap, --envmapRotation, --envmapExposure CLI args
├── Task 22: Add ImGui envmap control panel
├── Task 23: Add functional screenshot test (run_pbr_screenshot_test.sh)
├── Task 24: Add CMakeLists.txt entries for new files
├── Task 25: Write doc/pbr_ray_tracing.md
└── Task 26: Update AGENTS.md with new options

Wave FINAL (After all tasks — 4 parallel reviews):
├── Task F1: Plan compliance audit
├── Task F2: Code quality review
├── Task F3: Real manual QA (screenshot tests)
└── Task F4: Scope fidelity check
```

### Dependency Matrix

| Task | Depends On | Blocks |
|------|-----------|--------|
| 1 (update) | — | 2, 3, all subsequent |
| 2 (fix build) | 1 | 3, all subsequent |
| 3 (audit IBL API) | 2 | 10, 12, 15, 17 |
| 4 (SplatSet) | — | 5, 8 |
| 5 (PLY parser) | 4 | 6, 8 |
| 6 (PLY test) | 5 | — |
| 7 (BRDF math test) | 3 | — |
| 8 (GPU buffers) | 4 | 9, 13 |
| 9 (shader fetch) | 8 | 13, 14 |
| 10 (HdrIbl) | 3 | 12, 16, 18 |
| 11 (envmap download) | — | 10 |
| 12 (IBL bindings) | 10, 11 | 13, 16, 18 |
| 13 (G-buffer output) | 9, 12 | 14, 16, 18 |
| 14 (G-buffer compose) | 13 | 16, 18, 20 |
| 15 (PBR shader) | 3 | 16, 20 |
| 16 (wire PBR) | 14, 15 | 20 |
| 17 (MIS shadow) | 3, 13 | 18, 20 |
| 18 (wire shadow) | 17 | 20 |
| 19 (AgX tonemap) | 16, 18 | 20 |
| 20 (compose final) | 16, 18, 19 | 21, 22, 23 |
| 21 (CLI args) | 20 | 22, 23 |
| 22 (ImGui panel) | 20, 21 | 23 |
| 23 (screenshot test) | 20, 21, 22 | 26 |
| 24 (CMakeLists) | 8, 9, 12, 15, 17, 19, 23 | — |
| 25 (doc) | 20, 21, 22 | 26 |
| 26 (AGENTS.md) | 23, 25 | — |

### Agent Dispatch Summary

- **Wave 1**: T1 → `quick` (git submodule), T2 → `unspecified-high` (build fix), T3 → `deep` (API audit)
- **Wave 2**: T4 → `unspecified-high`, T5 → `unspecified-high`, T6 → `unspecified-high`, T7 → `unspecified-high` (all parallel)
- **Wave 3**: T8 → `unspecified-high`, T9 → `unspecified-high`, T10 → `deep`, T11 → `quick`, T12 → `unspecified-high` (all parallel)
- **Wave 4**: T13 → `unspecified-high`, T14 → `ultrabrain`, T15 → `ultrabrain`, T16 → `ultrabrain`, T17 → `ultrabrain`, T18 → `ultrabrain`, T19 → `unspecified-high`
- **Wave 5**: T20 → `unspecified-high`, T21 → `quick`, T22 → `visual-engineering`, T23 → `unspecified-high`, T24 → `quick`, T25 → `writing`, T26 → `writing`
- **FINAL**: F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

> Implementation + Test = ONE Task. Never separate.
> Every task MUST have QA Scenarios.

---

### Wave 1: Phase 0 — nvpro_core2 Update (BLOCKING)

- [x] 1. Update nvpro_core2 submodule to latest upstream main

  **What to do**:
  - `cd nvpro_core2 && git fetch origin && git checkout origin/main`
  - `cd .. && git add nvpro_core2 && git commit -m "chore(deps): update nvpro_core2 to latest upstream main"`
  - Document the previous commit hash and the new commit hash in commit message

  **Must NOT do**:
  - Do NOT modify any nvpro_core2 files
  - Do NOT force-push

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `["git-master"]`
  - **Reason**: Trivial git submodule update operation

  **Parallelization**:
  - **Can Run In Parallel**: NO (blocks all subsequent tasks)
  - **Blocked By**: None

  **References**:
  - `.gitmodules` — submodule configuration
  - `nvpro_core2/CMakeLists.txt` — current version's build config
  - Upstream: `https://github.com/nvpro-samples/nvpro_core2.git`

  **Acceptance Criteria**:
  - [ ] `git -C nvpro_core2 log --oneline -1` shows new commit hash
  - [ ] `git -C nvpro_core2 status` shows clean working tree
  - [ ] `git diff --submodule nvpro_core2` shows updated reference

  **QA Scenarios**:
  ```
  Scenario: Submodule updated to latest main
    Tool: Bash
    Preconditions: Clean working tree, no uncommitted changes
    Steps:
      1. cd nvpro_core2 && git fetch origin
      2. git rev-parse origin/main → record hash
      3. git checkout origin/main
      4. cd .. && git add nvpro_core2
      5. git status → expect modified: nvpro_core2
    Expected Result: Submodule pointer updated, no merge conflicts
    Failure Indicators: Merge conflicts, detached HEAD, uncommitted changes
    Evidence: .sisyphus/evidence/task-1-submodule-update.log
  ```

  **Commit**: YES (C1)
  - Message: `chore(deps): update nvpro_core2 to latest upstream main`
  - Files: `nvpro_core2 (submodule)`
  - Pre-commit: `cmake --build build --config Debug` (will likely fail — that's Task 2)

---

- [ ] 2. Build and fix any API breaks in existing vk_viewer code

  **What to do**:
  - Run `cmake --build build --config Debug` and capture all errors
  - Run `cmake --build build --config Release` and capture all errors
  - For each compilation error in `src/*.cpp` or `src/*.h`:
    - Identify the API change in nvpro_core2
    - Adapt the vk_viewer code to use the new API
    - Document the change in commit message
  - Run all existing unit tests: `ctest --output-on-failure`
  - All tests must pass

  **Must NOT do**:
  - Do NOT modify nvpro_core2 (work around it in vk_viewer)
  - Do NOT skip tests to make them pass

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `["git-master"]`
  - **Reason**: Build fix-up work, may require understanding C++ API changes

  **Parallelization**:
  - **Can Run In Parallel**: NO (must complete before Task 3)
  - **Blocked By**: Task 1

  **References**:
  - `CMakeLists.txt` — vk_viewer build config
  - `src/*.cpp`, `src/*.h` — files that may need fixes
  - `nvpro_core2/` — new API surface

  **Acceptance Criteria**:
  - [ ] `cmake --build build --config Debug` exits 0 with no errors
  - [ ] `cmake --build build --config Release` exits 0 with no errors
  - [ ] `ctest --output-on-failure` all pass
  - [ ] No new warnings introduced (compare warning count before/after)

  **QA Scenarios**:
  ```
  Scenario: Debug build succeeds
    Tool: Bash
    Preconditions: nvpro_core2 updated (Task 1 complete)
    Steps:
      1. cmake --build build --config Debug 2>&1 | tee /tmp/build-debug.log
      2. echo $? → expect 0
    Expected Result: Build succeeds, no errors
    Failure Indicators: Compilation errors, linker errors
    Evidence: .sisyphus/evidence/task-2-build-debug.log

  Scenario: Release build succeeds
    Tool: Bash
    Steps:
      1. cmake --build build --config Release 2>&1 | tee /tmp/build-release.log
      2. echo $? → expect 0
    Expected Result: Build succeeds
    Failure Indicators: Optimization errors, ODR violations
    Evidence: .sisyphus/evidence/task-2-build-release.log

  Scenario: All existing tests pass
    Tool: Bash
    Steps:
      1. ctest --output-on-failure 2>&1 | tee /tmp/ctest.log
      2. grep "100% tests passed" /tmp/ctest.log
    Expected Result: All tests pass
    Failure Indicators: Test failures, crashes
    Evidence: .sisyphus/evidence/task-2-ctest.log
  ```

  **Commit**: YES (C2)
  - Message: `fix(vk_viewer): adapt to nvpro_core2 API changes`
  - Files: `src/*` (any files that needed fixes)
  - Pre-commit: `cmake --build build --config Debug && ctest`

---

- [ ] 3. Audit HdrIbl/HdrEnvDome/bsdfEvaluateSimple APIs for changes

  **What to do**:
  - Read current nvpro_core2 headers: `nvpro_core2/nvvk/hdr_ibl.hpp`, `nvpro_core2/nvshaders_host/hdr_env_dome.hpp`, `nvpro_core2/nvshaders/bsdf_functions.h.slang`, `nvpro_core2/nvshaders/hdr_env_sampling.h.slang`, `nvpro_core2/nvshaders/hdr_integrate_brdf.slang`
  - Compare against the API surface documented in the research findings
  - Document any API changes in a new file: `docs/tron-pbr-api-audit.md`
  - If major changes found, update the plan (Phases 2, 4, 5) to use new API

  **Must NOT do**:
  - Do NOT modify nvpro_core2 source
  - Do NOT skip the audit even if APIs appear unchanged

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: `[]`
  - **Reason**: API research and documentation, requires reading multiple header files

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 4, 5, 6, 7)
  - **Blocked By**: Task 2

  **References**:
  - `nvpro_core2/nvvk/hdr_ibl.hpp` — HdrIbl class
  - `nvpro_core2/nvshaders_host/hdr_env_dome.hpp` — HdrEnvDome class
  - `nvpro_core2/nvshaders/bsdf_functions.h.slang` — bsdfEvaluateSimple function
  - `nvpro_core2/nvshaders/hdr_env_sampling.h.slang` — environmentSample function
  - `nvpro_core2/nvshaders/hdr_integrate_brdf.slang` — BRDF LUT integration

  **Acceptance Criteria**:
  - [ ] `docs/tron-pbr-api-audit.md` exists and documents findings
  - [ ] All 5 header files read and analyzed
  - [ ] Any API changes called out with before/after signatures
  - [ ] If changes found, plan updated accordingly

  **QA Scenarios**:
  ```
  Scenario: All 5 API headers read
    Tool: Read
    Preconditions: nvpro_core2 updated
    Steps:
      1. Read nvvk/hdr_ibl.hpp → record class signature
      2. Read nvshaders_host/hdr_env_dome.hpp → record class signature
      3. Read nvshaders/bsdf_functions.h.slang → find bsdfEvaluateSimple
      4. Read nvshaders/hdr_env_sampling.h.slang → find environmentSample
      5. Read nvshaders/hdr_integrate_brdf.slang → record output format
    Expected Result: All 5 APIs documented in audit file
    Failure Indicators: Missing documentation, outdated API references
    Evidence: .sisyphus/evidence/task-3-api-audit.md (copy of audit file)
  ```

  **Commit**: NO (documentation only, commit with C24)

---

### Wave 2: CPU Data Model (parallel, no GPU changes)

- [ ] 4. Extend SplatSet with material vectors (CPU-only)

  **What to do**:
  - Edit `src/splat_set.h`:
    - Add `bool has_material_data = false;` after `has_time_data`
    - Add `std::vector<float> basecolor = {};` (3 floats per splat: R, G, B)
    - Add `std::vector<float> roughness = {};` (1 float per splat)
    - Add `std::vector<float> metallic = {};` (1 float per splat)
  - Update `clear()`, `merge()`, `reorderByMortonCode()`, `removeBlackSplats()` to handle new vectors
  - GPU upload is a separate task (Task 8)

  **Must NOT do**:
  - Do NOT modify the GPU side yet (Task 8)
  - Do NOT change existing fields' layout

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `[]`
  - **Reason**: Data structure extension, straightforward but must touch all SplatSet methods

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 5, 6, 7)
  - **Blocked By**: None

  **References**:
  - `src/splat_set.h:58-75` — current SplatSet struct
  - `src/splat_set.h` — all methods that iterate over splats
  - `src/splat_set.cpp` — if it exists (or inline in header)

  **Acceptance Criteria**:
  - [ ] `has_material_data` bool added
  - [ ] `basecolor` (vec3), `roughness` (float), `metallic` (float) vectors added
  - [ ] `clear()` resets all new vectors to empty
  - [ ] `merge()` handles missing material data (skip or use defaults)
  - [ ] `reorderByMortonCode()` reorders new vectors
  - [ ] `removeBlackSplats()` filters new vectors
  - [ ] `cmake --build build --config Debug` succeeds

  **QA Scenarios**:
  ```
  Scenario: SplatSet compiles with new fields
    Tool: Bash
    Steps:
      1. cmake --build build --config Debug 2>&1 | grep -E "error|warning"
      2. echo $? → expect 0, no output
    Expected Result: Clean build
    Failure Indicators: Compilation errors in splat_set.h
    Evidence: .sisyphus/evidence/task-4-build.log

  Scenario: Existing PLY load still works
    Tool: Bash
    Steps:
      1. ./_bin/Debug/vk_viewer --inputFile _downloaded_resources/flowers_1/flowers_1.ply --pipeline 2 --screenshot /tmp/legacy.png --screenshotDelay 3.0
      2. file /tmp/legacy.png → expect "PNG image data"
    Expected Result: Legacy scene loads, uses default material
    Failure Indicators: Crash, has_material_data logic error
    Evidence: .sisyphus/evidence/task-4-legacy-load.png
  ```

  **Commit**: YES (C3)
  - Message: `feat(splat): add material vectors to SplatSet (CPU-only)`
  - Files: `src/splat_set.h`
  - Pre-commit: `cmake --build build --config Debug`

---

- [ ] 5. Extend PLY parser for new properties

  **What to do**:
  - Edit `src/splat_loader_fast.cpp`:
    - Add `basecolorOffset[3]`, `roughnessOffset`, `metallicOffset` to `PropertyLayout` struct
    - In `parseHeader()`: add parsing for `basecolor_0`, `basecolor_1`, `basecolor_2`, `roughness`, `metallic` (default to -1 if missing)
    - In `load()`: after temporal data block, add material data extraction:
      - If any material offset != -1, set `has_material_data = true`
      - If `basecolor_0/1/2` present, extract 3 floats per splat
      - If `roughness` present, extract 1 float per splat
      - If `metallic` present, extract 1 float per splat
      - If missing, use defaults: `basecolor = SH DC color`, `roughness = 0.5`, `metallic = 0.0`
    - Log warning if partial material data (some properties present, some missing)

  **Must NOT do**:
  - Do NOT require all 6 properties (backward compat)
  - Do NOT error out on missing properties

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `[]`
  - **Reason**: Parser extension with backward compat, follows existing temporal data pattern

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Task 4)
  - **Blocked By**: Task 4

  **References**:
  - `src/splat_loader_fast.cpp:128-168` — parseHeader() pattern
  - `src/splat_loader_fast.cpp:224-333` — load() extraction pattern
  - `src/splat_loader_fast.cpp` — temporal data extraction (pattern to follow)

  **Acceptance Criteria**:
  - [ ] `basecolor_0/1/2`, `roughness`, `metallic` properties parsed from PLY header
  - [ ] Missing properties use defaults (basecolor = SH DC, roughness = 0.5, metallic = 0.0)
  - [ ] `has_material_data` set correctly (true if any property present)
  - [ ] Warning logged if partial material data
  - [ ] Existing PLY files (no material properties) load without error
  - [ ] PLY files with all 6 properties load with correct values

  **QA Scenarios**:
  ```
  Scenario: Legacy PLY (no material) loads
    Tool: Bash
    Steps:
      1. ./_bin/Debug/vk_viewer --inputFile _downloaded_resources/flowers_1/flowers_1.ply --pipeline 2 --screenshot /tmp/legacy.png --screenshotDelay 3.0
      2. file /tmp/legacy.png → expect "PNG image data, 800 x 600"
    Expected Result: Loads with default material, no crash
    Failure Indicators: Crash, "property not found" error
    Evidence: .sisyphus/evidence/task-5-legacy-ply.png

  Scenario: PLY with material properties loads
    Tool: Bash
    Preconditions: Test PLY file with basecolor/roughness/metallic exists
    Steps:
      1. ./_bin/Debug/vk_viewer --inputFile /tmp/test_with_material.ply --pipeline 2 --screenshot /tmp/material.png --screenshotDelay 3.0
      2. file /tmp/material.png → expect "PNG image data, 800 x 600"
    Expected Result: Loads with material data, no crash
    Failure Indicators: Crash, "has_material_data not set" warning
    Evidence: .sisyphus/evidence/task-5-material-ply.png
  ```

  **Commit**: YES (C4)
  - Message: `feat(ply): parse basecolor/roughness/metallic from PLY`
  - Files: `src/splat_loader_fast.cpp`, `src/splat_loader_fast.h`
  - Pre-commit: `cmake --build build --config Debug`

---

- [ ] 6. Add CPU unit test for PLY material parsing

  **What to do**:
  - Create `tests/test_ply_material.cpp`:
    - Test 1: Parse PLY with no material properties → `has_material_data == false`, vectors empty
    - Test 2: Parse PLY with all 6 material properties → `has_material_data == true`, vectors populated
    - Test 3: Parse PLY with partial material (only roughness) → `has_material_data == true`, roughness populated, basecolor/metallic use defaults
    - Test 4: Verify defaults: basecolor = SH DC, roughness = 0.5, metallic = 0.0
  - Use small in-memory PLY data (no file I/O) for speed
  - Add to `tests/CMakeLists.txt`

  **Must NOT do**:
  - Do NOT use file I/O (use in-memory test data)
  - Do NOT require GPU

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `[]`
  - **Reason**: CPU unit test, follows existing test patterns

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Task 5)
  - **Blocked By**: Task 5

  **References**:
  - `tests/test_stochastic_rendering.cpp` — existing test pattern
  - `tests/doctest.h` — doctest header
  - `tests/CMakeLists.txt` — test target configuration

  **Acceptance Criteria**:
  - [ ] `tests/test_ply_material.cpp` exists
  - [ ] 4 test cases pass
  - [ ] Test target `unit_tests` includes the new file
  - [ ] `ctest --output-on-failure` all pass

  **QA Scenarios**:
  ```
  Scenario: Unit tests pass
    Tool: Bash
    Steps:
      1. cmake --build build --config Debug --target unit_tests
      2. ctest --output-on-failure -R test_ply_material
    Expected Result: All 4 test cases pass
    Failure Indicators: Test failures, crashes
    Evidence: .sisyphus/evidence/task-6-unittest.log
  ```

  **Commit**: YES (C5)
  - Message: `test(ply): add CPU unit test for PLY material parsing`
  - Files: `tests/test_ply_material.cpp`, `tests/CMakeLists.txt`
  - Pre-commit: `ctest --output-on-failure`

---

- [ ] 7. Add CPU unit tests for BRDF + MIS math helpers

  **What to do**:
  - Create `tests/test_pbr_helpers.cpp`:
    - Port the BRDF evaluation logic from `nvpro_core2/nvshaders/bsdf_functions.h.slang` to C++ (CPU-portable version)
    - Test 1: GGX D (normal distribution) — known values for specific (NdotH, roughness)
    - Test 2: Smith G (geometry) — known values for specific (NdotV, NdotL, roughness)
    - Test 3: Schlick F (Fresnel) — known values for specific (VdotH, F0)
    - Test 4: Full Cook-Torrance BRDF — known values
    - Test 5: MIS weight (balance heuristic) — known values for specific (pdfA, pdfB)
    - Test 6: Cosine-weighted hemisphere sampling — verify distribution
    - Test 7: Environment map importance sampling — verify samples are valid
    - Test 8: AgX tone mapping — verify curve at specific input values
  - Use doctest assertions with `doctest::Approx` for floating-point
  - Add to `tests/CMakeLists.txt`

  **Must NOT do**:
  - Do NOT include GPU code
  - Do NOT require Vulkan

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `[]`
  - **Reason**: CPU math tests, requires porting Slang to C++

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 6)
  - **Blocked By**: Task 3 (needs API audit)

  **References**:
  - `nvpro_core2/nvshaders/bsdf_functions.h.slang` — BRDF functions to port
  - TRON paper Section 3.3 — Cook-Torrance formula
  - TRON paper Algorithm 2 — MIS weighting
  - `tests/test_stochastic_rendering.cpp` — test pattern

  **Acceptance Criteria**:
  - [ ] `tests/test_pbr_helpers.cpp` exists
  - [ ] 8 test cases pass
  - [ ] BRDF math matches Slang reference implementation
  - [ ] MIS weight matches balance heuristic formula
  - [ ] `ctest --output-on-failure` all pass

  **QA Scenarios**:
  ```
  Scenario: BRDF math tests pass
    Tool: Bash
    Steps:
      1. cmake --build build --config Debug --target unit_tests
      2. ctest --output-on-failure -R test_pbr_helpers
    Expected Result: All 8 test cases pass
    Failure Indicators: Test failures, math errors
    Evidence: .sisyphus/evidence/task-7-unittest.log

  Scenario: Reference values match Slang implementation
    Tool: Bash
    Steps:
      1. For each test case, compare C++ output against known reference value (from BRDF literature)
      2. Tolerance: 1e-6 for exact math, 1e-4 for sampling
    Expected Result: All values within tolerance
    Failure Indicators: Values outside tolerance
    Evidence: .sisyphus/evidence/task-7-reference-comparison.log
  ```

  **Commit**: YES (C8)
  - Message: `test(pbr): add CPU unit tests for BRDF + MIS math`
  - Files: `tests/test_pbr_helpers.cpp`, `tests/CMakeLists.txt`
  - Pre-commit: `ctest --output-on-failure`

---

### Wave 3: GPU Data + IBL Infrastructure (parallel, depends on Wave 1+2)

- [ ] 8. Add material GPU buffers/textures to SplatSetVk

  **What to do**:
  - Edit `src/splat_set_vk.h`:
    - Add `nvvk::Buffer basecolorBuffer;` (buffer mode)
    - Add `nvvk::Buffer roughnessBuffer;` (buffer mode)
    - Add `nvvk::Buffer metallicBuffer;` (buffer mode)
    - Add `nvvk::Image basecolorMap;` (texture mode)
    - Add `nvvk::Image roughnessMap;` (texture mode)
    - Add `nvvk::Image metallicMap;` (texture mode)
  - Edit `src/splat_set_vk.cpp`:
    - In `initDataBuffers()`: create buffers, upload from SplatSet vectors (if `has_material_data`)
    - In `initDataTextures()`: create images, upload to RGBA textures (pad to 4 components)
    - In `deinitDataBuffers()` / `deinitDataTextures()`: destroy new resources
    - Skip if `!has_material_data` (use defaults in shader)

  **Must NOT do**:
  - Do NOT change the storage format enum
  - Do NOT break backward compat for LCC packed format

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `[]`
  - **Reason**: Vulkan buffer/image management, follows existing pattern

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 9, 10, 11)
  - **Blocked By**: Task 4

  **References**:
  - `src/splat_set_vk.h:147-170` — current GPU resource declarations
  - `src/splat_set_vk.cpp:203-292` — initDataBuffers() pattern
  - `src/splat_set_vk.cpp:592-670` — initDataTextures() pattern

  **Acceptance Criteria**:
  - [ ] 3 new buffers (basecolor, roughness, metallic) in buffer mode
  - [ ] 3 new images (basecolor, roughness, metallic) in texture mode
  - [ ] Upload from SplatSet vectors if `has_material_data`
  - [ ] Skip upload if `!has_material_data`
  - [ ] Destruction in deinit functions
  - [ ] `cmake --build build --config Debug` succeeds

  **QA Scenarios**:
  ```
  Scenario: GPU resources created correctly
    Tool: Bash
    Steps:
      1. cmake --build build --config Debug
      2. ./_bin/Debug/vk_viewer --inputFile _downloaded_resources/flowers_1/flowers_1.ply --pipeline 2 --validation 1 --screenshotDelay 2.0
      3. Check log for "VUID-VkDeviceMemory-allocationCount" or similar
    Expected Result: No validation errors, clean exit
    Failure Indicators: Resource leak warnings, validation errors
    Evidence: .sisyphus/evidence/task-8-validation.log
  ```

  **Commit**: YES (C6)
  - Message: `feat(splat): upload material buffers to GPU`
  - Files: `src/splat_set_vk.h`, `src/splat_set_vk.cpp`
  - Pre-commit: `cmake --build build --config Debug`

---

- [ ] 9. Add material fetch functions in threedgs_particles_storage.h.slang

  **What to do**:
  - Edit `shaders/threedgs_particles_storage.h.slang`:
    - Add buffer declarations: `RWStructuredBuffer<float> basecolorBuffer;`, `roughnessBuffer;`, `metallicBuffer;`
    - Add texture declarations: `Sampler2D basecolorTexture;`, `roughnessTexture;`, `metallicTexture;`
    - Add `fetchBasecolor(in uint splatIndex) -> float3`
    - Add `fetchRoughness(in uint splatIndex) -> float`
    - Add `fetchMetallic(in uint splatIndex) -> float`
    - Use 3-way conditional (LCC packed / textures / buffers)
    - Return defaults (0.5, 0.5, 0.0) for LCC packed (no material data)

  **Must NOT do**:
  - Do NOT change existing fetch functions
  - Do NOT require all material properties present

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `[]`
  - **Reason**: Shader code addition, follows existing 3-way pattern

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 10, 11, 8)
  - **Blocked By**: Task 8

  **References**:
  - `shaders/threedgs_particles_storage.h.slang:108-199` — existing fetch functions (pattern)
  - `shaders/threedgs_particles_storage.h.slang:23-60` — buffer/texture declarations (pattern)

  **Acceptance Criteria**:
  - [ ] `fetchBasecolor()`, `fetchRoughness()`, `fetchMetallic()` added
  - [ ] 3-way conditional handles all storage modes
  - [ ] Defaults returned for LCC packed (no material data)
  - [ ] `cmake --build build --config Debug` succeeds
  - [ ] Vulkan validation clean

  **QA Scenarios**:
  ```
  Scenario: Shaders compile with new fetch functions
    Tool: Bash
    Steps:
      1. cmake --build build --config Debug
      2. ./_bin/Debug/vk_viewer --inputFile _downloaded_resources/flowers_1/flowers_1.ply --pipeline 2 --validation 1
    Expected Result: No shader compilation errors, no validation errors
    Failure Indicators: Slang errors, SPIR-V errors
    Evidence: .sisyphus/evidence/task-9-shader-compile.log
  ```

  **Commit**: YES (C7)
  - Message: `feat(shader): add material fetch functions for particles`
  - Files: `shaders/threedgs_particles_storage.h.slang`
  - Pre-commit: `cmake --build build --config Debug`

---

- [ ] 10. Instantiate HdrIbl + HdrEnvDome in vk_viewer

  **What to do**:
  - Edit `src/vk_viewer.cpp`:
    - Add `nvshaders::HdrIbl m_hdrIbl;` member
    - Add `nvshaders::HdrEnvDome m_hdrEnvDome;` member
    - In `createResources()` (or equivalent init function):
      - Call `m_hdrIbl.init(m_alloc, m_samplerPool, m_queueInfo)`
      - Call `m_hdrEnvDome.init(m_alloc, m_samplerPool, m_queueInfo)`
    - In destructor / deinit:
      - Call `m_hdrIbl.deinit()` and `m_hdrEnvDome.deinit()`
    - Add `loadEnvironment(const std::string& path)` method:
      - Call `m_hdrIbl.loadEnvironment(path)`
      - If success, call `m_hdrEnvDome.create()` to generate BRDF LUT + prefilter
      - Store textures for descriptor binding

  **Must NOT do**:
  - Do NOT load envmap on startup (defer until user selects one)
  - Do NOT crash on missing envmap file

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: `[]`
  - **Reason**: Integration of nvpro_core2 classes, requires understanding init lifecycle

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 8, 9, 11)
  - **Blocked By**: Task 3 (needs API audit)

  **References**:
  - `nvpro_core2/nvvk/hdr_ibl.hpp` — HdrIbl class API
  - `nvpro_core2/nvshaders_host/hdr_env_dome.hpp` — HdrEnvDome class API
  - `docs/tron-pbr-api-audit.md` — API audit results (from Task 3)
  - `src/vk_viewer.cpp` — VkViewer class init/deinit

  **Acceptance Criteria**:
  - [ ] `m_hdrIbl` and `m_hdrEnvDome` members added
  - [ ] `init()` called in createResources()
  - [ ] `deinit()` called in destructor
  - [ ] `loadEnvironment()` method implemented
  - [ ] `cmake --build build --config Debug` succeeds
  - [ ] `cmake --build build --config Release` succeeds

  **QA Scenarios**:
  ```
  Scenario: HdrIbl/HdrEnvDome instantiate correctly
    Tool: Bash
    Steps:
      1. cmake --build build --config Debug
      2. ./_bin/Debug/vk_viewer --inputFile _downloaded_resources/flowers_1/flowers_1.ply --pipeline 2 --validation 1
    Expected Result: No instantiation errors, no validation errors
    Failure Indicators: Constructor errors, init failures
    Evidence: .sisyphus/evidence/task-10-init.log
  ```

  **Commit**: YES (C9)
  - Message: `feat(ibl): instantiate HdrIbl + HdrEnvDome in vk_viewer`
  - Files: `src/vk_viewer.cpp`, `src/vk_viewer.h`
  - Pre-commit: `cmake --build build --config Debug`

---

- [ ] 11. Download + verify default HDR envmap

  **What to do**:
  - Create CMake script to download `studio_small_07_4k.hdr` from Poly Haven
  - Use `FetchContent` or `file(DOWNLOAD ...)` in `CMakeLists.txt`
  - Save to `_downloaded_resources/studio_small_07_4k.hdr`
  - Verify SHA256 hash (document expected hash in CMake script)
  - Create LICENSE file `_downloaded_resources/studio_small_07_4k_LICENSE.txt` with CC0 notice

  **Must NOT do**:
  - Do NOT commit the .hdr file to git (use FetchContent or runtime download)
  - Do NOT use a non-CC0 envmap

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`
  - **Reason**: Simple file download, CMake boilerplate

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 8, 9, 10)
  - **Blocked By**: None

  **References**:
  - `CMakeLists.txt` — existing download patterns
  - `_downloaded_resources/` — existing downloaded resources
  - https://polyhaven.com/a/studio_small_07 — Poly Haven page (CC0)

  **Acceptance Criteria**:
  - [ ] `_downloaded_resources/studio_small_07_4k.hdr` exists after build
  - [ ] SHA256 hash matches documented expected value
  - [ ] LICENSE file exists with CC0 notice
  - [ ] File is gitignored (or in .gitignore)
  - [ ] CMake configure + build succeeds

  **QA Scenarios**:
  ```
  Scenario: Envmap downloaded successfully
    Tool: Bash
    Steps:
      1. rm -rf _downloaded_resources/studio_small_07_4k.hdr
      2. cmake --build build --config Debug
      3. ls -la _downloaded_resources/studio_small_07_4k.hdr → expect exists
      4. file _downloaded_resources/studio_small_07_4k.hdr → expect "HDR image data"
      5. sha256sum _downloaded_resources/studio_small_07_4k.hdr → match expected
    Expected Result: File exists, valid HDR, correct hash
    Failure Indicators: Download failure, hash mismatch, corrupt file
    Evidence: .sisyphus/evidence/task-11-envmap-download.log
  ```

  **Commit**: YES (C10)
  - Message: `chore(resources): download + verify default HDR envmap`
  - Files: `CMakeLists.txt`, `_downloaded_resources/studio_small_07_4k_LICENSE.txt`
  - Pre-commit: `file _downloaded_resources/studio_small_07_4k.hdr`

---

- [ ] 12. Add IBL bindings to descriptor set + shaderio.h

  **What to do**:
  - Edit `shaders/shaderio.h`:
    - Add `#define BINDING_PBR_ENVMAP 37` (HDR envmap, combined image sampler)
    - Add `#define BINDING_PBR_PREFILTERED 38` (prefiltered envmap cubemap)
    - Add `#define BINDING_PBR_BRDF_LUT 39` (BRDF LUT 2D texture)
    - Add to `FrameInfo` struct: `float envMapRotation;`, `float envMapExposure;`
  - Edit `src/vk_viewer_pipelines.cpp`:
    - In `initPipelines()`: add 3 new bindings for IBL textures
    - Write descriptors after envmap loaded
  - Bind IBL textures in RTX pipeline layout (Set 0)

  **Must NOT do**:
  - Do NOT use binding slots 0-36 (already taken)
  - Do NOT create a new descriptor set (extend Set 0)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `[]`
  - **Reason**: Descriptor set extension, follows existing pattern

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Tasks 9, 10, 11)
  - **Blocked By**: Tasks 9, 10, 11

  **References**:
  - `shaders/shaderio.h` — existing BINDING_* definitions
  - `src/vk_viewer_pipelines.cpp` — existing descriptor set creation
  - `nvpro_core2/nvshaders_host/hdr_env_dome.hpp` — texture handles

  **Acceptance Criteria**:
  - [ ] 3 new BINDING_* defines added to shaderio.h
  - [ ] `FrameInfo` struct extended with envMapRotation, envMapExposure
  - [ ] 3 new bindings added to Set 0 descriptor set
  - [ ] Descriptors written after envmap loaded
  - [ ] `cmake --build build --config Debug` succeeds
  - [ ] Vulkan validation clean

  **QA Scenarios**:
  ```
  Scenario: IBL bindings compile and link
    Tool: Bash
    Steps:
      1. cmake --build build --config Debug
      2. ./_bin/Debug/vk_viewer --inputFile _downloaded_resources/flowers_1/flowers_1.ply --pipeline 2 --validation 1
    Expected Result: No descriptor errors, no validation errors
    Failure Indicators: Descriptor set errors, validation errors
    Evidence: .sisyphus/evidence/task-12-bindings.log
  ```

  **Commit**: YES (C11)
  - Message: `feat(descriptors): add IBL bindings (envmap, prefiltered, BRDF LUT)`
  - Files: `src/vk_viewer_pipelines.cpp`, `shaders/shaderio.h`
  - Pre-commit: `cmake --build build --config Debug`

---

### Wave 4: PBR Shading (depends on Wave 3)

- [ ] 13. Add G-buffer output bindings to 3DGRT

  **What to do**:
  - Edit `shaders/shaderio.h`:
    - Add `#define RTX_BINDING_GBUFFER_ALBEDO 13` (RGBA16F, basecolor RGB + alpha)
    - Add `#define RTX_BINDING_GBUFFER_NORMAL 14` (RGBA16F, world normal XYZ + depth)
    - Add `#define RTX_BINDING_GBUFFER_PBR 15` (RGBA16F, roughness, metallic, unused, unused)
  - Edit `src/vk_viewer_rtx.cpp`:
    - Add 3 new `nvvk::Image` members for G-buffer outputs
    - Create images in `createRtResources()` (or equivalent)
    - Add bindings to RTX descriptor set (Set 1, slots 13-15)
    - Destroy in deinit

  **Must NOT do**:
  - Do NOT use DLSS-RR slots (6-11) for PBR G-buffer
  - Do NOT change existing RTX bindings

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `[]`
  - **Reason**: Descriptor set + image creation, follows existing pattern

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 15, 17)
  - **Blocked By**: Tasks 9, 12

  **References**:
  - `src/vk_viewer_rtx.cpp` — existing RTX descriptor set
  - `shaders/shaderio.h` — existing RTX_BINDING_* definitions
  - `shaders/threedgrt_raytrace.rgen.slang` — existing output writes

  **Acceptance Criteria**:
  - [ ] 3 new RTX_BINDING_* defines added
  - [ ] 3 new images created with RGBA16F format
  - [ ] 3 new bindings added to Set 1
  - [ ] Images destroyed in deinit
  - [ ] `cmake --build build --config Debug` succeeds
  - [ ] Vulkan validation clean

  **QA Scenarios**:
  ```
  Scenario: G-buffer images created correctly
    Tool: Bash
    Steps:
      1. cmake --build build --config Debug
      2. ./_bin/Debug/vk_viewer --inputFile _downloaded_resources/flowers_1/flowers_1.ply --pipeline 2 --validation 1
    Expected Result: No image creation errors, no validation errors
    Failure Indicators: Format errors, memory allocation errors
    Evidence: .sisyphus/evidence/task-13-gbuffer-images.log
  ```

  **Commit**: YES (C12)
  - Message: `feat(descriptors): add G-buffer output bindings to 3DGRT`
  - Files: `src/vk_viewer_rtx.cpp`, `shaders/shaderio.h`
  - Pre-commit: `cmake --build build --config Debug`

---

- [ ] 14. Implement particleProcessHitGbuffer with front-to-back Over

  **What to do**:
  - Edit `shaders/threedgrt.h.slang`:
    - Add new function: `bool particleProcessHitGbuffer(in FrameInfo frameInfo, in float3 modelRayOrigin, in float3 modelRayDirection, in int splatId, inout GBufferData gbuffer)`
    - `GBufferData` struct: `float3 position; float3 normal; float3 basecolor; float roughness; float metallic; float alpha_acc;`
    - Fetch particle PSR (same as `particleProcessHit`)
    - Fetch geometric normal: `n = Rᵀ · [0,0,1]` (TRON paper Eq. 4)
    - Fetch basecolor, roughness, metallic
    - Apply front-to-back Over operator:
      - `weight = alpha * alpha_acc`
      - `gbuffer.position += (particlePosition - gbuffer.position) * weight`
      - `gbuffer.normal += (normal - gbuffer.normal) * weight` (then re-normalize)
      - `gbuffer.basecolor += (basecolor - gbuffer.basecolor) * weight`
      - `gbuffer.roughness += (roughness - gbuffer.roughness) * weight`
      - `gbuffer.metallic += (metallic - gbuffer.metallic) * weight`
      - `gbuffer.alpha_acc *= (1.0 - alpha)`
    - Return acceptHit

  **Must NOT do**:
  - Do NOT use back-to-front blending (wrong for volumetric)
  - Do NOT skip front-to-back compositing

  **Recommended Agent Profile**:
  - **Category**: `ultrabrain`
  - **Skills**: `[]`
  - **Reason**: Complex blending math, requires understanding volumetric compositing

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 15, 17)
  - **Blocked By**: Task 13

  **References**:
  - `shaders/threedgrt.h.slang:130-195` — existing `particleProcessHit` (pattern)
  - TRON paper Section 3.1 (Eq. 4) — geometric normal formula
  - TRON paper Section 3.2 — deferred shading strategy
  - TRON paper Appendix Sec. 4 — median-depth cutoff for shading point

  **Acceptance Criteria**:
  - [ ] `particleProcessHitGbuffer()` implemented with front-to-back Over
  - [ ] Geometric normal computed correctly: `n = Rᵀ · [0,0,1]`
  - [ ] Position, normal, basecolor, roughness, metallic composited
  - [ ] Median-depth cutoff implemented (alpha_acc < 0.5)
  - [ ] `cmake --build build --config Debug` succeeds
  - [ ] Vulkan validation clean

  **QA Scenarios**:
  ```
  Scenario: G-buffer compositing produces correct output
    Tool: Bash
    Steps:
      1. cmake --build build --config Debug
      2. ./_bin/Debug/vk_viewer --inputFile _downloaded_resources/flowers_1/flowers_1.ply --pipeline 2 --validation 1 --screenshot /tmp/gbuffer.png --screenshotDelay 3.0
      3. Compare G-buffer output against reference (manually verified for first frame)
    Expected Result: G-buffer values match expected (basecolor ≈ SH DC, roughness ≈ 0.5, metallic ≈ 0.0 for legacy PLY)
    Failure Indicators: Compositing errors, NaN values
    Evidence: .sisyphus/evidence/task-14-gbuffer-output.png
  ```

  **Commit**: YES (C13)
  - Message: `feat(shader): add particleProcessHitGbuffer with front-to-back Over`
  - Files: `shaders/threedgrt.h.slang`
  - Pre-commit: `cmake --build build --config Debug`

---

- [ ] 15. Implement Cook-Torrance PBR shader

  **What to do**:
  - Create new file `shaders/pbr_shading.h.slang`:
    - Include `bsdf_functions.h.slang` from nvpro_core2 (or port simple BSDF)
    - Implement `float3 cookTorrancePBR(float3 basecolor, float roughness, float metallic, float3 N, float3 V, TextureCube prefilteredEnv, Sampler envSampler, Texture2D brdfLUT, Sampler lutSampler, float envRotation, float exposure)`
    - Split-sum approximation:
      - F0 = lerp(0.04, basecolor, metallic)
      - F = F0 + (1 - F0) * pow(1 - VdotH, 5)
      - NDF = D_GGX(NdotH, roughness)
      - G = G_Smith(NdotV, NdotL, roughness)
      - Specular = (NDF * G * F) / (4 * NdotV * NdotL)
      - kS = F
      - kD = (1 - kS) * (1 - metallic)
      - Prefiltered envmap query: `textureLod(prefilteredEnv, reflect(-V, N), roughness * MAX_LOD).rgb`
      - BRDF LUT query: `texture(brdfLUT, vec2(NdotV, roughness)).rgb`
      - Specular IBL = prefiltered * (F * brdfLut.r + brdfLut.g)
      - Diffuse IBL = basecolor * irradiance (from shadow ray pass, Task 18)
      - Final = (kD * diffuse + specular) * exposure
    - Add `float3 applyEnvMapRotation(float3 dir, float rotation)` helper
  - Document expected behavior in comments

  **Must NOT do**:
  - Do NOT use Schlick approximation for F (use full Fresnel)
  - Do NOT skip metallic blend (metals have no diffuse)

  **Recommended Agent Profile**:
  - **Category**: `ultrabrain`
  - **Skills**: `[]`
  - **Reason**: Complex PBR math, requires deep understanding of Cook-Torrance

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 14, 17)
  - **Blocked By**: Task 3 (needs API audit)

  **References**:
  - TRON paper Section 3.3 — split-sum approximation
  - TRON paper Appendix Sec. 3 — PBR preliminaries (Cook-Torrance, split-sum)
  - `nvpro_core2/nvshaders/bsdf_functions.h.slang` — reference BSDF implementation
  - `nvpro_core2/nvshaders/hdr_integrate_brdf.slang` — BRDF LUT format
  - Karis 2013 — "Real Shading in Unreal Engine 4" (split-sum paper)

  **Acceptance Criteria**:
  - [ ] `shaders/pbr_shading.h.slang` exists
  - [ ] Cook-Torrance BRDF implemented (D, G, F)
  - [ ] Split-sum approximation: prefiltered envmap × BRDF LUT
  - [ ] Metallic blend: F0 = lerp(0.04, basecolor, metallic)
  - [ ] Envmap rotation applied
  - [ ] Exposure applied
  - [ ] `cmake --build build --config Debug` succeeds
  - [ ] CPU unit tests from Task 7 pass (validates math)

  **QA Scenarios**:
  ```
  Scenario: PBR shader compiles
    Tool: Bash
    Steps:
      1. cmake --build build --config Debug
      2. ./_bin/Debug/vk_viewer --inputFile _downloaded_resources/flowers_1/flowers_1.ply --pipeline 2 --validation 1
    Expected Result: No shader errors, no validation errors
    Failure Indicators: Slang errors, SPIR-V errors
    Evidence: .sisyphus/evidence/task-15-pbr-compile.log

  Scenario: PBR math matches CPU reference
    Tool: Bash
    Steps:
      1. Compare shader output (via screenshot) against CPU reference implementation
      2. Tolerance: mean pixel diff < 5/255 for legacy PLY (no material change)
    Expected Result: Visually similar (PBR with default material ≈ SH compositing)
    Failure Indicators: Large visual differences
    Evidence: .sisyphus/evidence/task-15-pbr-comparison.png
  ```

  **Commit**: YES (C14)
  - Message: `feat(shader): add Cook-Torrance PBR shader`
  - Files: `shaders/pbr_shading.h.slang`
  - Pre-commit: `cmake --build build --config Debug && ctest`

---

- [ ] 16. Wire PBR shading to G-buffer + IBL

  **What to do**:
  - Edit `shaders/threedgrt_raytrace.rgen.slang`:
    - In main raygen, after multi-pass any-hit loop, check if `frameInfo.pbrEnabled != 0`
    - If yes:
      - Call `particleProcessHitGbuffer()` to populate G-buffer
      - At shading point (median depth), call `cookTorrancePBR()` from pbr_shading.h.slang
      - Write result to `gbufferAlbedo` (or use as radiance)
    - If no: use existing SH compositing
    - Add `#include "pbr_shading.h.slang"` at top
  - Edit `shaders/shaderio.h`:
    - Add `int32_t pbrEnabled` to `FrameInfo` struct

  **Must NOT do**:
  - Do NOT break legacy SH compositing
  - Do NOT enable PBR by default (set `pbrEnabled = 0` initially)

  **Recommended Agent Profile**:
  - **Category**: `ultrabrain`
  - **Skills**: `[]`
  - **Reason**: Complex shader integration, requires understanding existing raygen flow

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Tasks 14, 15)
  - **Blocked By**: Tasks 14, 15

  **References**:
  - `shaders/threedgrt_raytrace.rgen.slang:178-429` — existing bounce loop
  - `shaders/threedgrt.h.slang:130-195` — `particleProcessHit` (pattern)
  - `shaders/pbr_shading.h.slang` — new PBR function (from Task 15)

  **Acceptance Criteria**:
  - [ ] PBR shading called when `pbrEnabled != 0`
  - [ ] Legacy SH compositing when `pbrEnabled == 0`
  - [ ] G-buffer used for PBR shading point
  - [ ] PBR output written to final radiance
  - [ ] `cmake --build build --config Debug` succeeds
  - [ ] Vulkan validation clean

  **QA Scenarios**:
  ```
  Scenario: PBR path produces visible output
    Tool: Bash
    Steps:
      1. ./_bin/Debug/vk_viewer --inputFile _downloaded_resources/flowers_1/flowers_1.ply --pipeline 2 --pbrEnabled 1 --screenshot /tmp/pbr.png --screenshotDelay 3.0
      2. file /tmp/pbr.png → expect "PNG image data, 800 x 600"
      3. Mean brightness > 10 (non-black)
    Expected Result: PBR shading produces visible output
    Failure Indicators: Black output, crashes
    Evidence: .sisyphus/evidence/task-16-pbr-enabled.png

  Scenario: Legacy path still works
    Tool: Bash
    Steps:
      1. ./_bin/Debug/vk_viewer --inputFile _downloaded_resources/flowers_1/flowers_1.ply --pipeline 2 --pbrEnabled 0 --screenshot /tmp/legacy.png --screenshotDelay 3.0
      2. Compare against pre-PBR baseline (from Task 4)
    Expected Result: Legacy path unchanged
    Failure Indicators: Visual regression
    Evidence: .sisyphus/evidence/task-16-legacy-baseline.png
  ```

  **Commit**: YES (C15)
  - Message: `feat(shader): wire PBR shading to G-buffer + IBL`
  - Files: `shaders/threedgrt_raytrace.rgen.slang`, `shaders/shaderio.h`
  - Pre-commit: `cmake --build build --config Debug`

---

- [ ] 17. Implement MIS shadow ray pass for irradiance

  **What to do**:
  - Create new file `shaders/irradiance_shadow.h.slang`:
    - Implement `float3 traceIrradianceMIS(float3 shadingPoint, float3 normal, TextureCube envMap, Sampler envSampler, StructuredBuffer<EnvAccel> envAccel, ...)`
    - For N samples (N=2 for interactive, N=64 for high quality):
      - Sample direction: combine cosine-weighted (BRDF) and env importance (EnvAccel)
      - Compute MIS weight: `w = pdf1^2 / (pdf1^2 + pdf2^2)` (balance heuristic)
      - Trace shadow ray from shadingPoint in sampled direction
      - If unoccluded: accumulate `envMap.SampleLevel(direction, 0).rgb * w / pdf`
    - Use top-K splats from K-buffer as potential occluders
    - Return accumulated irradiance
  - Document MIS formula in comments
  - Reference Veach 1998 (MIS paper)

  **Must NOT do**:
  - Do NOT trace against full splat set (use top-K)
  - Do NOT use uniform hemisphere sampling (variance too high)

  **Recommended Agent Profile**:
  - **Category**: `ultrabrain`
  - **Skills**: `[]`
  - **Reason**: Complex statistical methods, requires deep understanding of MIS

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 14, 15)
  - **Blocked By**: Task 3 (needs API audit)

  **References**:
  - TRON paper Section 3.3 — irradiance via any-hit shadow rays
  - TRON paper Algorithm 2 — MIS weighting
  - TRON paper Appendix Sec. 4 — out-of-order secondary rays
  - Veach 1998 — "Robust Monte Carlo Methods for Light Transport Simulation" (MIS paper)
  - `nvpro_core2/nvshaders/hdr_env_sampling.h.slang` — `environmentSample()` function
  - `shaders/threedgrt_payload.h.slang` — K-buffer access

  **Acceptance Criteria**:
  - [ ] `shaders/irradiance_shadow.h.slang` exists
  - [ ] MIS sampling (cosine + env importance) implemented
  - [ ] Balance heuristic weighting correct
  - [ ] Shadow ray traced against top-K K-buffer splats
  - [ ] Configurable sample count (N=2 interactive, N=64 quality)
  - [ ] CPU unit tests from Task 7 validate MIS math
  - [ ] `cmake --build build --config Debug` succeeds

  **QA Scenarios**:
  ```
  Scenario: MIS shader compiles
    Tool: Bash
    Steps:
      1. cmake --build build --config Debug
      2. ./_bin/Debug/vk_viewer --inputFile _downloaded_resources/flowers_1/flowers_1.ply --pipeline 2 --validation 1
    Expected Result: No shader errors, no validation errors
    Failure Indicators: Slang errors, NaN values
    Evidence: .sisyphus/evidence/task-17-mis-compile.log

  Scenario: MIS weights match reference
    Tool: Bash
    Steps:
      1. Compare shader MIS weights against CPU reference (from Task 7)
      2. Tolerance: 1e-4
    Expected Result: MIS weights within tolerance
    Failure Indicators: Large differences
    Evidence: .sisyphus/evidence/task-17-mis-comparison.log
  ```

  **Commit**: YES (C16)
  - Message: `feat(shader): add MIS shadow ray pass for irradiance`
  - Files: `shaders/irradiance_shadow.h.slang`
  - Pre-commit: `cmake --build build --config Debug && ctest`

---

- [ ] 18. Wire shadow rays to top-K K-buffer splats

  **What to do**:
  - Edit `shaders/threedgrt_raytrace.rgen.slang`:
    - After PBR shading (Task 16), if `frameInfo.irradianceEnabled != 0`:
      - Call `traceIrradianceMIS()` from irradiance_shadow.h.slang
      - Pass top-K splat IDs from K-buffer as potential occluders
      - Write result to `gbufferIrradiance` (new image) or as separate buffer
  - Edit `shaders/shaderio.h`:
    - Add `int32_t irradianceEnabled` to `FrameInfo` struct
    - Add `#define RTX_BINDING_GBUFFER_IRRADIANCE 16` (RGBA16F, irradiance RGB + unused)

  **Must NOT do**:
  - Do NOT trace against full splat set (intractable)
  - Do NOT use uniform sampling (high variance)

  **Recommended Agent Profile**:
  - **Category**: `ultrabrain`
  - **Skills**: `[]`
  - **Reason**: Complex integration, requires understanding K-buffer + shadow rays

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Task 17)
  - **Blocked By**: Task 17

  **References**:
  - `shaders/threedgrt_raytrace.rgen.slang` — existing raygen (where to add shadow rays)
  - `shaders/irradiance_shadow.h.slang` — new MIS function (from Task 17)
  - `shaders/threedgrt_payload.h.slang` — K-buffer access
  - TRON paper Appendix Sec. 4 — out-of-order secondary rays

  **Acceptance Criteria**:
  - [ ] Shadow rays traced from PBR shading point
  - [ ] Top-K splats from K-buffer used as occluders
  - [ ] Irradiance output written to dedicated buffer
  - [ ] `cmake --build build --config Debug` succeeds
  - [ ] Vulkan validation clean

  **QA Scenarios**:
  ```
  Scenario: Shadow rays produce shadows
    Tool: Bash
    Steps:
      1. ./_bin/Debug/vk_viewer --inputFile _downloaded_resources/flowers_1/flowers_1.ply --pipeline 2 --pbrEnabled 1 --irradianceEnabled 1 --screenshot /tmp/shadow.png --screenshotDelay 3.0
      2. Visually verify shadows are present (compare against no-shadow baseline)
    Expected Result: Shadows visible in output
    Failure Indicators: No shadows, all-bright output
    Evidence: .sisyphus/evidence/task-18-shadows.png
  ```

  **Commit**: YES (C17)
  - Message: `feat(shader): wire shadow rays to top-K K-buffer splats`
  - Files: `shaders/threedgrt_raytrace.rgen.slang`, `shaders/shaderio.h`
  - Pre-commit: `cmake --build build --config Debug`

---

- [ ] 19. Implement AgX tone mapping

  **What to do**:
  - Create new file `shaders/tonemap.h.slang`:
    - Implement `float3 agxToneMap(float3 color, float exposure)`
    - Use Three.js AgX implementation (open source, no patents)
    - Steps:
      1. Apply exposure: `color *= exposure`
      2. Convert to AgX log space
      3. Apply AgX curve (sigmoid-like)
      4. Convert back to linear sRGB
    - Reference: https://iolite-engine.com/blog_posts/minimal_agx_implementation
  - Document expected behavior in comments

  **Must NOT do**:
  - Do NOT use proprietary AgX (use Three.js open implementation)
  - Do NOT skip exposure step

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `[]`
  - **Reason**: Well-known tone mapping, but careful implementation needed

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 14, 15, 17)
  - **Blocked By**: Task 3 (needs API audit)

  **References**:
  - TRON paper Section 6 — AgX tone mapping
  - Three.js AgX implementation: https://github.com/mrdoob/three.js/blob/dev/src/renderers/shaders/ShaderChunk/tonemapping_pars_fragment.glsl.js
  - nvpro_core2/nvshaders/tonemapper.slang — reference tonemapper
  - https://iolite-engine.com/blog_posts/minimal_agx_implementation — minimal AgX

  **Acceptance Criteria**:
  - [ ] `shaders/tonemap.h.slang` exists
  - [ ] AgX tone mapping implemented (Three.js version)
  - [ ] Exposure applied
  - [ ] `cmake --build build --config Debug` succeeds
  - [ ] CPU unit tests from Task 7 validate tone mapping curve

  **QA Scenarios**:
  ```
  Scenario: AgX tone mapping produces correct output
    Tool: Bash
    Steps:
      1. ./_bin/Debug/vk_viewer --inputFile _downloaded_resources/flowers_1/flowers_1.ply --pipeline 2 --pbrEnabled 1 --screenshot /tmp/tonemap.png --screenshotDelay 3.0
      2. Compare output against reference (e.g., Blender AgX render)
    Expected Result: Output matches AgX reference
    Failure Indicators: Oversaturated, undersaturated, wrong colors
    Evidence: .sisyphus/evidence/task-19-tonemap.png
  ```

  **Commit**: YES (C18)
  - Message: `feat(shader): add AgX tone mapping`
  - Files: `shaders/tonemap.h.slang`
  - Pre-commit: `cmake --build build --config Debug && ctest`

---

### Wave 5: Composition + UI + CLI + Tests (depends on Wave 4)

- [ ] 20. Compose PBR + irradiance + tone map in final output

  **What to do**:
  - Edit `shaders/threedgrt_raytrace.rgen.slang`:
    - In final output write (line ~483), if PBR enabled:
      - Compute final color: `finalColor = pbrColor + irradiance` (or as configured)
      - Apply AgX tone mapping: `finalColor = agxToneMap(finalColor, envMapExposure)`
    - If PBR disabled: use existing linear-to-sRGB conversion
  - Edit `shaders/shaderio.h`:
    - Add `int32_t toneMapEnabled` to `FrameInfo` struct (0=legacy, 1=AgX)

  **Must NOT do**:
  - Do NOT change output for PBR-disabled path
  - Do NOT skip tone mapping for HDR output

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `[]`
  - **Reason**: Final composition, straightforward integration

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Tasks 16, 18, 19)
  - **Blocked By**: Tasks 16, 18, 19

  **References**:
  - `shaders/threedgrt_raytrace.rgen.slang:483` — existing output write
  - `shaders/pbr_shading.h.slang` — PBR output (from Task 15)
  - `shaders/irradiance_shadow.h.slang` — irradiance output (from Task 17)
  - `shaders/tonemap.h.slang` — AgX function (from Task 19)

  **Acceptance Criteria**:
  - [ ] PBR + irradiance + AgX composed correctly
  - [ ] Legacy path unchanged
  - [ ] `cmake --build build --config Debug` succeeds
  - [ ] Vulkan validation clean

  **QA Scenarios**:
  ```
  Scenario: Final output matches expected
    Tool: Bash
    Steps:
      1. ./_bin/Debug/vk_viewer --inputFile _downloaded_resources/flowers_1/flowers_1.ply --pipeline 2 --pbrEnabled 1 --irradianceEnabled 1 --toneMapEnabled 1 --screenshot /tmp/final.png --screenshotDelay 3.0
      2. file /tmp/final.png → expect "PNG image data, 800 x 600"
      3. Mean brightness > 10
    Expected Result: Final output is visible, non-black
    Failure Indicators: Black output, crashes
    Evidence: .sisyphus/evidence/task-20-final-output.png
  ```

  **Commit**: YES (C19)
  - Message: `feat(shader): compose PBR + irradiance + tone map in final output`
  - Files: `shaders/threedgrt_raytrace.rgen.slang`, `shaders/shaderio.h`
  - Pre-commit: `cmake --build build --config Debug`

---

- [ ] 21. Add --envmap, --envmapRotation, --envmapExposure CLI args

  **What to do**:
  - Edit `src/parameters.cpp`:
    - Add parameter registration for `--envmap <path>` (string, default empty)
    - Add parameter registration for `--envmapRotation <float>` (float, default 0.0)
    - Add parameter registration for `--envmapExposure <float>` (float, default 1.0)
    - Add parameter registration for `--pbrEnabled <0/1>` (int, default 0)
    - Add parameter registration for `--irradianceEnabled <0/1>` (int, default 0)
    - Add parameter registration for `--toneMapEnabled <0/1>` (int, default 0)
  - Edit `src/parameters.h`:
    - Add `PbrParameters prmPbr;` global
  - Edit `src/main.cpp`:
    - Wire CLI args to `prmPbr`
  - Edit `src/vk_viewer.cpp`:
    - On startup, if `prmPbr.envMap` is not empty, call `loadEnvironment(prmPbr.envMap)`
    - Set `prmFrame.envMapRotation`, `prmFrame.envMapExposure`, `prmFrame.pbrEnabled`, etc.

  **Must NOT do**:
  - Do NOT load envmap if path is empty
  - Do NOT error on missing envmap file (log warning, continue with default)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`
  - **Reason**: Simple parameter registration, follows existing pattern

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 22, 23, 25)
  - **Blocked By**: Task 20

  **References**:
  - `src/parameters.cpp` — existing parameter registration
  - `src/parameters.h` — existing parameter structs
  - `src/main.cpp` — existing CLI arg wiring
  - `src/vk_viewer.cpp` — `loadEnvironment()` method (from Task 10)

  **Acceptance Criteria**:
  - [ ] 6 new CLI args registered
  - [ ] `prmPbr` struct added
  - [ ] CLI args wired to `prmFrame` and `loadEnvironment()`
  - [ ] `--envmap ""` (empty) skips loading
  - [ ] Missing envmap file logs warning, continues
  - [ ] `cmake --build build --config Debug` succeeds

  **QA Scenarios**:
  ```
  Scenario: --envmap arg loads envmap
    Tool: Bash
    Steps:
      1. ./_bin/Debug/vk_viewer --inputFile _downloaded_resources/flowers_1/flowers_1.ply --pipeline 2 --envmap _downloaded_resources/studio_small_07_4k.hdr --screenshot /tmp/envmap.png --screenshotDelay 3.0
      2. file /tmp/envmap.png → expect "PNG image data, 800 x 600"
    Expected Result: Envmap loaded, PBR uses it
    Failure Indicators: "envmap not found" error, crash
    Evidence: .sisyphus/evidence/task-21-envmap-arg.png

  Scenario: --envmapRotation changes output
    Tool: Bash
    Steps:
      1. ./_bin/Debug/vk_viewer ... --envmapRotation 0.0 --screenshot /tmp/rot0.png
      2. ./_bin/Debug/vk_viewer ... --envmapRotation 1.57 --screenshot /tmp/rot90.png
      3. Compare: expect different output
    Expected Result: Rotation produces different output
    Failure Indicators: Identical output (rotation not applied)
    Evidence: .sisyphus/evidence/task-21-rotation-comparison.png
  ```

  **Commit**: YES (C20)
  - Message: `feat(cli): add --envmap, --envmapRotation, --envmapExposure args`
  - Files: `src/parameters.cpp`, `src/parameters.h`, `src/main.cpp`
  - Pre-commit: `cmake --build build --config Debug`

---

- [ ] 22. Add ImGui envmap control panel

  **What to do**:
  - Edit `src/vk_viewer_ui.cpp`:
    - In `guiDrawRendererProperties()` (or new function `guiDrawPbrProperties()`):
      - Add new tab "PBR Settings" inside `BeginTabBar("##SpecificsBar")`
      - Add controls:
        - `PE::entry("Envmap")` with button "Load..." that opens file dialog
        - `PE::SliderFloat("Rotation", &prmPbr.envMapRotation, 0.0f, 6.28f)`
        - `PE::SliderFloat("Exposure", &prmPbr.envMapExposure, 0.1f, 10.0f)`
        - `PE::Checkbox("PBR Enabled", &prmPbr.pbrEnabled)`
        - `PE::Checkbox("Irradiance (Shadows)", &prmPbr.irradianceEnabled)`
        - `PE::Checkbox("AgX Tone Mapping", &prmPbr.toneMapEnabled)`
        - Display current envmap filename
    - Wire changes to `prmFrame` on each frame

  **Must NOT do**:
  - Do NOT add new window (use existing Properties window)
  - Do NOT change existing UI panels

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
  - **Skills**: `["frontend-ui-ux"]`
  - **Reason**: ImGui UI work, requires understanding existing panel structure

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Task 21)
  - **Blocked By**: Task 21

  **References**:
  - `src/vk_viewer_ui.cpp:23-100` — existing `guiDrawRendererProperties()` pattern
  - `src/vk_viewer_ui.cpp:401-560` — file dialog pattern
  - `nvgui::PropertyEditor` — PE helper

  **Acceptance Criteria**:
  - [ ] "PBR Settings" tab added to Properties window
  - [ ] All 6 controls present and functional
  - [ ] File dialog opens for envmap selection
  - [ ] Changes wire to `prmFrame` on each frame
  - [ ] `cmake --build build --config Debug` succeeds

  **QA Scenarios**:
  ```
  Scenario: PBR Settings tab visible
    Tool: Bash (with X11 forwarding or screenshot)
    Steps:
      1. ./_bin/Debug/vk_viewer --inputFile _downloaded_resources/flowers_1/flowers_1.ply --pipeline 2
      2. Screenshot UI (manual or scripted)
      3. Verify "PBR Settings" tab exists in Properties window
    Expected Result: Tab visible, all controls present
    Failure Indicators: Tab missing, controls broken
    Evidence: .sisyphus/evidence/task-22-pbr-tab.png

  Scenario: UI controls change output
    Tool: Bash
    Steps:
      1. Open viewer, load envmap via UI
      2. Change rotation slider
      3. Verify output changes (via screenshot)
    Expected Result: UI controls functional
    Failure Indicators: Controls don't affect output
    Evidence: .sisyphus/evidence/task-22-ui-controls.png
  ```

  **Commit**: YES (C21)
  - Message: `feat(ui): add ImGui envmap control panel`
  - Files: `src/vk_viewer_ui.cpp`, `src/vk_viewer_ui.h`
  - Pre-commit: `cmake --build build --config Debug`

---

- [ ] 23. Add functional screenshot test (run_pbr_screenshot_test.sh)

  **What to do**:
  - Create `tests/run_pbr_screenshot_test.sh`:
    - Follow pattern from `tests/run_stochastic_screenshot_test.sh`
    - Build if needed
    - Run viewer with `--pipeline 2 --pbrEnabled 1 --envmap _downloaded_resources/studio_small_07_4k.hdr --screenshot /tmp/pbr_test.png --size 800 600 --screenshotDelay 3.0 --validation 0`
    - Verify: file exists → valid PNG → 800x600 → non-black (mean > 10)
    - Optionally: compare against legacy baseline (visual diff < threshold)
  - Add to `tests/CMakeLists.txt`:
    - `add_test(NAME pbr_screenshot_test COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_pbr_screenshot_test.sh $<CONFIG>)`
    - Mark as `DISABLED` by default (no GPU in CI)

  **Must NOT do**:
  - Do NOT require GPU in CI (mark as DISABLED)
  - Do NOT use hardcoded paths

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `[]`
  - **Reason**: Bash script + CMake integration, follows existing pattern

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 25, 26)
  - **Blocked By**: Tasks 21, 22

  **References**:
  - `tests/run_stochastic_screenshot_test.sh` — canonical pattern
  - `tests/CMakeLists.txt` — existing test registration
  - AGENTS.md — screenshot test instructions

  **Acceptance Criteria**:
  - [ ] `tests/run_pbr_screenshot_test.sh` exists
  - [ ] Test target added to CMakeLists.txt (DISABLED for CI)
  - [ ] Test passes locally: PNG exists, 800x600, non-black
  - [ ] Test cleans up temporary files

  **QA Scenarios**:
  ```
  Scenario: Screenshot test passes locally
    Tool: Bash
    Steps:
      1. bash tests/run_pbr_screenshot_test.sh Release
      2. Check exit code → expect 0
      3. Check /tmp/pbr_test.png exists
      4. file /tmp/pbr_test.png → expect "PNG image data, 800 x 600"
    Expected Result: Test passes
    Failure Indicators: Test fails, missing output
    Evidence: .sisyphus/evidence/task-23-screenshot-test.log

  Scenario: Test registered with CTest
    Tool: Bash
    Steps:
      1. ctest -N | grep pbr_screenshot_test → expect entry (DISABLED)
    Expected Result: Test registered
    Failure Indicators: Test not registered
    Evidence: .sisyphus/evidence/task-23-ctest-list.log
  ```

  **Commit**: YES (C22)
  - Message: `test(screenshot): add PBR functional screenshot test`
  - Files: `tests/run_pbr_screenshot_test.sh`, `tests/CMakeLists.txt`
  - Pre-commit: `bash tests/run_pbr_screenshot_test.sh Release`

---

- [ ] 24. Add new shader files to CMakeLists

  **What to do**:
  - Edit `CMakeLists.txt`:
    - Verify shader files are auto-discovered (check for `file(GLOB ...)` or explicit list)
    - If explicit list, add: `pbr_shading.h.slang`, `irradiance_shadow.h.slang`, `tonemap.h.slang`
    - If auto-discovered, no action needed
  - Verify all 3 new shader files are compiled by Slang

  **Must NOT do**:
  - Do NOT break existing shader compilation
  - Do NOT change shader compilation flags

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`
  - **Reason**: Simple CMake update, likely no action needed

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 25, 26)
  - **Blocked By**: Tasks 15, 17, 19

  **References**:
  - `CMakeLists.txt` — shader file discovery pattern
  - `shaders/` directory — shader file location

  **Acceptance Criteria**:
  - [ ] All 3 new shader files compiled
  - [ ] `cmake --build build --config Debug` succeeds
  - [ ] `cmake --build build --config Release` succeeds

  **QA Scenarios**:
  ```
  Scenario: New shaders compile
    Tool: Bash
    Steps:
      1. cmake --build build --config Debug 2>&1 | grep -E "pbr_shading|irradiance_shadow|tonemap"
      2. Expect: shader compilation entries in build log
    Expected Result: All 3 shaders compiled
    Failure Indicators: Missing shader compilation
    Evidence: .sisyphus/evidence/task-24-shader-compile.log
  ```

  **Commit**: YES (C23)
  - Message: `build: add new shader files to CMakeLists`
  - Files: `CMakeLists.txt`
  - Pre-commit: `cmake --build build --config Debug && ctest`

---

- [ ] 25. Write doc/pbr_ray_tracing.md

  **What to do**:
  - Create `doc/pbr_ray_tracing.md`:
    - Overview: PBR ray tracing pipeline (TRON-style)
    - Architecture: G-buffer compositing → Cook-Torrance PBR → irradiance → AgX tone mapping
    - CLI args: `--envmap`, `--envmapRotation`, `--envmapExposure`, `--pbrEnabled`, `--irradianceEnabled`, `--toneMapEnabled`
    - UI controls: PBR Settings tab
    - PLY format: basecolor_0/1/2, roughness, metallic properties
    - Default values: basecolor = SH DC, roughness = 0.5, metallic = 0.0
    - Limitations: PLY-only, 3DGRT-only, no temporal accumulation
    - References: TRON paper, Karis 2013, Veach 1998

  **Must NOT do**:
  - Do NOT include code snippets (link to source instead)
  - Do NOT repeat AGENTS.md content

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Skills**: `[]`
  - **Reason: Technical writing, follows existing doc pattern

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 26)
  - **Blocked By**: Tasks 21, 22

  **References**:
  - `doc/ray_tracing_3d_gaussians.md` — existing pipeline doc (template)
  - `doc/overview_of_vk_gaussian_splatting.md` — overview doc
  - TRON paper — for citations

  **Acceptance Criteria**:
  - [ ] `doc/pbr_ray_tracing.md` exists
  - [ ] All sections present (overview, architecture, CLI, UI, PLY, limitations, references)
  - [ ] Markdown formatting correct
  - [ ] Links to source files (not code dumps)

  **QA Scenarios**:
  ```
  Scenario: Doc file exists and is well-formed
    Tool: Bash
    Steps:
      1. ls -la doc/pbr_ray_tracing.md → expect exists
      2. wc -l doc/pbr_ray_tracing.md → expect > 50 lines
      3. grep "^## " doc/pbr_ray_tracing.md → expect multiple sections
    Expected Result: Doc exists with all sections
    Failure Indicators: Missing file, empty file
    Evidence: .sisyphus/evidence/task-25-doc-exists.log
  ```

  **Commit**: YES (C24)
  - Message: `docs: add doc/pbr_ray_tracing.md`
  - Files: `doc/pbr_ray_tracing.md`
  - Pre-commit: (docs only, no build)

---

- [ ] 26. Update AGENTS.md with new PBR options

  **What to do**:
  - Edit `AGENTS.md`:
    - Add section "TRON PBR Pipeline" with:
      - Brief overview
      - CLI args reference
      - Default envmap location
      - PLY material properties
      - Limitations (3DGRT only, etc.)
    - Add "Build & Test" section for PBR:
      - `cmake --build build --target unit_tests --config Debug` → run CPU unit tests
      - `bash tests/run_pbr_screenshot_test.sh Release` → run screenshot test
    - Add "Default envmap" note in Resources section

  **Must NOT do**:
  - Do NOT remove existing AGENTS.md content
  - Do NOT repeat doc/pbr_ray_tracing.md

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Skills**: `[]`
  - **Reason: Technical writing, follows existing AGENTS.md structure

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Tasks 23, 25)
  - **Blocked By**: Tasks 23, 25

  **References**:
  - `AGENTS.md` — existing structure
  - `doc/pbr_ray_tracing.md` — detailed doc (from Task 25)

  **Acceptance Criteria**:
  - [ ] `AGENTS.md` updated with PBR section
  - [ ] CLI args documented
  - [ ] Build & test instructions added
  - [ ] Default envmap location noted
  - [ ] Existing content preserved

  **QA Scenarios**:
  ```
  Scenario: AGENTS.md updated correctly
    Tool: Bash
    Steps:
      1. grep "TRON PBR" AGENTS.md → expect match
      2. grep -- "--envmap" AGENTS.md → expect match
      3. grep "studio_small_07_4k" AGENTS.md → expect match
    Expected Result: All PBR references present
    Failure Indicators: Missing references
    Evidence: .sisyphus/evidence/task-26-agents-updated.log
  ```

  **Commit**: YES (C25)
  - Message: `docs: update AGENTS.md with new PBR options`
  - Files: `AGENTS.md`
  - Pre-commit: (docs only, no build)

---

## Final Verification Wave (MANDATORY)

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, build, run command). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in `.sisyphus/evidence/`.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run `cmake --build build --config Debug` + `cmake --build build --config Release` + `ctest --output-on-failure`. Review all changed files for: `as any`/`@ts-ignore` (N/A in C++), empty catches, `std::cout`/`std::cerr` (must use LOGI/LOGW/LOGE), commented-out code, unused includes, AI slop (excessive comments, over-abstraction, generic names).
  Output: `Build Debug [PASS/FAIL] | Build Release [PASS/FAIL] | Tests [N pass/N fail] | Files [N clean/N issues] | VERDICT`

- [ ] F3. **Real Manual QA** — `unspecified-high`
  Run the functional screenshot test script. Verify PNG is 800x600, non-black, contains PBR-shaded elements. Test envmap rotation produces different output. Test legacy fallback works. Test CLI args override ImGui. Save evidence to `.sisyphus/evidence/final-qa/`.
  Output: `Scenarios [N/N pass] | Integration [N/N] | Edge Cases [N tested] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff (git log/diff). Verify 1:1 — everything in spec was built (no missing), nothing beyond spec was built (no creep). Check "Must NOT do" compliance. Detect cross-task contamination. Flag unaccounted changes.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

Atomic commits — each commit is self-contained, buildable, and passes tests.

```
C1: chore(deps): update nvpro_core2 to latest upstream main
    Files: nvpro_core2 (submodule)
    Pre-commit: cmake --build build --config Debug

C2: fix(vk_viewer): adapt to nvpro_core2 API changes
    Files: src/*, shaders/* (any files broken by update)
    Pre-commit: cmake --build build --config Debug && ctest

C3: feat(splat): add material vectors to SplatSet (CPU-only)
    Files: src/splat_set.h
    Pre-commit: cmake --build build --config Debug

C4: feat(ply): parse basecolor/roughness/metallic from PLY
    Files: src/splat_loader_fast.cpp, src/splat_loader_fast.h
    Pre-commit: cmake --build build --config Debug

C5: test(ply): add CPU unit test for PLY material parsing
    Files: tests/test_ply_material.cpp, tests/CMakeLists.txt
    Pre-commit: ctest --output-on-failure

C6: feat(splat): upload material buffers to GPU
    Files: src/splat_set_vk.h, src/splat_set_vk.cpp
    Pre-commit: cmake --build build --config Debug

C7: feat(shader): add material fetch functions for particles
    Files: shaders/threedgs_particles_storage.h.slang
    Pre-commit: cmake --build build --config Debug

C8: test(pbr): add CPU unit tests for BRDF + MIS math
    Files: tests/test_pbr_helpers.cpp, tests/CMakeLists.txt
    Pre-commit: ctest --output-on-failure

C9: feat(ibl): instantiate HdrIbl + HdrEnvDome in vk_viewer
    Files: src/vk_viewer.cpp, src/vk_viewer.h
    Pre-commit: cmake --build build --config Debug

C10: chore(resources): download + verify default HDR envmap
     Files: _downloaded_resources/studio_small_07_4k.hdr
     Pre-commit: file _downloaded_resources/studio_small_07_4k.hdr

C11: feat(descriptors): add IBL bindings (envmap, prefiltered, BRDF LUT)
     Files: src/vk_viewer_pipelines.cpp, shaders/shaderio.h
     Pre-commit: cmake --build build --config Debug

C12: feat(descriptors): add G-buffer output bindings to 3DGRT
     Files: src/vk_viewer_rtx.cpp, shaders/shaderio.h
     Pre-commit: cmake --build build --config Debug

C13: feat(shader): add particleProcessHitGbuffer with front-to-back Over
     Files: shaders/threedgrt.h.slang
     Pre-commit: cmake --build build --config Debug

C14: feat(shader): add Cook-Torrance PBR shader
     Files: shaders/pbr_shading.h.slang
     Pre-commit: cmake --build build --config Debug

C15: feat(shader): wire PBR shading to G-buffer + IBL
     Files: shaders/threedgrt_raytrace.rgen.slang
     Pre-commit: cmake --build build --config Debug

C16: feat(shader): add MIS shadow ray pass for irradiance
     Files: shaders/irradiance_shadow.h.slang
     Pre-commit: cmake --build build --config Debug

C17: feat(shader): wire shadow rays to top-K K-buffer splats
     Files: shaders/threedgrt_raytrace.rgen.slang
     Pre-commit: cmake --build build --config Debug

C18: feat(shader): add AgX tone mapping
     Files: shaders/tonemap.h.slang
     Pre-commit: cmake --build build --config Debug

C19: feat(shader): compose PBR + irradiance + tone map in final output
     Files: shaders/threedgrt_raytrace.rgen.slang
     Pre-commit: cmake --build build --config Debug

C20: feat(cli): add --envmap, --envmapRotation, --envmapExposure args
     Files: src/parameters.cpp, src/parameters.h, src/main.cpp
     Pre-commit: cmake --build build --config Debug

C21: feat(ui): add ImGui envmap control panel
     Files: src/vk_viewer_ui.cpp, src/vk_viewer_ui.h
     Pre-commit: cmake --build build --config Debug

C22: test(screenshot): add PBR functional screenshot test
     Files: tests/run_pbr_screenshot_test.sh, tests/CMakeLists.txt
     Pre-commit: bash tests/run_pbr_screenshot_test.sh Release

C23: build: add new shader files to CMakeLists
     Files: CMakeLists.txt
     Pre-commit: cmake --build build --config Debug && ctest

C24: docs: add doc/pbr_ray_tracing.md
     Files: doc/pbr_ray_tracing.md
     Pre-commit: (docs only, no build)

C25: docs: update AGENTS.md with new PBR options
     Files: AGENTS.md
     Pre-commit: (docs only, no build)
```

---

## Success Criteria

### Verification Commands
```bash
# Build verification
cmake --build build --config Debug     # Expected: success, no warnings
cmake --build build --config Release   # Expected: success, no warnings

# Unit tests
ctest --output-on-failure                # Expected: all pass

# Screenshot test
bash tests/run_pbr_screenshot_test.sh Release
# Expected: PASS, 800x600 PNG, non-black, mean brightness > 10

# Vulkan validation
./_bin/Debug/vk_viewer --inputFile _downloaded_resources/flowers_1/flowers_1.ply --pipeline 2 --validation 1
# Expected: no validation errors, clean shutdown

# Backward compat test
./_bin/Debug/vk_viewer --inputFile _downloaded_resources/flowers_1/flowers_1.ply --pipeline 2 --screenshot /tmp/legacy.png
# Expected: loads without error, uses default material
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent (no neural renderer, no DLSS-RR, no mesh PBR, etc.)
- [ ] All 26 tasks + 4 final verification tasks complete
- [ ] All unit tests pass
- [ ] All screenshot tests pass
- [ ] Build succeeds in Debug + Release
- [ ] Vulkan validation layers clean
- [ ] Backward compat: old PLY files load with defaults
- [ ] New PLY properties work when present
- [ ] Envmap rotation produces different output
- [ ] PBR/legacy toggle works
- [ ] CLI args override ImGui settings
- [ ] Default envmap shipped in repo
- [ ] Doc page written
- [ ] AGENTS.md updated
