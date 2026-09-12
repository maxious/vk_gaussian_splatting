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
 */

// This file is included from vk_viewer_ui.cpp - do not compile separately
// Contains: Renderer properties UI (guiDrawRendererProperties)

void VkViewerUI::guiDrawRendererProperties()
{

  namespace PE = nvgui::PropertyEditor;

  PE::begin("## Global settings ");
  bool vsync = m_app->isVsync();
  if(PE::Checkbox("V-Sync", &vsync))
    m_app->setVsync(vsync);

  if(PE::entry(
         "Pipeline", [&]() { return m_ui.enumCombobox(GUI_PIPELINE, "##ID", &prmSelectedPipeline); }, "Selects the rendering method"))
  {
    m_requestUpdateShaders = true;
  }

  if(PE::entry(
         "Default settings", [&] { return ImGui::Button("Reset"); }, "resets to default settings"))
  {
    resetRenderSettings();
    m_requestUpdateShaders   = true;
    m_requestUpdateSplatData = true;
  }

  bool allowVisualize = (prmSelectedPipeline == PIPELINE_RTX) || 
                        (m_depthManager != nullptr);

  ImGui::BeginDisabled(!allowVisualize);
  if(PE::entry(
         "Visualize", [&]() { return m_ui.enumCombobox(GUI_VISUALIZE, "##ID", &prmRender.visualize); }, "Selects the visualization mode"))
  {
    m_requestUpdateShaders = true;
    prmFrame.visualize = prmRender.visualize;
  }

  ImGui::BeginDisabled(prmRender.visualize == VISUALIZE_FINAL);
  if(PE::DragFloat("Multiplier", (float*)&prmFrame.multiplier, 1.0F, 0.0F, 1000.0F))
    resetFrameCounter();
  ImGui::EndDisabled();

  ImGui::EndDisabled();

  PE::end();

  PE::begin("## Common settings");
  if(PE::Checkbox("Wireframe", &prmRender.wireframe, "Show particle bounds in wireframe "))
    m_requestUpdateShaders = true;

  bool linearToSrgb = prmFrame.linearToSrgb != 0;
  if(PE::Checkbox("Linear to sRGB", &linearToSrgb,
                  "Apply linear-to-sRGB gamma correction for ML-SHARP PLY files.\n"
                  "Enable this when loading linearRGB Gaussians (e.g., from Apple ML-SHARP)\n"
                  "to prevent dark/incorrect colors."))
  {
    prmFrame.linearToSrgb = linearToSrgb ? 1 : 0;
    resetFrameCounter();
  }

  int alphaThres = int(255.0 * prmFrame.alphaCullThreshold);
  if(PE::SliderInt("Alpha culling threshold", &alphaThres, 0, 255, "%d", 0, "Discard splats with low opacity (with low contribution)."))
  {
    prmFrame.alphaCullThreshold = (float)alphaThres / 255.0f;
  }

  if(PE::SliderFloat("Alpha boost", &prmFrame.alphaBoost, 0.5f, 3.0f, "%.2f", 0,
                     "Increases splat opacity to reduce see-through artifacts. Values > 1 make splats more opaque."))
  {
    resetFrameCounter();
  }

  const int maxModelShDegree = m_splatSet.maxShDegree();
  prmRender.maxShDegree      = std::min(prmRender.maxShDegree, maxModelShDegree);

  if(PE::SliderInt("Maximum SH degree", (int*)&prmRender.maxShDegree, 0, maxModelShDegree, "%d", 0,
                   "Sets the highest degree of Spherical Harmonics (SH) used for view-dependent effects."))
    m_requestUpdateShaders = true;

  if(PE::Checkbox("Show SH deg > 0 only", &prmRender.showShOnly,
                  "Removes the base color from SH degree 0, applying only color deduced from \n"
                  "higher-degree SH to a neutral gray. This helps visualize their contribution."))
    m_requestUpdateShaders = true;

  if(PE::Checkbox("Disable opacity gaussian ", &prmRender.opacityGaussianDisabled,
                  "Disables the alpha component of the Gaussians, making their full range visible.\n"
                  "This helps analyze splat distribution and scales, especially when combined with Splat Scale adjustments."))
    m_requestUpdateShaders = true;

  PE::end();

  if(m_splatSet.has_time_data)
  {
    PE::begin("## 4D Controls");
    static bool animate = false;
    static float timeSpeed = 1.0f;
    PE::Checkbox("Animate", &animate);
    
    float tMin = m_splatSet.minTime;
    float tMax = m_splatSet.maxTime;
    if (tMin >= tMax) { tMin = 0.0f; tMax = 10.0f; }
    
    PE::SliderFloat("Time", &prmFrame.currentTime, tMin, tMax);
    PE::SliderFloat("Speed", &timeSpeed, 0.1f, 5.0f);
    
    // Toggle for temporal culling
    bool tempCull = (prmFrame.temporalCulling != 0);
    if(PE::Checkbox("Temporal Culling", &tempCull)) {
        prmFrame.temporalCulling = tempCull ? 1 : 0;
    }
    
    // Toggle for motion
    bool appMotion = (prmFrame.applyMotion != 0);
    if(PE::Checkbox("Apply 4D Motion", &appMotion)) {
        prmFrame.applyMotion = appMotion ? 1 : 0;
    }
    
    if(animate) {
       prmFrame.currentTime += ImGui::GetIO().DeltaTime * timeSpeed;
       if (prmFrame.currentTime > tMax) prmFrame.currentTime = tMin;
    }
    PE::end();
  }

  PE::begin("## SBS Stereo");
#ifdef WITH_OPENXR
  bool xrWasEnabled = m_useXrHmd;
  (void)xrWasEnabled; // Suppress unused warning
  if(PE::Checkbox("OpenXR HMD", &m_useXrHmd, "Enable OpenXR head-mounted display rendering"))
  {
    if(m_useXrHmd && !m_xrInitialized)
    {
      // Try to initialize OpenXR
      initializeOpenXR();
    }
    else if(!m_useXrHmd && m_xrInitialized)
    {
      // Shutdown OpenXR
      shutdownOpenXR();
      m_renderSBS = false;
    }
  }
  if(m_xrInitialized && m_xr)
  {
    VkExtent2D perEye = m_xr->getPerEyeExtent();
    PE::Text("XR Status", "Connected");
    PE::Text("Per-eye resolution", "%dx%d", perEye.width, perEye.height);
    PE::Text("Controllers", m_xr->hasControllers() ? "Active" : "Not detected");

    // Locomotion settings

    if(m_xr->hasControllers())
    {
      ImGui::Separator();
      PE::Text("## Locomotion", "");
      PE::SliderFloat("Move Speed", &m_xrMoveSpeed, 0.5f, 10.0f, "%.1f m/s", 0, "Movement speed in meters per second");
      PE::SliderFloat("Sprint Multiplier", &m_xrSprintMultiplier, 1.0f, 5.0f, "%.1fx", 0, "Speed multiplier when thumbstick is clicked");
      PE::Checkbox("Smooth Turn", &m_xrUseSmoothTurn, "Use smooth turning instead of snap turning");
      if(m_xrUseSmoothTurn)
      {
        PE::SliderFloat("Turn Speed", &m_xrSmoothTurnSpeed, 30.0f, 180.0f, "%.0f deg/s", 0, "Smooth turn speed in degrees per second");
      }
      else
      {
        PE::SliderFloat("Snap Angle", &m_xrSnapTurnAngle, 15.0f, 90.0f, "%.0f deg", 0, "Snap turn angle in degrees");
      }
    }

    if(m_xr->isColorSpaceSupported())
    {
      ImGui::Separator();
      PE::Text("## Color Space", "");
      PE::Text("Current", "%s", GsOpenXr::colorSpaceToString(m_xr->getCurrentColorSpace()));

      const auto& supportedSpaces = m_xr->getSupportedColorSpaces();
      if(!supportedSpaces.empty() && ImGui::BeginCombo("Color Space", GsOpenXr::colorSpaceToString(m_xr->getCurrentColorSpace())))
      {
        for(auto cs : supportedSpaces)
        {
          bool isSelected = (cs == m_xr->getCurrentColorSpace());
          if(ImGui::Selectable(GsOpenXr::colorSpaceToString(cs), isSelected))
          {
            m_xr->setColorSpace(cs);
          }
          if(isSelected)
          {
            ImGui::SetItemDefaultFocus();
          }
        }
        ImGui::EndCombo();
      }
    }
  }
  else if(m_useXrHmd)
  {
    PE::Text("XR Status", "Failed to connect");
  }
  else
  {
    PE::Text("XR Status", "Disabled");
  }
  ImGui::Separator();
  ImGui::BeginDisabled(m_xrInitialized);  // Disable SBS controls when XR is active
#else
  ImGui::BeginDisabled(false);
#endif
  PE::Checkbox("Enable SBS Stereo", &m_renderSBS, "Render side-by-side stereo for VR headsets");
  ImGui::EndDisabled();
  ImGui::BeginDisabled(!m_renderSBS);
  float ipdMM = m_stereoSeparation * 1000.0f;
  if(PE::SliderFloat("IPD (mm)", &ipdMM, 50.0f, 80.0f, "%.1f", 0,
                     "Inter-pupillary distance (eye separation). Default is 63mm."))
  {
    m_stereoSeparation = ipdMM / 1000.0f;
  }
  PE::SliderFloat("Convergence (m)", &m_stereoConvergence, 0.1f, 10.0f, "%.2f", 0,
                  "Distance where stereo images overlap perfectly (zero parallax). "
                  "Set closer for near objects, farther for distant scenes.");
  PE::Checkbox("Off-Axis Projection", &m_stereoOffAxisProj,
               "Use asymmetric frustum projection for proper stereo convergence. "
               "Reduces eye strain and improves depth perception at the convergence distance.");
  ImGui::EndDisabled();
  PE::end();

#ifdef WITH_DLSS_RR
  PE::begin("## DLSS Super Resolution");
  if(PE::Checkbox("Enable DLSS Neural Rendering", &m_dlssEnabled,
                  "Enable the Vulkan DLSS Super Resolution contract used by the optional DLSS5 bridge.\n"
                  "Runs in DLAA mode at the current viewport resolution."))
  {
    if(m_dlssEnabled)
    {
      // DLSS-RR and DLSS SR share the output image and cannot run together.
      if(m_dlssRREnabled)
      {
        m_dlssRREnabled = false;
        shutdownDlssRR();
      }
      initializeDlss();
      if(m_dlssInitialized)
      {
        updateDlssDescriptorSet();
        m_requestUpdateShaders = true;
      }
      else
      {
        m_dlssEnabled = false;
      }
    }
    else if(m_dlssInitialized)
    {
      shutdownDlss();
      m_requestUpdateShaders = true;
    }
    m_dlssRRNeedsReset = true;
  }
  ImGui::BeginDisabled(!m_dlssEnabled || !m_dlssInitialized);
  PE::Text("Mode", "DLAA (same-size input/output)");
  if(PE::entry("Reset History", [&]() { return ImGui::Button("Reset##DLSS"); },
               "Reset DLSS temporal history"))
  {
    m_dlssRRNeedsReset = true;
  }
  PE::Text("Status", m_dlssInitialized ? "Active" : "Disabled");
  ImGui::EndDisabled();
  PE::end();

  PE::begin("## DLSS-RR Denoising");
  bool wasEnabled = m_dlssRREnabled;
  (void)wasEnabled; // Suppress unused warning
  if(PE::Checkbox("Enable DLSS-RR", &m_dlssRREnabled, 
                  "Enable NVIDIA DLSS Ray Reconstruction denoiser for RTX mode.\n"
                  "Requires RTX GPU with DLSS support."))
  {
    if(m_dlssRREnabled && !m_dlssRRInitialized)
    {
      if(m_dlssEnabled)
      {
        m_dlssEnabled = false;
        shutdownDlss();
      }
      initializeDlssRR();
      if(m_dlssRRInitialized)
      {
        updateDlssRRDescriptorSet();
        m_requestUpdateShaders = true;
      }
      else
      {
        m_dlssRREnabled = false;
      }
    }
    else if(!m_dlssRREnabled && m_dlssRRInitialized)
    {
      shutdownDlssRR();
      m_requestUpdateShaders = true;
    }
    m_dlssRRNeedsReset = true;
  }
  ImGui::BeginDisabled(!m_dlssRREnabled || !m_dlssRRInitialized);
  static const char* qualityNames[] = {"Max Performance", "Balanced", "Max Quality", "Ultra Performance", "Ultra Quality", "DLAA"};
  int qualityIndex = 0;
  switch(m_dlssRRQuality)
  {
    case NVSDK_NGX_PerfQuality_Value_MaxPerf: qualityIndex = 0; break;
    case NVSDK_NGX_PerfQuality_Value_Balanced: qualityIndex = 1; break;
    case NVSDK_NGX_PerfQuality_Value_MaxQuality: qualityIndex = 2; break;
    case NVSDK_NGX_PerfQuality_Value_UltraPerformance: qualityIndex = 3; break;
    case NVSDK_NGX_PerfQuality_Value_UltraQuality: qualityIndex = 4; break;
    case NVSDK_NGX_PerfQuality_Value_DLAA: qualityIndex = 5; break;
    default: qualityIndex = 2; break;
  }
  if(PE::entry("Quality", [&]() { return ImGui::Combo("##DLSSQuality", &qualityIndex, qualityNames, 6); },
               "DLSS-RR quality preset"))
  {
    switch(qualityIndex)
    {
      case 0: m_dlssRRQuality = NVSDK_NGX_PerfQuality_Value_MaxPerf; break;
      case 1: m_dlssRRQuality = NVSDK_NGX_PerfQuality_Value_Balanced; break;
      case 2: m_dlssRRQuality = NVSDK_NGX_PerfQuality_Value_MaxQuality; break;
      case 3: m_dlssRRQuality = NVSDK_NGX_PerfQuality_Value_UltraPerformance; break;
      case 4: m_dlssRRQuality = NVSDK_NGX_PerfQuality_Value_UltraQuality; break;
      case 5: m_dlssRRQuality = NVSDK_NGX_PerfQuality_Value_DLAA; break;
    }
    // Reinitialize DLSS-RR with new quality
    if(m_dlssRRInitialized)
    {
      shutdownDlssRR();
      initializeDlssRR();
      if(m_dlssRRInitialized)
      {
        updateDlssRRDescriptorSet();
      }
    }
    m_dlssRRNeedsReset = true;
  }
  if(PE::entry("Reset History", [&]() { return ImGui::Button("Reset"); },
               "Reset DLSS-RR temporal history"))
  {
    m_dlssRRNeedsReset = true;
  }
  if(m_dlssRRInitialized)
  {
    PE::Text("Status", "Active");
  }
  else
  {
    PE::Text("Status", m_dlssRREnabled ? "Initialization failed" : "Disabled");
  }
  ImGui::EndDisabled();
  PE::end();
#endif

  ImGuiTabBarFlags tab_bar_flags = ImGuiTabBarFlags_None;
  if(ImGui::BeginTabBar("##SpecificsBar", tab_bar_flags))
  {
    if(prmSelectedPipeline != PIPELINE_RTX)
    {
      if(ImGui::BeginTabItem("Rasterization specifics"))
      {
        PE::begin("## Raster settings");

        if(PE::entry("Sorting method", [&]() { return m_ui.enumCombobox(GUI_SORTING, "##ID", &prmRaster.sortingMethod); }))
        {
          if(prmRaster.sortingMethod != SORTING_GPU_SYNC_RADIX && prmRaster.frustumCulling == FRUSTUM_CULLING_AT_DIST)
          {
            prmRaster.frustumCulling = FRUSTUM_CULLING_AT_RASTER;
            m_requestUpdateShaders   = true;
          }
          if(prmRaster.sortingMethod == SORTING_GPU_SYNC_RADIX && prmRaster.frustumCulling != FRUSTUM_CULLING_AT_DIST)
          {
            prmRaster.frustumCulling = FRUSTUM_CULLING_AT_DIST;
            m_requestUpdateShaders   = true;
          }
        }

        ImGui::BeginDisabled(prmRaster.sortingMethod == SORTING_GPU_SYNC_RADIX);
        PE::Checkbox("Lazy CPU sorting", &prmRaster.cpuLazySort, "Perform sorting only if viewpoint changes");

        PE::Text("CPU sorting state", m_cpuSorter.getStatus() == SplatSorterAsync::E_SORTING ? "Sorting" : "Idled");
        ImGui::EndDisabled();

        // GPU sort temporal stability options
        ImGui::BeginDisabled(prmRaster.sortingMethod != SORTING_GPU_SYNC_RADIX);
        PE::Checkbox("Skip GPU sort when stable", &prmRaster.gpuSortSkipWhenStable, 
                     "Skip GPU radix sorting when camera is stationary to improve performance");
        
        ImGui::BeginDisabled(!prmRaster.gpuSortSkipWhenStable);
        PE::Checkbox("Force sort every frame", &prmRaster.gpuSortForceEveryFrame, 
                     "Debug: Force GPU sorting every frame (disables stability optimization)");
        ImGui::EndDisabled();
        
        PE::Text("GPU sort status", m_sortSkippedThisFrame ? "Skipped" : "Active");
        ImGui::EndDisabled();

        // Radio buttons for exclusive selection
        PE::entry(
            "Frustum culling",
            [&]() {
              if(ImGui::RadioButton("Disabled", prmRaster.frustumCulling == FRUSTUM_CULLING_NONE))
              {
                prmRaster.frustumCulling = FRUSTUM_CULLING_NONE;
                m_requestUpdateShaders   = true;
              }

              ImGui::BeginDisabled(prmRaster.sortingMethod != SORTING_GPU_SYNC_RADIX);
              if(ImGui::RadioButton("At distance stage", prmRaster.frustumCulling == FRUSTUM_CULLING_AT_DIST))
              {
                prmRaster.frustumCulling = FRUSTUM_CULLING_AT_DIST;
                m_requestUpdateShaders   = true;
              }
              ImGui::EndDisabled();

              if(ImGui::RadioButton("At raster stage", prmRaster.frustumCulling == FRUSTUM_CULLING_AT_RASTER))
              {
                prmRaster.frustumCulling = FRUSTUM_CULLING_AT_RASTER;
                m_requestUpdateShaders   = true;
              }
              return true;
            },
            "Defines where frustum culling is performed: in the distance compute shader or \n"
            "at rasterization (in vertex or mesh shader). Culling can also be disabled for performance comparisons.");

        PE::SliderFloat("Frustum dilation", &prmFrame.frustumDilation, 0.0f, 1.0f, "%.1f", 0,
                        "Adjusts the frustum culling bounds to account for the fact that visibility is tested \n"
                        "only at the center of each splat, rather than its full elliptical shape. A positive \n"
                        "value expands the frustum by the given percentage, reducing the risk of prematurely \n"
                        "discarding splats near the frustum boundaries.");

        // Chunk-based hierarchical frustum culling
        ImGui::BeginDisabled(prmRaster.sortingMethod != SORTING_GPU_SYNC_RADIX);
        if(PE::Checkbox("Chunk culling", &prmRaster.chunkCullingEnabled,
                        "Enable chunk-based hierarchical frustum culling.\n"
                        "Splats are grouped into chunks of 256 and tested against the frustum at the chunk level\n"
                        "before individual splat processing. Can significantly improve performance when large\n"
                        "portions of the scene are outside the view frustum."))
        {
          // No shader rebuild needed - this is a runtime toggle
        }
        
        if(m_numChunks > 0)
        {
          char chunkInfo[64];
          snprintf(chunkInfo, sizeof(chunkInfo), "%u chunks (%u splats)", m_numChunks, prmFrame.splatCount);
          PE::Text("Chunks", chunkInfo);
        }
        ImGui::EndDisabled();

        if(PE::entry(
               "Dist WG size",
               [&]() { return m_ui.enumCombobox(GUI_DIST_SHADER_WG_SIZE, "##ID", &prmRaster.distShaderWorkgroupSize); },
               "Distance shader workgroup size"))
        {
          m_requestUpdateShaders = true;
        }

        if(PE::entry(
               "Mesh WG size",
               [&]() { return m_ui.enumCombobox(GUI_MESH_SHADER_WG_SIZE, "##ID", &prmRaster.meshShaderWorkgroupSize); },
               "Mesh shader workgroup size"))
        {
          m_requestUpdateShaders = true;
        }

        bool forceExtentProjection = prmSelectedPipeline == PIPELINE_VERT || prmSelectedPipeline == PIPELINE_MESH
                                     || prmSelectedPipeline == PIPELINE_HYBRID;

        ImGui::BeginDisabled(forceExtentProjection);
        if(PE::entry(
               "Projection Method",
               [&]() { return m_ui.enumCombobox(GUI_EXTENT_METHOD, "##ID", &prmRaster.extentProjection); },
               "Available for 3DGUT pipelines only, 3DGS allways uses Eigen.\n"
               "Method used to compute the 2D extent projection from the 3D covariance:\n"
               "- Eigen method leads to basis aligned rectangular extent, more performant\n"
               "- Conic method leads to axis aligned rectangular extent as in 3DGS and 3DGUT papers"))
        {
          m_requestUpdateShaders = true;
        }
        ImGui::EndDisabled();

        if(PE::Checkbox("Mip splatting antialiasing", &prmRaster.msAntialiasing,
                        "Indicates if Gaussians were trained (and should be rendered) with mip-splatting antialiasing method."))
          m_requestUpdateShaders = true;

        ImGui::BeginDisabled(prmSelectedPipeline == PIPELINE_MESH_3DGUT || prmSelectedPipeline == PIPELINE_HYBRID_3DGUT);

        if(PE::Checkbox("Fragment shader barycentric", &prmRaster.fragmentBarycentric,
                        "Enables fragment shader barycentric to reduce vertex and mesh shaders outputs."))
          m_requestUpdateShaders = true;

        // we set a different size range for point and splat rendering
        PE::SliderFloat("Splat scale", (float*)&prmFrame.splatScale, 0.1f, prmRaster.pointCloudModeEnabled != 0 ? 10.0f : 2.0f,
                        "%.3f", 0, "Adjusts the size of the splats for visualization purposes.");

        PE::SliderFloat("Min pixel radius", (float*)&prmFrame.minPixelRadius, 0.0f, 10.0f,
                        "%.1f", 0, "Minimum pixel radius for splats. Smaller splats are culled (removes noise).\n0 = no minimum culling.");

        PE::SliderFloat("Max pixel radius", (float*)&prmFrame.maxPixelRadius, 10.0f, 2000.0f,
                        "%.0f", 0, "Maximum pixel radius for splats. Larger splats are clamped to this size.");

        PE::SliderFloat("Sigma coverage", (float*)&prmFrame.sigmaCoverage, 2.0f, 4.0f,
                        "%.2f", 0, "Sigma coverage for Gaussian extent.\n2.0 = SuperSplat default\n2.83 = sqrt(8), original 3DGS\n3.0 = 3 sigma standard");

        if(PE::Checkbox("Disable splatting", &prmRaster.pointCloudModeEnabled,
                        "Switches to point cloud mode, displaying only the splat centers. \n"
                        "Other parameters such as Splat Scale still apply in this mode."))
          m_requestUpdateShaders = true;

        ImGui::EndDisabled();

        PE::end();

        ImGui::EndTabItem();
      }
    }

    if(prmSelectedPipeline == PIPELINE_RTX || prmSelectedPipeline == PIPELINE_HYBRID
       || prmSelectedPipeline == PIPELINE_HYBRID_3DGUT || prmSelectedPipeline == PIPELINE_MESH_3DGUT)
    {
      if(ImGui::BeginTabItem("Ray tracing and 3DGUT specifics"))
      {
        PE::begin("## Raytrace sampling and bounces");

        ImGui::BeginDisabled(prmSelectedPipeline == PIPELINE_MESH_3DGUT);
        PE::SliderInt("Max bounces", &prmFrame.rtxMaxBounces, 1, 16);
        ImGui::EndDisabled();

        ImGui::BeginDisabled(prmSelectedPipeline == PIPELINE_HYBRID);
        {
          if(PE::entry(
                 "Temporal sampling",
                 [&]() { return m_ui.enumCombobox(GUI_TEMPORAL_SAMPLING, "##ID", &prmRtx.temporalSamplingMode); },
                 "Enable accumulation of frame results over time.\n"
                 "Automatic will activate sampling depending on other effects such as DoF.\n"
                 "If enabled, the specified number of temporal samples will be accumulated over \"Temporal samples count\" frames,\n"
                 "and the last accumulated frame will be presented without additional rendering.\n"
                 "Note that rendering converges faster if v-sync is off.\n"
                 "If disabled, the system renders in free run mode."))
          {
            resetFrameCounter();
            m_requestUpdateShaders = true;
          }

          if(PE::InputInt("Temporal samples count", &prmFrame.frameSampleMax, 1, 100, 0,
                          "Number of frames after which temporal sampling is stopped. \n"
                          "A value of 0 disables temporal sampling."))
          {
            prmFrame.frameSampleMax = std::clamp(prmFrame.frameSampleMax, 1, 1000);
            resetFrameCounter();
          }
        }
        ImGui::EndDisabled();

        PE::end();

        PE::begin("## Raytrace gaussians settings");

        if(PE::entry("Kernel degree",
                     [&]() { return m_ui.enumCombobox(GUI_KERNEL_DEGREE, "##ID", &prmRtx.kernelDegree); }))
          m_requestUpdateSplatData = true;

        ImGui::BeginDisabled(prmSelectedPipeline == PIPELINE_MESH_3DGUT);

        int particleFormat = prmRtxData.useSpheres ? PARTICLE_FORMAT_SPHERE :
                             prmRtxData.useAABBs ? PARTICLE_FORMAT_PARAMETRIC : PARTICLE_FORMAT_ICOSAHEDRON;

        if(PE::entry(
               "Particles format", [&]() { return m_ui.enumCombobox(GUI_PARTICLE_FORMAT, "##ID", &particleFormat); },
               "Icosahedron: 20-triangle mesh per splat (slow)\n"
               "AABB + parametric: Axis-aligned bounding box with intersection shader (fast)\n"
               "Sphere (Blackwell): Native hardware ray-sphere intersection (RTX 50 series only, fastest)\n"))
        {
          if(particleFormat == PARTICLE_FORMAT_ICOSAHEDRON)
          {
            prmRtxData.useAABBs   = false;
            prmRtxData.useSpheres = false;
          }
          else if(particleFormat == PARTICLE_FORMAT_PARAMETRIC)
          {
            prmRtxData.useAABBs         = true;
            prmRtxData.useSpheres       = false;
            prmRtxData.useTlasInstances = true;
          }
          else if(particleFormat == PARTICLE_FORMAT_SPHERE)
          {
            prmRtxData.useAABBs         = false;  // Spheres replace AABBs
            prmRtxData.useSpheres       = true;
            prmRtxData.useTlasInstances = true;  // Sphere mode requires instances
          }
          m_requestUpdateSplatData = true;
          m_requestUpdateShaders   = true;  // Need to recompile shaders for RTX_USE_SPHERES
        }

        if(PE::Checkbox("Adaptive clamp", &prmRtx.kernelAdaptiveClamping))
          m_requestUpdateSplatData = true;

        // RTXMU-style scratch buffer pooling
        if(PE::entry(
               "Scratch pooling",
               [&]() { return ImGui::Checkbox("##ID", &prmRtxData.useScratchPooling); },
               "Reuse a persistent scratch buffer for AS builds (RTXMU optimization)\n"
               "Reduces allocation churn and memory fragmentation"))
        {
          // No immediate action needed - takes effect on next AS build
        }

        if(prmRtxData.useScratchPooling)
        {
          int poolSizeMB = static_cast<int>(prmRtxData.scratchPoolSizeMB);
          if(PE::InputInt("Pool size (MB)", &poolSizeMB))
          {
            prmRtxData.scratchPoolSizeMB = std::clamp(static_cast<uint32_t>(poolSizeMB), 1u, 256u);
          }
        }

        // PTLAS for FreeTimeGS sparse updates
        if(PE::entry(
               "PTLAS (Blackwell)",
               [&]() { return ImGui::Checkbox("##ID", &prmRtxData.usePtlas); },
               "Partitioned TLAS for sparse FreeTimeGS updates\n"
               "Only rebuilds partitions containing animated splats\n"
               "Requires RTX 50 series (Blackwell) or newer"))
        {
          m_requestUpdateSplatData = true;
        }

        if(prmRtxData.usePtlas)
        {
          if(PE::InputFloat("PTLAS cell size", &prmRtxData.ptlasCellSize, 0.1f, 10.0f, "%.2f"))
          {
            prmRtxData.ptlasCellSize = std::clamp(prmRtxData.ptlasCellSize, 0.1f, 100.0f);
            m_requestUpdateSplatData = true;
          }
        }

        PE::InputFloat("Alpha clamp", &prmFrame.alphaClamp, 0.0, 3.0, "%.2f", ImGuiInputTextFlags_EnterReturnsTrue);

        PE::InputFloat("Minimum transmittance", &prmFrame.minTransmittance, 0.0, 1.0, "%.2f", ImGuiInputTextFlags_EnterReturnsTrue);

        if(PE::entry(
               "Ray hits per pass",
               [&]() { return m_ui.enumCombobox(GUI_RAY_HIT_PER_PASS, "##ID", &prmRtx.payloadArraySize); },
               "Max number of ray hits stored per pass (i.e. payload array size)"))
        {
          m_requestUpdateShaders = true;
        }

        if(PE::InputInt("Maximum pass count", &prmFrame.maxPasses))
        {
          prmFrame.maxPasses = std::clamp(prmFrame.maxPasses, 1, 1000);
        }

        PE::Text("Maximum anyhit/pixel", std::to_string(prmRtx.payloadArraySize * prmFrame.maxPasses));

        ImGui::EndDisabled();

        PE::end();

        ImGui::EndTabItem();
      }
    }

    if(prmSelectedPipeline == PIPELINE_STOCHASTIC_GS)
    {
      if(ImGui::BeginTabItem("Stochastic GS settings"))
      {
        PE::begin("## Stochastic GS Settings");

        PE::Checkbox("Use GPS mode", (bool*)&prmStochastic.stochasticUseGps, "0=Stochastic Transparency, 1=GPS mode");

        PE::SliderInt("Samples per pixel", (int*)&prmStochastic.stochasticSamplesPerPixel, 1, 64);

        PE::SliderInt("Max samples", (int*)&prmStochastic.stochasticMaxSamples, 1, 256);

        ImGui::BeginDisabled(true);  // v1: SSAA deferred
        PE::Combo("Supersampling", (int*)&prmStochastic.stochasticSupersamplingFactor, "1x\0" "2x\0" "4x\0");
        ImGui::EndDisabled();

        ImGui::BeginDisabled(true);  // v1: DoF deferred
        PE::Checkbox("Depth of Field", &prmStochastic.stochasticEnableDof);
        ImGui::EndDisabled();

        PE::Checkbox("Progressive", &prmStochastic.stochasticEnableProgressive);

        PE::end();

        ImGui::EndTabItem();
      }
    }
  }
  ImGui::EndTabBar();
}

void VkViewerUI::guiDrawPbrSettings()
{
  namespace PE = nvgui::PropertyEditor;

  PE::begin("## PBR Settings");

  if(PE::entry("Envmap",
               [&]() {
                 ImGui::TextUnformatted(m_envmapFilename.empty() ? "<none>" : m_envmapFilename.filename().string().c_str());
                 ImGui::SameLine();
                 return ImGui::Button("Load##Envmap");
               },
               "Path of the currently loaded HDR environment map.\n"
               "Click 'Load' to select a Radiance HDR (.hdr) file from disk."))
  {
    std::filesystem::path selected = nvgui::windowOpenFileDialog(m_app->getWindowHandle(), "Load HDR Environment Map",
                                                                 "HDR Files|*.hdr;*.exr|All Files|*.*");
    if(!selected.empty())
    {
      m_envmapFilename = selected;
      m_requestUpdateShaders = true;
      LOGI("PBR envmap selected: %s\n", selected.string().c_str());
    }
  }

  if(!m_envmapFilename.empty())
  {
    ImGui::SameLine();
    if(ImGui::Button("Clear##Envmap"))
    {
      m_envmapFilename.clear();
      m_requestUpdateShaders = true;
      LOGI("PBR envmap cleared\n");
    }
  }

  bool pbrEnabled  = prmFrame.pbrEnabled != 0;
  bool irrEnabled  = prmFrame.irradianceEnabled != 0;
  bool tonemapEnab = prmFrame.toneMapEnabled != 0;
  if(PE::Checkbox("PBR shading", &pbrEnabled,
                  "Enable image-based lighting from the loaded environment map.\n"
                  "When enabled, the PBR shader samples the prefiltered cubemap\n"
                  "for specular and the irradiance map for diffuse ambient term."))
  {
    prmFrame.pbrEnabled = pbrEnabled ? 1 : 0;
    m_requestUpdateShaders = true;
  }

  ImGui::BeginDisabled(!pbrEnabled);
  if(PE::Checkbox("Irradiance", &irrEnabled,
                  "Enable diffuse irradiance lookup from the environment map.\n"
                  "Provides a soft ambient term based on convolution of the envmap."))
  {
    prmFrame.irradianceEnabled = irrEnabled ? 1 : 0;
    m_requestUpdateShaders = true;
  }
  ImGui::EndDisabled();

  if(PE::Checkbox("Tone mapping", &tonemapEnab,
                  "Enable HDR tone mapping (AgX) of the final composited image\n"
                  "before display. Compresses high dynamic range into the SDR range\n"
                  "while preserving hue and detail in highlights."))
  {
    prmFrame.toneMapEnabled = tonemapEnab ? 1 : 0;
    resetFrameCounter();
  }

  ImGui::BeginDisabled(!pbrEnabled);
  PE::SliderFloat("Rotation", (float*)&prmFrame.envMapRotation, 0.0f, 6.2831853f, "%.3f", 0,
                  "Yaw rotation of the environment cubemap in radians (0 .. 2π).\n"
                  "Useful for reorienting pre-baked HDRIs without re-baking.");
  PE::SliderFloat("Exposure", (float*)&prmFrame.envMapExposure, 0.1f, 10.0f, "%.2f", 0,
                  "Linear exposure multiplier applied to the environment map samples\n"
                  "before tone mapping. 1.0 = neutral, > 1 brightens, < 1 darkens.");
  ImGui::EndDisabled();

  PE::end();
}
