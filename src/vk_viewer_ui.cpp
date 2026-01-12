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

#include "nvutils/file_operations.hpp"
#include <nvutils/logger.hpp>

#include "nvgui/fonts.hpp"
#include "nvgui/tooltip.hpp"

#include <glm/vec2.hpp>
// clang-format off
#define IM_VEC2_CLASS_EXTRA ImVec2(const glm::vec2& f) {x = f.x; y = f.y;} operator glm::vec2() const { return glm::vec2(x, y); }
// clang-format on

#include <chrono>
#include <thread>
#include <filesystem>
#include <algorithm>  // for std::clamp
#include <fstream>    // for debug mesh export

#include <GLFW/glfw3.h>

#include "vk_viewer_ui.h"
#include "animation_ui.h"
#include "utilities.h"
#include "lcc_loader.h"
#include <backends/imgui_impl_vulkan.h>
#include <imgui/imgui_internal.h>

namespace vk_viewer {

VkViewerUI::VkViewerUI(nvutils::ProfilerManager*   profilerManager,
                                         nvutils::ParameterRegistry* parameterRegistry,
                                         bool*                       benchmarkEnabled)
    : VkViewer(profilerManager, parameterRegistry)
    , m_pBenchmarkEnabled(benchmarkEnabled)
{

  // Register some very sepcific command line parameters, related to benchmarking, other parameters are registered in main or in registerCommandLineParameters

  parameterRegistry->add({"updateData", "Use only in benchmark script. 1=triggers an update of data buffers or textures after a parameter change."},
                         &m_requestUpdateSplatData);

  parameterRegistry->add({.name = "screenshot",
                          .help = "Use only in benchmark script. Takes a screenshot.",
                          .callbackSuccess =
                              [&](const nvutils::ParameterBase* const) {
                                if(m_app)
                                {
                                  m_app->screenShot(m_screenshotFilename);
                                }
                              }},
                          {".png"}, &m_screenshotFilename);

  parameterRegistry->add({.name = "screenshotDelay",
                          .help = "Take screenshot after N seconds and exit. Usage: --screenshotDelay 2.0 --screenshot output.png",
                          .callbackSuccess =
                              [&](const nvutils::ParameterBase* const) {
                                m_autoScreenshotPending = (m_autoScreenshotDelay > 0.0f);
                                m_autoScreenshotTimer = 0.0f;
                              }},
                          &m_autoScreenshotDelay);

  m_supersplatClient = std::make_unique<SupersplatClient>();
};

VkViewerUI::~VkViewerUI(){
    // Nothing to do here
};

void VkViewerUI::onAttach(nvapp::Application* app)
{
    VkViewer::onAttach(app);

  // we hide the UI dy default in benchmark mode
  m_showUI = !(*m_pBenchmarkEnabled);

  // Init combo selectors used in UI

  m_ui.enumAdd(GUI_STORAGE, STORAGE_BUFFERS, "Buffers");
  m_ui.enumAdd(GUI_STORAGE, STORAGE_TEXTURES, "Textures");

  m_ui.enumAdd(GUI_PIPELINE, PIPELINE_VERT, "Raster vertex shader 3DGS");
  m_ui.enumAdd(GUI_PIPELINE, PIPELINE_MESH, "Raster mesh shader 3DGS");
  m_ui.enumAdd(GUI_PIPELINE, PIPELINE_MESH_3DGUT, "Raster mesh shader 3DGUT");
  m_ui.enumAdd(GUI_PIPELINE, PIPELINE_RTX, "Ray tracing 3DGRT");
  m_ui.enumAdd(GUI_PIPELINE, PIPELINE_HYBRID, "Hybrid 3DGS+3DGRT");
  m_ui.enumAdd(GUI_PIPELINE, PIPELINE_HYBRID_3DGUT, "Hybrid 3DGUT+3DGRT");

  m_ui.enumAdd(GUI_EXTENT_METHOD, EXTENT_EIGEN, "Eigen");
  m_ui.enumAdd(GUI_EXTENT_METHOD, EXTENT_CONIC, "Conic");

  m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_FINAL, "Final render");
  m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_CLOCK, "Clock cycles");
  m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_DEPTH, "Splats depth");
  m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_RAYHITS, "Ray Hit Count");

  m_ui.enumAdd(GUI_SORTING, SORTING_GPU_SYNC_RADIX, "GPU radix sort");
  m_ui.enumAdd(GUI_SORTING, SORTING_CPU_ASYNC_MULTI, "CPU async std multi");

  m_ui.enumAdd(GUI_SH_FORMAT, FORMAT_FLOAT32, "Float 32");
  m_ui.enumAdd(GUI_SH_FORMAT, FORMAT_FLOAT16, "Float 16");
  m_ui.enumAdd(GUI_SH_FORMAT, FORMAT_UINT8, "Uint8");

  m_ui.enumAdd(GUI_PARTICLE_FORMAT, PARTICLE_FORMAT_ICOSAHEDRON, "Icosahedron");
  m_ui.enumAdd(GUI_PARTICLE_FORMAT, PARTICLE_FORMAT_PARAMETRIC, "AABB + parametric");

  m_ui.enumAdd(GUI_CAMERA_TYPE, CAMERA_PINHOLE, "Pinhole");
  m_ui.enumAdd(GUI_CAMERA_TYPE, CAMERA_FISHEYE, "Fisheye");

  m_ui.enumAdd(GUI_TEMPORAL_SAMPLING, TEMPORAL_SAMPLING_AUTO, "Automatic");
  m_ui.enumAdd(GUI_TEMPORAL_SAMPLING, TEMPORAL_SAMPLING_ENABLED, "Force enabled");
  m_ui.enumAdd(GUI_TEMPORAL_SAMPLING, TEMPORAL_SAMPLING_DISABLED, "Force disabled");

  m_ui.enumAdd(GUI_KERNEL_DEGREE, KERNEL_DEGREE_QUINTIC, "5 (Quintic)");
  m_ui.enumAdd(GUI_KERNEL_DEGREE, KERNEL_DEGREE_TESSERACTIC, "4 (Tesseractic)");
  m_ui.enumAdd(GUI_KERNEL_DEGREE, KERNEL_DEGREE_CUBIC, "3 (Cubic)");
  m_ui.enumAdd(GUI_KERNEL_DEGREE, KERNEL_DEGREE_QUADRATIC, "2 (Quadratic)");
  m_ui.enumAdd(GUI_KERNEL_DEGREE, KERNEL_DEGREE_LAPLACIAN, "1 (Laplacian)");
  m_ui.enumAdd(GUI_KERNEL_DEGREE, KERNEL_DEGREE_LINEAR, "0 (Linear)");

  m_ui.enumAdd(GUI_LIGHT_TYPE, LIGHT_TYPE_POINT, "Point");
  m_ui.enumAdd(GUI_LIGHT_TYPE, LIGHT_TYPE_DIRECTIONAL, "Directional");

  m_ui.enumAdd(GUI_ILLUM_MODEL, 0, "No indirect");
  m_ui.enumAdd(GUI_ILLUM_MODEL, 1, "Reflective");
  m_ui.enumAdd(GUI_ILLUM_MODEL, 2, "Refractive");

  m_ui.enumAdd(GUI_DIST_SHADER_WG_SIZE, 512, "512");
  m_ui.enumAdd(GUI_DIST_SHADER_WG_SIZE, 256, "256");
  m_ui.enumAdd(GUI_DIST_SHADER_WG_SIZE, 128, "128");
  m_ui.enumAdd(GUI_DIST_SHADER_WG_SIZE, 64, "64");
  m_ui.enumAdd(GUI_DIST_SHADER_WG_SIZE, 32, "32");
  m_ui.enumAdd(GUI_DIST_SHADER_WG_SIZE, 16, "16");

  m_ui.enumAdd(GUI_MESH_SHADER_WG_SIZE, 128, "128");
  m_ui.enumAdd(GUI_MESH_SHADER_WG_SIZE, 64, "64");
  m_ui.enumAdd(GUI_MESH_SHADER_WG_SIZE, 32, "32");
  m_ui.enumAdd(GUI_MESH_SHADER_WG_SIZE, 16, "16");
  m_ui.enumAdd(GUI_MESH_SHADER_WG_SIZE, 8, "8");

  m_ui.enumAdd(GUI_RAY_HIT_PER_PASS, 128, "128");
  m_ui.enumAdd(GUI_RAY_HIT_PER_PASS, 64, "64");
  m_ui.enumAdd(GUI_RAY_HIT_PER_PASS, 32, "32");
  m_ui.enumAdd(GUI_RAY_HIT_PER_PASS, 20, "20");
  m_ui.enumAdd(GUI_RAY_HIT_PER_PASS, 18, "18");
  m_ui.enumAdd(GUI_RAY_HIT_PER_PASS, 16, "16");
  m_ui.enumAdd(GUI_RAY_HIT_PER_PASS, 8, "8");
  m_ui.enumAdd(GUI_RAY_HIT_PER_PASS, 4, "4");

#ifdef WITH_COMFYUI
  m_comfyClient = std::make_unique<ComfyUIClient>();
  m_comfyClient->setCompletionCallback([this](const ComfyUIClient::WorkflowResult& result) {
    onComfyUIWorkflowComplete(result);
  });
#endif
}

void VkViewerUI::onDetach()
{
#ifdef WITH_OPENXR
    destroyHandMeshes();
#endif
    VkViewer::onDetach();
}

void VkViewerUI::onResize(VkCommandBuffer cmd, const VkExtent2D& size)
{
  VkViewer::onResize(cmd, size);
}

void VkViewerUI::onPreRender()
{
#ifdef WITH_COMFYUI
  if (m_comfyClient)
  {
    m_comfyClient->update();
  }
#endif

  // Update animation state
  if (isAnimationActive())
  {
    updateAnimation(ImGui::GetIO().DeltaTime * 1000.0f);
  }

  // Handle auto-screenshot with delay
  if(m_autoScreenshotPending && m_app)
  {
    m_autoScreenshotTimer += ImGui::GetIO().DeltaTime;
    if(m_autoScreenshotTimer >= m_autoScreenshotDelay)
    {
      m_autoScreenshotPending = false;
      if(!m_screenshotFilename.empty())
      {
        m_app->screenShot(m_screenshotFilename);
        LOGI("Auto-screenshot requested: %s\n", m_screenshotFilename.string().c_str());
      }
      // Wait a few frames for screenshot to complete before exiting
      m_autoScreenshotExitCountdown = 10;
    }
  }
  
  // Handle delayed exit after screenshot
  if(m_autoScreenshotExitCountdown > 0)
  {
    m_autoScreenshotExitCountdown--;
    if(m_autoScreenshotExitCountdown == 0 && m_app)
    {
      LOGI("Exiting after screenshot\n");
      m_app->close();
    }
  }

  // Depth video specific controls (when in depth-only mode, camera manipulator is ineffective)
  bool isDepthOnlyMode = m_enableDepthRendering && !m_splatLoader.getStatus() == SplatLoaderAsync::State::STATE_READY && m_meshSetVk.instances.empty();
  bool hasDepthContent = m_enableDepthRendering || m_videoDepthPlaybackMode || m_hlsPlaybackMode;

  if(hasDepthContent && prmFrame.vdzParallaxStrength > 0.0f)
  {
    ImGuiIO& io = ImGui::GetIO();
    float deltaTime = io.DeltaTime;

    // LMB drag: parallax offset (simulates head movement)
    if(io.MouseDown[0])  // Left mouse button
    {
      if(m_lastMousePos.x >= 0.0f)
      {
        glm::vec2 mouseDelta = glm::vec2(io.MouseDelta.x, io.MouseDelta.y);
        prmFrame.vdzParallaxOffset += mouseDelta * m_parallaxSensitivity;
        prmFrame.vdzParallaxOffset.x = std::clamp(prmFrame.vdzParallaxOffset.x, -1.0f, 1.0f);
        prmFrame.vdzParallaxOffset.y = std::clamp(prmFrame.vdzParallaxOffset.y, -1.0f, 1.0f);
      }
      m_lastMousePos = glm::vec2(io.MousePos.x, io.MousePos.y);
    }
    else
    {
      // Reset parallax offset when not dragging
      prmFrame.vdzParallaxOffset = glm::vec2(0.0f, 0.0f);
      m_lastMousePos = glm::vec2(-1.0f, -1.0f);
    }

    // Mouse wheel: focus plane adjustment
    if(io.MouseWheel != 0.0f)
    {
      prmFrame.vdzParallaxFocus = std::clamp(prmFrame.vdzParallaxFocus + io.MouseWheel * 0.1f, 0.0f, 1.0f);
    }

    // WASD / Arrow keys: timeline scrubbing (only when paused or for fine control)
    float scrubSpeed = 1000.0f * deltaTime;  // 1 second per second at default
    if(ImGui::IsKeyDown(ImGuiKey_LeftShift) || ImGui::IsKeyDown(ImGuiKey_RightShift))
    {
      scrubSpeed *= 5.0f;  // Faster scrubbing with shift
    }
    if(ImGui::IsKeyDown(ImGuiKey_LeftCtrl) || ImGui::IsKeyDown(ImGuiKey_RightCtrl))
    {
      scrubSpeed *= 0.1f;  // Slower scrubbing with ctrl
    }

    if(ImGui::IsKeyDown(ImGuiKey_A) || ImGui::IsKeyDown(ImGuiKey_LeftArrow))
    {
      m_playbackTimeOffset -= scrubSpeed;
      m_playbackPaused = true;  // Auto-pause when scrubbing
    }
    if(ImGui::IsKeyDown(ImGuiKey_D) || ImGui::IsKeyDown(ImGuiKey_RightArrow))
    {
      m_playbackTimeOffset += scrubSpeed;
      m_playbackPaused = true;
    }
    if(ImGui::IsKeyDown(ImGuiKey_W) || ImGui::IsKeyDown(ImGuiKey_UpArrow))
    {
      m_playbackTimeOffset += scrubSpeed * 2.0f;  // Page up
      m_playbackPaused = true;
    }
    if(ImGui::IsKeyDown(ImGuiKey_S) || ImGui::IsKeyDown(ImGuiKey_DownArrow))
    {
      m_playbackTimeOffset -= scrubSpeed * 2.0f;  // Page down
      m_playbackPaused = true;
    }

    // Home/End: jump to start/end
    if(ImGui::IsKeyDown(ImGuiKey_Home))
    {
      m_playbackTimeOffset = 0.0;
      m_playbackPaused = true;
    }
    if(ImGui::IsKeyDown(ImGuiKey_End))
    {
      if(m_hlsPlaybackMode && m_hlsMetadata.duration > 0)
      {
        m_playbackTimeOffset = m_hlsMetadata.duration * 1000.0;
      }
      else
      {
        m_playbackTimeOffset = 0.0;  // Default to start
      }
      m_playbackPaused = true;
    }

    // Space: play/pause toggle
    if(ImGui::IsKeyPressed(ImGuiKey_Space, false))
    {
      m_playbackPaused = !m_playbackPaused;
      if(!m_playbackPaused)
      {
        // Resume from current position
        m_playbackStartTime = std::chrono::steady_clock::now();
      }
    }
  }
  else
  {
    m_lastMousePos = glm::vec2(-1.0f, -1.0f);
    prmFrame.vdzParallaxOffset = glm::vec2(0.0f, 0.0f);
  }

  VkViewer::onPreRender();
}

void VkViewerUI::onRender(VkCommandBuffer cmd)
{
#ifdef WITH_OPENXR
  // Update hand meshes before rendering
  updateHandMeshes();
#endif

  VkViewer::onRender(cmd);

#ifdef WITH_OPENXR
  // Render hand meshes after main scene
  // Only render if XR is fully initialized and we have valid rendering resources
  // Also wait a few frames after init to ensure all resources are ready
  if (m_handMeshReadyFrameDelay > 0) {
    m_handMeshReadyFrameDelay--;
  }
  if (m_xr && m_xr->handsSupported() && m_xrInitialized && m_descriptorSet != VK_NULL_HANDLE && m_handMeshReadyFrameDelay == 0) {
    auto poseToMatrix = [](const XrPosef& pose) -> glm::mat4 {
      glm::quat q(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
      glm::vec3 t(pose.position.x, pose.position.y, pose.position.z);
      return glm::translate(glm::mat4(1.0f), t) * glm::mat4_cast(q);
    };

    const auto& leftHand = m_xr->getHandInput(GsOpenXr::Hand::Left);
    if (leftHand.tracked) {
      glm::mat4 wristTransform = poseToMatrix(leftHand.jointPoses[XR_HAND_JOINT_WRIST_EXT]);
      renderHandMesh(cmd, m_leftHandMesh, wristTransform);
    }
    else if (m_leftHandMesh.initialized && m_debugForceRenderHands) {
      // Debug: render at fixed position in front of camera when not tracked
      glm::mat4 debugTransform = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -0.5f));
      renderHandMesh(cmd, m_leftHandMesh, debugTransform);
    }

    const auto& rightHand = m_xr->getHandInput(GsOpenXr::Hand::Right);
    if (rightHand.tracked) {
      glm::mat4 wristTransform = poseToMatrix(rightHand.jointPoses[XR_HAND_JOINT_WRIST_EXT]);
      renderHandMesh(cmd, m_rightHandMesh, wristTransform);
    }
    else if (m_rightHandMesh.initialized && m_debugForceRenderHands) {
      // Debug: render at fixed position in front of camera when not tracked
      glm::mat4 debugTransform = glm::translate(glm::mat4(1.0f), glm::vec3(0.2f, 0.0f, -0.5f));
      renderHandMesh(cmd, m_rightHandMesh, debugTransform);
    }
  }
#endif
}

#define ICON_BLANK "     "

void VkViewerUI::onUIMenu()
{
  static bool close_app{false};
  bool        v_sync = m_app->isVsync();
#ifndef NDEBUG
  static bool s_showDemo{false};
  static bool s_showDemoPlot{false};
  static bool s_showDemoIcons{false};
#endif
  if(ImGui::BeginMenu("File"))
  {
    if(ImGui::MenuItem(ICON_MS_FILE_OPEN " Open file", ""))
    {
      prmScene.sceneToLoadFilename = nvgui::windowOpenFileDialog(m_app->getWindowHandle(), "Load splat file",
                                                                 "All Files|*.ply;*.spz;*.sog;*.4dv|PLY Files|*.ply|SPZ files|*.spz|SOG files|*.sog|4DV files|*.4dv");
      prmScene.addSceneToExisting = false;
    }
    if(ImGui::MenuItem(ICON_MS_ADD " Add file", ""))
    {
      prmScene.sceneToLoadFilename = nvgui::windowOpenFileDialog(m_app->getWindowHandle(), "Add splat file",
                                                                 "All Files|*.ply;*.spz;*.sog;*.4dv|PLY Files|*.ply|SPZ files|*.spz|SOG files|*.sog|4DV files|*.4dv");
      prmScene.addSceneToExisting = true;
    }
    if(ImGui::MenuItem(ICON_MS_FOLDER_OPEN " Load from Resources...", ""))
    {
      m_showFileDialog = true;
      m_fileList.clear();
      std::vector<std::filesystem::path> resourceDirs = getResourcesDirs();
      if (!resourceDirs.empty())
      {
        std::filesystem::path resourcesDir = resourceDirs[0];
        if (std::filesystem::exists(resourcesDir))
        {
          for (const auto& entry : std::filesystem::directory_iterator(resourcesDir))
          {
            if (entry.is_regular_file())
              m_fileList.push_back(entry.path().string());
          }
        }
      }
    }
    if(ImGui::MenuItem(ICON_MS_CLOUD_DOWNLOAD " Supersplat...", ""))
    {
      m_showSupersplatDialog = true;
      if (m_supersplatClient)
      {
        m_supersplatClient->fetchSceneList("", [this](const std::vector<SupersplatClient::Scene>& scenes) {
          std::lock_guard<std::mutex> lock(m_thumbnailMutex);
          m_supersplatScenes = scenes;
          
          for (const auto& scene : scenes) {
              if (!scene.thumbnailUrl.empty()) {
                  m_supersplatClient->fetchThumbnail(scene.thumbnailUrl, 
                      [this, url=scene.thumbnailUrl](const std::vector<uint8_t>& data, int w, int h, int c) {
                          if (data.empty()) return;
                          std::lock_guard<std::mutex> lock(m_thumbnailMutex);
                          std::vector<uint8_t> rgba = data;
                          if (c == 3) {
                              rgba.resize(w * h * 4);
                              for (int i = w * h - 1; i >= 0; --i) {
                                  rgba[i * 4 + 3] = 255;
                                  rgba[i * 4 + 2] = data[i * 3 + 2];
                                  rgba[i * 4 + 1] = data[i * 3 + 1];
                                  rgba[i * 4 + 0] = data[i * 3 + 0];
                              }
                          }
                          m_pendingThumbnails.push_back({url, rgba, w, h});
                      });
              }
          }
        });
      }
    }
    if(ImGui::MenuItem(ICON_MS_CLOUD_DOWNLOAD " Load from SuperSplat URL...", ""))
    {
      m_showSuperSplatUrlPopup = true;
    }
    if(ImGui::MenuItem(ICON_MS_RESTORE_PAGE " Re Open", "F5", false, !m_radianceFields.empty()))
    {
      prmScene.sceneToLoadFilename = getLoadedSceneFilename();
    }
    if(ImGui::BeginMenu(ICON_MS_HISTORY " Recent Files"))
    {
      for(const auto& file : m_recentFiles)
      {
        if(ImGui::MenuItem(file.string().c_str()))
        {
          prmScene.sceneToLoadFilename = file;
        }
      }
      ImGui::EndMenu();
    }
    if(ImGui::BeginMenu(ICON_MS_PALETTE " Color Space"))
    {
      if(ImGui::MenuItem("None (standard 3DGS)", "", prmScene.colorSpaceConversion == 0))
        prmScene.colorSpaceConversion = 0;
      if(ImGui::MenuItem("sRGB to Linear (ML-SHARP)", "", prmScene.colorSpaceConversion == 1))
        prmScene.colorSpaceConversion = 1;
      ImGui::Separator();
      ImGui::TextDisabled("Applied when loading PLY files");
      ImGui::EndMenu();
    }
    
    ImGui::Separator();
    if(ImGui::MenuItem(ICON_MS_FILE_OPEN " Open project", ""))
    {
      prmScene.projectToLoadFilename =
          nvgui::windowOpenFileDialog(m_app->getWindowHandle(), "Load project file", "VKGS Files|*.vkgs");
    }
    if(ImGui::BeginMenu(ICON_MS_HISTORY " Recent projects"))
    {
      for(const auto& file : m_recentProjects)
      {
        if(ImGui::MenuItem(file.string().c_str()))
        {
          prmScene.projectToLoadFilename = file;
        }
      }
      ImGui::EndMenu();
    }
    if(ImGui::MenuItem(ICON_MS_FILE_SAVE " Save project", ""))
    {
      auto path = nvgui::windowSaveFileDialog(m_app->getWindowHandle(), "Save project file", "VKGS Files|*.vkgs");
      if(!path.empty())
      {
        saveProject(path.string());
      }
    }
    ImGui::Separator();
    if(ImGui::MenuItem(ICON_MS_MOVIE " Open Depth Video (metadata.json)...", ""))
    {
      auto metadataPath = nvgui::windowOpenFileDialog(m_app->getWindowHandle(), "Select metadata.json or Folder",
                                                      "JSON Files|metadata.json;*.json|All Files|*");
      if(!metadataPath.empty())
      {
        enableDepthVideoPlayback(metadataPath.string());
        prmFrame.vdzUseVideoTexture = 1;
        m_requestUpdateShaders = true;
      }
    }
    ImGui::Separator();
    if(ImGui::MenuItem(ICON_MS_MOVIE " Open PLY Sequence...", ""))
    {
      auto path = nvgui::windowOpenFileDialog(m_app->getWindowHandle(), "Select PLY Sequence", "PLY Files|*.ply");
      if(!path.empty())
      {
        // If user picked a file, use its parent folder
        if(std::filesystem::is_regular_file(path))
        {
          path = path.parent_path();
        }
        if(std::filesystem::is_directory(path))
        {
          enablePlySequencePlayback(path);
        }
      }
    }
    ImGui::Separator();
    if(ImGui::MenuItem(ICON_MS_SCAN_DELETE " Close", ""))
    {
      deinitAll();
    }
    ImGui::Separator();
    if(ImGui::MenuItem(ICON_MS_EXIT_TO_APP " Exit", "Ctrl+Q"))
    {
      close_app = true;
    }
    ImGui::EndMenu();
  }
  if(ImGui::BeginMenu("View"))
  {
    ImGui::MenuItem(ICON_MS_BOTTOM_PANEL_OPEN " V-Sync", "Ctrl+Shift+V", &v_sync);
    ImGui::MenuItem(ICON_MS_SPACE_DASHBOARD " ShowUI", "", &m_showUI);
    ImGui::Separator();
    ImGui::MenuItem(ICON_MS_QUERY_STATS " Depth Performance", "", &m_showDepthPerformance);

#ifdef WITH_COMFYUI
    ImGui::Separator();
    ImGui::MenuItem(ICON_MS_AUTO_AWESOME " ComfyUI Generator", "", &m_showComfyUIWindow);
#endif
    ImGui::EndMenu();
  }
#ifndef NDEBUG
  if(ImGui::BeginMenu("Debug"))
  {
    ImGui::MenuItem("Show ImGui Demo", nullptr, &s_showDemo);
    ImGui::MenuItem("Show ImPlot Demo", nullptr, &s_showDemoPlot);
    ImGui::MenuItem("Show Icons Demo", nullptr, &s_showDemoIcons);
    ImGui::EndMenu();
  }
#endif  // !NDEBUG

  // Shortcuts
  if(ImGui::IsKeyPressed(ImGuiKey_Space))
  {
    m_lastLoadedCamera = (m_lastLoadedCamera + 1) % m_cameraSet.size();
    m_cameraSet.loadPreset(m_lastLoadedCamera, false);
    m_requestUpdateShaders = true;
  }
  if(ImGui::IsKeyPressed(ImGuiKey_Q) && ImGui::IsKeyDown(ImGuiKey_LeftCtrl))
  {
    close_app = true;
  }

  if(ImGui::IsKeyPressed(ImGuiKey_V) && ImGui::IsKeyDown(ImGuiKey_LeftCtrl) && ImGui::IsKeyDown(ImGuiKey_LeftShift))
  {
    v_sync = !v_sync;
  }
  if(ImGui::IsKeyPressed(ImGuiKey_F5))
  {
    if(!m_recentFiles.empty())
      prmScene.sceneToLoadFilename = m_recentFiles[0];
  }
  if(ImGui::IsKeyPressed(ImGuiKey_F1))
  {
    std::string statsFrame;
    std::string statsSingle;
    m_profilerManager->appendPrint(statsFrame, statsSingle, true);
    // print old stats
    nvutils::Logger::getInstance().log(nvutils::Logger::eSTATS, "ParameterSequence %d \"%s\" = {\n%s\n%s}\n", 0,
                                       "F1 pressed ", statsFrame.c_str(), statsSingle.c_str());
  }
  if(ImGui::IsKeyPressed(ImGuiKey_1))
    prmSelectedPipeline = PIPELINE_VERT;
  if(ImGui::IsKeyPressed(ImGuiKey_2))
    prmSelectedPipeline = PIPELINE_MESH;
  if(ImGui::IsKeyPressed(ImGuiKey_3))
    prmSelectedPipeline = PIPELINE_RTX;
  if(ImGui::IsKeyPressed(ImGuiKey_4))  // TODO find why the shortcut does not work
    prmSelectedPipeline = PIPELINE_HYBRID;

  // hot rebuild of shaders only if scene exist
  if(ImGui::IsKeyPressed(ImGuiKey_R))
  {
    if(!m_radianceFields.empty())
      m_requestUpdateShaders = true;
    else
      LOGW("No scene loaded, cannot rebuild shader\n");
  }
  if(close_app)
  {
    m_app->close();
  }
#ifndef NDEBUG
  if(s_showDemo)
  {
    ImGui::ShowDemoWindow(&s_showDemo);
  }
  if(s_showDemoPlot)
  {
    //ImPlot::ShowDemoWindow(&s_showDemoPlot);
  }
  if(s_showDemoIcons)
  {
    //nvgui::showDemoIcons();
  }
#endif  // !NDEBUG

  if(m_app->isVsync() != v_sync)
  {
    m_app->setVsync(v_sync);
  }

  if(ImGui::IsKeyPressed(ImGuiKey_P))
    dumpSplat(m_indirectReadback.particleID);
}

void VkViewerUI::onFileDrop(const std::filesystem::path& filename)
{
  // extension To lower case
  std::string extension = filename.extension().string();
  std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

  //
  if(extension == ".ply")
    prmScene.sceneToLoadFilename = filename;
  else if(extension == ".spz")
    prmScene.sceneToLoadFilename = filename;
  else if(extension == ".sog")
    prmScene.sceneToLoadFilename = filename;
  else if(extension == ".4dv")
    prmScene.sceneToLoadFilename = filename;
  else if(extension == ".vkgs")
    prmScene.projectToLoadFilename = filename;
  else if(extension == ".obj")
    prmScene.meshToImportFilename = filename;
  else
    LOGE("Error: unsupported file extension %s\n", extension.c_str());
}

void VkViewerUI::onUIRender()
{
  /////////////
  // Rendering Viewport display the GBuffer
  {
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
    ImGui::Begin("Viewport");

    // Display the G-Buffer image
    ImGui::Image((ImTextureID)m_gBuffers.getDescriptorSet(), ImGui::GetContentRegionAvail());

    ImVec2 wp = ImGui::GetWindowPos();
    ImVec2 ws = ImGui::GetWindowSize();

    // display the basis widget at bottom left
    float  size   = 25.F;
    ImVec2 offset = ImVec2(size * 1.1F, -size * 1.1F) * ImGui::GetWindowDpiScale();
    ImVec2 pos    = ImVec2(wp.x, wp.y + ws.y) + offset;
    nvgui::Axis(pos, cameraManip->getViewMatrix(), size);

    // store mouse cursor
    // will be available for next frame in frameInfo
    ImVec2 mp = ImGui::GetMousePos();  // Mouse position in screen space

    // Convert to viewport space (0,0 at bottom-left)
    ImVec2 mouseInViewport = ImVec2(mp.x - wp.x, mp.y - wp.y);

    if(mouseInViewport.x < 0 || mouseInViewport.y < 0 || mouseInViewport.x >= ws.x || mouseInViewport.y >= ws.y)
      prmFrame.cursor.x = prmFrame.cursor.y = -1;  // just so it is easy to test in shader if pos is valid
    else
      prmFrame.cursor = {mouseInViewport.x, mouseInViewport.y};

    ImGui::End();
    ImGui::PopStyleVar();
  }

  /////////////////
  // Handle project loading, may trigger a scene loading
  loadProjectIfNeeded();

  /////////////////
  // Handle radiance field deletion request
  if(m_requestDeleteRadianceField && m_radianceFieldToDelete < m_radianceFields.size())
  {
    // For now, deleting a radiance field clears all and requires re-adding
    // This is because the merged splat set would need to be rebuilt from remaining files
    // TODO: Implement proper incremental deletion by reloading remaining files
    vkDeviceWaitIdle(m_device);
    deinitAll();
    m_radianceFields.clear();
    m_splatSet.clear();
    LOGI("Radiance field deleted. All radiance fields cleared.\n");
    m_requestDeleteRadianceField = false;
    m_selectedItemIndex = -1;
  }
  m_requestDeleteRadianceField = false;

  /////////////////
  // Handle scene loading
  if(m_showSuperSplatUrlPopup)
  {
    ImGui::OpenPopup("Load from SuperSplat URL");
    m_showSuperSplatUrlPopup = false;
  }

  // Always center this window when appearing
  ImVec2 centerModal = ImGui::GetMainViewport()->GetCenter();
  ImGui::SetNextWindowPos(centerModal, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

  if(ImGui::BeginPopupModal("Load from SuperSplat URL", NULL, ImGuiWindowFlags_AlwaysAutoResize))
  {
    static char urlBuf[2048] = "";
    static bool firstFocus = true;
    if (firstFocus) {
        ImGui::SetKeyboardFocusHere();
        firstFocus = false;
    }
    
    ImGui::Text("Enter the URL of the .sog or .ply file:");
    bool enterPressed = ImGui::InputText("URL", urlBuf, IM_ARRAYSIZE(urlBuf), ImGuiInputTextFlags_EnterReturnsTrue);
    
    if(ImGui::Button("Load", ImVec2(120, 0)) || enterPressed)
    {
      if(strlen(urlBuf) > 0)
      {
        prmScene.sceneToLoadFilename = std::string(urlBuf);
        prmScene.addSceneToExisting = false;
        firstFocus = true;
        ImGui::CloseCurrentPopup();
      }
    }
    ImGui::SameLine();
    if(ImGui::Button("Cancel", ImVec2(120, 0)))
    {
      firstFocus = true;
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }

#ifdef WITH_DEFAULT_SCENE_FEATURE
  // load a default scene if none was provided by command line
  if(prmScene.enableDefaultScene && m_radianceFields.empty() && prmScene.sceneToLoadFilename.empty()
     && m_splatLoader.getStatus() == SplatLoaderAsync::State::STATE_READY && !isDepthVideoPlaying())
  {
    const std::vector<std::filesystem::path> defaultSearchPaths = getResourcesDirs();
    prmScene.sceneToLoadFilename = nvutils::findFile("flowers_1/flowers_1.ply", defaultSearchPaths).string();
    prmScene.enableDefaultScene  = false;
  }
#endif

  // do we need to load a new scene ?
  if(!prmScene.sceneToLoadFilename.empty() && m_splatLoader.getStatus() == SplatLoaderAsync::State::STATE_READY)
  {
    // Show confirmation popup only when replacing existing scene (not adding)
    if(!m_radianceFields.empty() && prmScene.projectToLoadFilename.empty() && !prmScene.addSceneToExisting)
      ImGui::OpenPopup("Load .ply file ?");

    // Always center this window when appearing
    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

    bool doLoad = true;

    if(ImGui::BeginPopupModal("Load .ply file ?", NULL, ImGuiWindowFlags_AlwaysAutoResize))
    {
      doLoad = false;

      ImGui::Text("The current project will be entirely replaced.\nThis operation cannot be undone!");
      ImGui::Separator();

      if(ImGui::Button("OK", ImVec2(120, 0)))
      {
        doLoad = true;
        ImGui::CloseCurrentPopup();
      }
      ImGui::SetItemDefaultFocus();
      ImGui::SameLine();
      if(ImGui::Button("Cancel", ImVec2(120, 0)))
      {
        // cancel any request leading to a reset
        prmScene.sceneToLoadFilename   = "";
        prmScene.projectToLoadFilename = "";
        prmScene.addSceneToExisting    = false;
        ImGui::CloseCurrentPopup();
      }
      ImGui::EndPopup();
    }

    if(doLoad)
    {
      // If replacing (not adding), reset existing scene
      if(!prmScene.addSceneToExisting)
      {
        const auto splatCount = m_splatSet.positions.size() / 3;
        if(splatCount)
        {
          deinitAll();
        }
        m_radianceFields.clear();
      }
      else
      {
        // When adding, we need to deinit GPU resources but keep CPU data
        const auto splatCount = m_splatSet.positions.size() / 3;
        if(splatCount)
        {
          vkDeviceWaitIdle(m_device);
          m_splatSetVk.deinitDataStorage();
          m_splatSetVk.rtxDeinitSplatModel();
          m_splatSetVk.rtxDeinitAccelerationStructures();
        }
      }

      //
      vkDeviceWaitIdle(m_device);

      LOGI("Start loading file %s (add=%s)\n", prmScene.sceneToLoadFilename.string().c_str(),
           prmScene.addSceneToExisting ? "true" : "false");

      if (std::filesystem::is_directory(prmScene.sceneToLoadFilename))
      {
        // Check for LCC format first
        if(LccLoader::canLoad(prmScene.sceneToLoadFilename))
        {
          // Store the pending filename for when load completes
          m_pendingLoadFilename = prmScene.sceneToLoadFilename;
          m_splatSetPending.clear();
          if(!m_splatLoader.loadScene(prmScene.sceneToLoadFilename, m_splatSetPending))
          {
            LOGE("Error: cannot start scene load while loader is not ready status=%d\n", static_cast<int>(m_splatLoader.getStatus()));
          }
          else
          {
            ImGui::OpenPopup("Loading");
          }
        }
        else
        {
          // Default to PLY sequence playback
          enablePlySequencePlayback(prmScene.sceneToLoadFilename);
          prmScene.sceneToLoadFilename.clear();
        }
      }
      else if (prmScene.sceneToLoadFilename.extension() == ".json")
      {
        enableDepthVideoPlayback(prmScene.sceneToLoadFilename.string());
        prmFrame.vdzUseVideoTexture = 1;
        m_requestUpdateShaders = true;
        prmScene.sceneToLoadFilename.clear();
      }
      else
      {
        // Store the pending filename for when load completes
        m_pendingLoadFilename = prmScene.sceneToLoadFilename;
        
        // Load into pending set (will be merged on success)
        m_splatSetPending.clear();
        if(!m_splatLoader.loadScene(prmScene.sceneToLoadFilename, m_splatSetPending))
        {
          // this should never occur since status is READY.
          LOGE("Error: cannot start scene load while loader is not ready status=%d\n", static_cast<int>(m_splatLoader.getStatus()));
        }
        else
        {
          // open the modal window that will collect results
          ImGui::OpenPopup("Loading");
        }
      }

      // reset request
      prmScene.sceneToLoadFilename.clear();
    }
  }

  // display loading jauge modal window
  // Always center this window when appearing
  ImVec2 center = ImGui::GetMainViewport()->GetCenter();
  ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
  if(ImGui::BeginPopupModal("Loading", NULL, ImGuiWindowFlags_AlwaysAutoResize))
  {
    // specific wait for benchmarking mode
    // prevent display of loading jauge and frame advancing while loading
    // ensure scene is loaded before moving to next frame
    if(*m_pBenchmarkEnabled)
    {
      while(m_splatLoader.getStatus() == SplatLoaderAsync::State::STATE_LOADING)
      {
        using namespace std::chrono_literals;
        std::this_thread::sleep_for(100ms);
      }
    }
    // managment of async load
    switch(m_splatLoader.getStatus())
    {
      case SplatLoaderAsync::State::STATE_LOADING: {
        ImGui::Text("%s", m_splatLoader.getFilename().string().c_str());
        ImGui::ProgressBar(m_splatLoader.getProgress(), ImVec2(ImGui::GetContentRegionAvail().x, 0.0f));
      }
      break;
      case SplatLoaderAsync::State::STATE_FAILURE: {
        ImGui::Text("Error: invalid ply file");
        if(ImGui::Button("Ok", ImVec2(120, 0)))
        {
          m_pendingLoadFilename = "";
          m_splatSetPending.clear();
          // destroy scene just in case it was
          // loaded but not properly since in error
          deinitScene();
          // set ready for next load
          m_splatLoader.reset();
          prmScene.addSceneToExisting = false;
          ImGui::CloseCurrentPopup();
        }
      }
      break;
      case SplatLoaderAsync::State::STATE_LOADED: {
        const std::string ext = m_pendingLoadFilename.extension().string();
        const bool        isSog = (ext == ".sog" || m_pendingLoadFilename.filename() == "meta.json");

        // Apply color space conversion if requested (for ML-SHARP files)
        if(prmScene.colorSpaceConversion == 1 || isSog)
        {
          if(!isSog)
          {
            LOGI("Converting color space: sRGB -> linearRGB (for ML-SHARP compatibility files)\n");
            m_splatSetPending.convertColorSpace(true);  // sRGB to linear
          }
          else
          {
            LOGI("SOG file loaded (linearized)\n");
          }

          // Auto-enable linear-to-sRGB post-processing for correct display
          if(prmFrame.linearToSrgb == 0)
          {
            prmFrame.linearToSrgb = 1;
            LOGI("Auto-enabled Linear to sRGB output for linearized content\n");
          }
        }

        // Remove black splats if requested
        if(prmScene.removeBlackSplats)
        {
          const size_t countBefore = m_splatSetPending.size();
          m_splatSetPending.removeBlackSplats();
          const size_t countAfter = m_splatSetPending.size();
          LOGI("Removed %zu black splats (%zu remaining)\n", countBefore - countAfter, countAfter);
        }

        // Merge the pending splat set into main splat set
        const size_t newSplatOffset = m_splatSet.merge(m_splatSetPending);
        const size_t newSplatCount = m_splatSetPending.size();
        
        // Add radiance field entry
        RadianceFieldEntry entry;
        entry.filename = m_pendingLoadFilename;
        entry.displayName = m_pendingLoadFilename.filename().string();
        entry.splatOffset = newSplatOffset;
        entry.splatCount = newSplatCount;
        entry.visible = true;
        m_radianceFields.push_back(entry);
        
        LOGI("Added radiance field: %s (offset=%zu, count=%zu, total=%zu)\n",
             entry.displayName.c_str(), newSplatOffset, newSplatCount, m_splatSet.size());
        
        // Clear pending data
        m_splatSetPending.clear();

        // If we are in the middle of a reload queue, trigger the next file
        if(!m_reloadQueue.empty())
        {
          prmScene.sceneToLoadFilename = m_reloadQueue.front();
          m_reloadQueue.erase(m_reloadQueue.begin());
          prmScene.addSceneToExisting = true;
          m_pendingLoadFilename = prmScene.sceneToLoadFilename;
        }
        else
        {
          // Queue finished or single file load
          if(!initAll())
          {
            // destroy scene
            deinitScene();
          }
          else if(!m_isReloading) // Only add to recent if not a reload
          {
            guiAddToRecentFiles(entry.filename);
          }
          
          m_isReloading = false;
          m_pendingLoadFilename = "";
          prmScene.addSceneToExisting = false;
          // set ready for next load
          m_splatLoader.reset();
          ImGui::CloseCurrentPopup();
        }
      }
      break;
      default: {
        // nothing to do for READY or SHUTDOWN
      }
    }
    ImGui::EndPopup();
  }

  /////////////////
  // Handle depth frame loading
  // (VDZ loading removed)

  if(!m_showUI)
    return;

  /////////////////
  // Draw the UI parts

  guiDrawAssetsWindow();
    guiDrawPropertiesWindow();
    guiDrawRendererStatisticsWindow();
    guiDrawMemoryStatisticsWindow();

    // Animation UI
    if (m_animationUI) {
        m_animationUI->renderAnimationControls(true);
    }

  guiDrawFooterBar();

  if(m_showDepthPerformance)
  {
    guiDrawPerformancePanel();
  }

  // Process pending thumbnails
  {
      std::lock_guard<std::mutex> lock(m_thumbnailMutex);
      for (const auto& pt : m_pendingThumbnails) {
          if (m_thumbnailTextures.find(pt.url) == m_thumbnailTextures.end()) {
              nvvk::Image texture;
              VkImageView view;
              createTextureFromRGBA(pt.data, pt.w, pt.h, texture, view);
              m_thumbnailTextures[pt.url] = texture;
              m_thumbnailViews[pt.url] = view;
              
              VkDescriptorSet ds = ImGui_ImplVulkan_AddTexture(m_sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
              m_thumbnailDescriptors[pt.url] = ds;
          }
      }
      m_pendingThumbnails.clear();
  }

  guiDrawFileDialog();
  guiDrawSupersplatDialog();
  if (m_showVrMenu)
    guiDrawVrMenu();

#ifdef WITH_COMFYUI
  if (m_showComfyUIWindow)
  {
    guiDrawComfyUIWindow();
  }
#endif
}

void VkViewerUI::guiDrawAssetsWindow()
{
  ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyle().Colors[ImGuiCol_ChildBg]);

  if(ImGui::Begin("Assets"))
  {
    guiDrawRendererTree();

    guiDrawCameraTree();

    guiDrawLightTree();

    guiDrawRadianceFieldsTree();

    guiDrawObjectTree();

    guiDrawDepthStreamTree();
  }
  ImGui::End();

  ImGui::PopStyleColor();
}

void VkViewerUI::guiDrawRendererTree()
{
  static ImGuiTreeNodeFlags base_flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick;

  bool node_open = false;

  ImGuiTreeNodeFlags node_flags;

  // Renderer
  std::string pipelineName = m_ui.getEnums(GUI_PIPELINE)[prmSelectedPipeline].name;
  node_flags               = base_flags;
  if(m_selectedAsset == GUI_RENDERER)
    node_flags |= ImGuiTreeNodeFlags_Selected;
  ImGui::SetNextItemOpen(true, ImGuiCond_Once);
  node_open = ImGui::TreeNodeEx(ICON_MS_CAMERA " Renderer", node_flags);
  if(ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen())
  {
    m_selectedAsset     = GUI_RENDERER;
    m_selectedItemIndex = -1;
  }
  if(node_open)
  {
    // display the pipeline selector
    int i = 0;
    {
      ImGui::Indent(30);
      ImGui::Text(ICON_MS_SUBDIRECTORY_ARROW_RIGHT);
      ImGui::SameLine();
      if(m_ui.enumCombobox(GUI_PIPELINE, "##ID", &prmSelectedPipeline))
      {
        m_requestUpdateShaders = true;
      }
      ImGui::Unindent(30);
    }
    ImGui::TreePop();
  }
}

void VkViewerUI::guiDrawCameraTree()
{

  const ImGuiTreeNodeFlags base_flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick;

  ImGuiTreeNodeFlags node_flags = base_flags;

  if(m_selectedAsset == GUI_CAMERA && m_selectedItemIndex == -1)
    node_flags |= ImGuiTreeNodeFlags_Selected;

  bool node_open = ImGui::TreeNodeEx(ICON_MS_PHOTO_CAMERA " Camera", node_flags);
  if(ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen())
  {
    m_selectedAsset     = GUI_CAMERA;
    m_selectedItemIndex = -1;
  }
  ImGui::PushID(-1);
  ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - 70);
  if(ImGui::SmallButton(ICON_MS_ADD_A_PHOTO))
  {
    m_cameraSet.storeCurrentCamera();
  }
  nvgui::tooltip("Store current camera settings in presets");
  ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - 30);
  if(ImGui::SmallButton(ICON_MS_FILE_OPEN))
  {
    auto name = nvgui::windowOpenFileDialog(m_app->getWindowHandle(), "Import INRIA Camera file", "INRIA Camera file|*.json");
    if(!name.empty())
    {
      importCamerasINRIA(name.string(), m_cameraSet);
    }
  }
  nvgui::tooltip("Import INRIA Camera file");
  ImGui::PopID();

  if(node_open)
  {
    // display the camera tree
    for(int i = 0; i < m_cameraSet.size(); ++i)
    {
      ImGui::PushID(i);
      node_flags = base_flags | ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
      if(m_selectedAsset == GUI_CAMERA && m_selectedItemIndex == i)
        node_flags |= ImGuiTreeNodeFlags_Selected;

      const auto name = i == 0 ? fmt::format(ICON_MS_SUBDIRECTORY_ARROW_RIGHT "Default Preset ", i) :
                                 fmt::format(ICON_MS_SUBDIRECTORY_ARROW_RIGHT "Camera Preset ({})", i);

      bool node_open = ImGui::TreeNodeEx(name.c_str(), node_flags);
      if(ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen())
      {
        m_selectedAsset     = GUI_CAMERA;
        m_selectedItemIndex = i;
      }
      ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - 110);
      if(ImGui::SmallButton(ICON_MS_LOCAL_SEE))
      {
        if(m_cameraSet.getPreset(i).model != m_cameraSet.getCamera().model)
        {
          m_requestUpdateShaders = true;
        }
        m_cameraSet.loadPreset(i, false);
        m_lastLoadedCamera     = i;
        m_selectedItemIndex    = -1;  // Will select current camera
        m_requestUpdateShaders = true;
      }
      nvgui::tooltip("Load camera preset");
      if(i > 0)
      {
        ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - 70);
        if(ImGui::SmallButton(ICON_MS_ADD_A_PHOTO))
        {
          m_cameraSet.setPreset(i, m_cameraSet.getCamera());
          m_lastLoadedCamera     = i;
          m_selectedItemIndex    = -1;  // Will select current camera
          m_requestUpdateShaders = true;
        }
        nvgui::tooltip("Overwrite preset with current camera settings");
      }
      // Delete button only if not default
      if(i != 0)
      {
        ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - 30);
        if(ImGui::SmallButton(ICON_MS_DELETE))
        {
          m_cameraSet.erasePreset(i);
        }
        nvgui::tooltip("Delete preset");
      }
      ImGui::PopID();
    }
    //
    ImGui::TreePop();
  }
}

void VkViewerUI::guiDrawLightTree()
{
  const ImGuiTreeNodeFlags base_flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick;

  ImGuiTreeNodeFlags node_flags = base_flags;
  if(m_selectedAsset == GUI_LIGHT && m_selectedItemIndex == -1)
    node_flags |= ImGuiTreeNodeFlags_Selected;

  bool node_open = ImGui::TreeNodeEx(ICON_MS_LIGHT_MODE " Lights", node_flags);
  if(ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen())
  {
    m_selectedAsset     = GUI_NONE;
    m_selectedItemIndex = -1;
  }
  ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - 70);
  if(ImGui::SmallButton(ICON_MS_ADD_CIRCLE))
  {
    m_selectedItemIndex         = m_lightSet.createLight();
    m_requestUpdateLightsBuffer = true;
  }
  nvgui::tooltip("Create light");

  if(node_open)
  {
    // display the lights tree
    for(int i = 0; i < m_lightSet.size(); ++i)
    {
      ImGui::PushID(i);
      ImGuiTreeNodeFlags node_flags = base_flags | ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
      if(m_selectedAsset == GUI_LIGHT && m_selectedItemIndex == i)
        node_flags |= ImGuiTreeNodeFlags_Selected;

      bool node_open = ImGui::TreeNodeEx((void*)(intptr_t)i, node_flags, ICON_MS_SUBDIRECTORY_ARROW_RIGHT "Light %d", i);
      if(ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen())
      {
        m_selectedAsset     = GUI_LIGHT;
        m_selectedItemIndex = i;
      }
      if(m_lightSet.size() > 1)
      {
        ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - 30);
        if(ImGui::SmallButton(ICON_MS_DELETE))
        {
          m_lightSet.eraseLight(i);
          m_requestUpdateLightsBuffer = true;
          // deselect all
          m_selectedAsset     = GUI_NONE;
          m_selectedItemIndex = -1;
        }
        nvgui::tooltip("Delete light");
      }
      ImGui::PopID();
    }
    ImGui::TreePop();
  }
}

void VkViewerUI::guiDrawRadianceFieldsTree()
{

  const ImGuiTreeNodeFlags base_flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick;

  ImGuiTreeNodeFlags node_flags = base_flags;

  if(m_selectedAsset == GUI_SPLATSET && m_selectedItemIndex == -1)
    node_flags |= ImGuiTreeNodeFlags_Selected;

  ImGui::SetNextItemOpen(true, ImGuiCond_Once);
  std::string rtxError = " ";
  if(m_splatSet.size() != 0 && !m_splatSetVk.rtxValid)
  {
    rtxError = " Error: RTX allocation failed";
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.0f, 0.0f, 1.0f));
  }
  bool node_open = ImGui::TreeNodeEx(
      fmt::format(ICON_MS_GRAIN " Radiance Fields ({}){}", m_radianceFields.size(), rtxError).c_str(), node_flags);
  if(m_splatSet.size() != 0 && !m_splatSetVk.rtxValid)
    ImGui::PopStyleColor();
  if(ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen())
  {
    m_selectedAsset     = GUI_NONE;
    m_selectedItemIndex = -1;
  }
  
  // Add button to load additional radiance field
  ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - 30);
  if(ImGui::SmallButton(ICON_MS_ADD "##AddSplat"))
  {
    prmScene.sceneToLoadFilename = nvgui::windowOpenFileDialog(m_app->getWindowHandle(), "Add splat file",
                                                               "All Files|*.ply;*.spz;*.sog|PLY Files|*.ply|SPZ files|*.spz|SOG files|*.sog");
    prmScene.addSceneToExisting = true;  // Add to existing instead of replacing
  }
  nvgui::tooltip("Add radiance field to scene");
  
  if(node_open)
  {
    // display the radiance fields tree
    for(size_t i = 0; i < m_radianceFields.size(); ++i)
    {
      ImGui::PushID(static_cast<int>(i));
      ImGuiTreeNodeFlags item_flags = base_flags | ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
      if(m_selectedAsset == GUI_SPLATSET && m_selectedItemIndex == static_cast<int64_t>(i))
        item_flags |= ImGuiTreeNodeFlags_Selected;

      const auto& field = m_radianceFields[i];
      bool item_open = ImGui::TreeNodeEx((void*)(intptr_t)i, item_flags, 
                                         ICON_MS_SUBDIRECTORY_ARROW_RIGHT "Splat set %zu - %s (%zu splats)",
                                         i, field.displayName.c_str(), field.splatCount);
      if(ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen())
      {
        m_selectedAsset     = GUI_SPLATSET;
        m_selectedItemIndex = static_cast<int64_t>(i);
      }
      
      // Delete button for individual radiance field
      ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - 30);
      if(ImGui::SmallButton(ICON_MS_DELETE))
      {
        m_requestDeleteRadianceField = true;
        m_radianceFieldToDelete = i;
      }
      nvgui::tooltip("Remove radiance field (clears all)");
      ImGui::PopID();
    }

    ImGui::TreePop();
  }
}

void VkViewerUI::guiDrawObjectTree()
{

  namespace PE = nvgui::PropertyEditor;

  const ImGuiTreeNodeFlags base_flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick;

  ImGuiTreeNodeFlags node_flags = base_flags;

  if(m_selectedAsset == GUI_MESH && m_selectedItemIndex == -1)
    node_flags |= ImGuiTreeNodeFlags_Selected;

  if(m_objListUpdated)
  {
    ImGui::SetNextItemOpen(true);
    m_objListUpdated = false;
  }
  bool node_open =
      ImGui::TreeNodeEx(fmt::format(ICON_MS_DEPLOYED_CODE " Mesh Models ({})", m_meshSetVk.instances.size()).c_str(), node_flags);
  if(ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen())
  {
    m_selectedAsset     = GUI_NONE;
    m_selectedItemIndex = -1;
  }
  ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - 30);
  if(ImGui::SmallButton(ICON_MS_FILE_OPEN))
  {
    prmScene.meshToImportFilename = nvgui::windowOpenFileDialog(m_app->getWindowHandle(), "Load mesh file", "Mesh files|*.obj;*.glb;*.gltf|OBJ|*.obj|GLTF|*.glb;*.gltf");
  }
  // Handle the request form file open or from drag and drop
  if(!prmScene.meshToImportFilename.empty())
  {
    bool valid = true;

    const auto name               = prmScene.meshToImportFilename;
    prmScene.meshToImportFilename = "";  // reset the request
    if(!name.empty())
    {
      // synchronous load
      valid = m_meshSetVk.loadModel(name);
    }

    if(!valid)
    {
      ImGui::OpenPopup("Obj Loading");
    }
    else
    {
      //
      m_requestUpdateMeshData = true;
      m_requestUpdateShaders  = true;
      //
      m_selectedAsset     = GUI_MESH;
      m_selectedItemIndex = m_meshSetVk.instances.size() - 1;
      //
      m_objListUpdated = true;  // so that next loop will force the Object open if selected
      
      // Auto-fit camera to newly loaded mesh
      if(!m_meshSetVk.meshes.empty() && cameraManip)
      {
        const auto& lastMesh = m_meshSetVk.meshes.back();
        cameraManip->fit(lastMesh.bboxMin, lastMesh.bboxMax, true, false, 1.0f);
      }
    }

    // definition of the obj error popup
    // Always center this window when appearing
    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    if(ImGui::BeginPopupModal("Obj Loading", NULL, ImGuiWindowFlags_AlwaysAutoResize))
    {
      ImGui::Text("Error: invalid obj file");
      if(ImGui::Button("Ok", ImVec2(120, 0)))
      {
        ImGui::CloseCurrentPopup();
      }
      ImGui::EndPopup();
    }
  }
  if(node_open)
  {
    // display the objects tree
    for(int i = 0; i < m_meshSetVk.instances.size(); ++i)
    {
      ImGui::PushID(i);
      int                instanceIndex = i;
      int                objectIndex   = m_meshSetVk.instances[instanceIndex].objIndex;
      ImGuiTreeNodeFlags node_flags    = base_flags | ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
      if(m_selectedAsset == GUI_MESH && m_selectedItemIndex == instanceIndex)
        node_flags |= ImGuiTreeNodeFlags_Selected;

      bool node_open = ImGui::TreeNodeEx((void*)(intptr_t)i, node_flags, ICON_MS_SUBDIRECTORY_ARROW_RIGHT "Model %d - %s",
                                         i, m_meshSetVk.meshes[objectIndex].name.c_str());
      if(ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen())
      {
        m_selectedAsset     = GUI_MESH;
        m_selectedItemIndex = instanceIndex;
      }
      ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - 30);
      if(ImGui::SmallButton(ICON_MS_DELETE))
      {
        m_requestDeleteSelectedMesh = true;
        m_selectedItemIndex         = instanceIndex;
        m_objListUpdated            = true;  // so that next loop will force the Object node open if selected
      }
      ImGui::PopID();
    }
    ImGui::TreePop();
  }
}

void VkViewerUI::guiDrawDepthStreamTree()
{
  const ImGuiTreeNodeFlags base_flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick;
  const ImGuiTreeNodeFlags leaf_flags = base_flags | ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;

  ImGuiTreeNodeFlags node_flags = base_flags;
  if(m_selectedAsset == GUI_DEPTH_STREAM && m_selectedItemIndex == -1)
    node_flags |= ImGuiTreeNodeFlags_Selected;

  // Show different icon and label based on mode
  const char* treeLabel = m_videoDepthPlaybackMode ? ICON_MS_MOVIE " Video+Depth" : ICON_MS_STREAM " Depth Streaming";
  
  bool node_open = ImGui::TreeNodeEx(treeLabel, node_flags);
  if(ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen())
  {
    m_selectedAsset     = GUI_DEPTH_STREAM;
    m_selectedItemIndex = -1;
  }
  if(node_open)
  {
    // Show video layer if in video+depth mode
#ifdef WITH_VIDEO_DECODER
    if(m_videoDepthPlaybackMode && m_videoDepthManager)
    {
      ImGuiTreeNodeFlags videoFlags = leaf_flags;
      if(m_selectedAsset == GUI_DEPTH_STREAM && m_selectedItemIndex == 1)
        videoFlags |= ImGuiTreeNodeFlags_Selected;
      
      const auto& metadata = m_videoDepthManager->getMetadata();
      std::string videoLabel = fmt::format(ICON_MS_VIDEOCAM " Video ({}x{})", metadata.sourceWidth, metadata.sourceHeight);
      ImGui::TreeNodeEx(videoLabel.c_str(), videoFlags);
      if(ImGui::IsItemClicked())
      {
        m_selectedAsset = GUI_DEPTH_STREAM;
        m_selectedItemIndex = 1;
      }
    }
#endif

    if(m_depthClient)
    {
      ImGuiTreeNodeFlags depthFlags = leaf_flags;
      if(m_selectedAsset == GUI_DEPTH_STREAM && m_selectedItemIndex == 2)
        depthFlags |= ImGuiTreeNodeFlags_Selected;
      
      ImGui::TreeNodeEx(ICON_MS_CLOUD_DOWNLOAD " Depth Stream (Live)", depthFlags);
      if(ImGui::IsItemClicked())
      {
        m_selectedAsset = GUI_DEPTH_STREAM;
        m_selectedItemIndex = 2;
      }
    }

    ImGui::TreePop();
  }
}

void VkViewerUI::guiDrawPropertiesWindow()
{
  if(ImGui::Begin("Properties"))
  {
    switch(m_selectedAsset)
    {
      case GUI_RENDERER:
        if(ImGui::CollapsingHeader("Renderer", ImGuiTreeNodeFlags_DefaultOpen))
        {
          guiDrawRendererProperties();
        }
        break;
      case GUI_SPLATSET:
        guiDrawSplatSetProperties();
        break;
      case GUI_MESH:
        m_selectedItemIndex = std::clamp<int64_t>(m_selectedItemIndex, -1, m_meshSetVk.instances.size() - 1);
        if(m_selectedItemIndex >= 0)
        {
          if(ImGui::CollapsingHeader("Transform", ImGuiTreeNodeFlags_DefaultOpen))
          {
            guiDrawMeshTransformProperties();
          }
          if(ImGui::CollapsingHeader("Materials", ImGuiTreeNodeFlags_DefaultOpen))
          {
            guiDrawMeshMaterialProperties();
          }
        }
        break;
      case GUI_CAMERA:
        m_selectedItemIndex = std::clamp<int64_t>(m_selectedItemIndex, -1, m_cameraSet.size() - 1);
        //if(ImGui::CollapsingHeader("Camera Intrinsics", ImGuiTreeNodeFlags_DefaultOpen))
        {
          guiDrawCameraProperties();
        }
        if(m_selectedItemIndex == -1)
        {
          if(ImGui::CollapsingHeader("Navigation", ImGuiTreeNodeFlags_DefaultOpen))
          {
            guiDrawNavigationProperties();
          }
        }
        break;
      case GUI_LIGHT:
        m_selectedItemIndex = std::clamp<int64_t>(m_selectedItemIndex, -1, m_lightSet.size() - 1);
        if(m_selectedItemIndex >= 0)
        {
          if(ImGui::CollapsingHeader("Light", ImGuiTreeNodeFlags_DefaultOpen))
          {
            guiDrawLightProperties();
          }
        }
        break;
      case GUI_DEPTH_STREAM:
        guiDrawDepthStreamProperties();
        break;
      default:
        // display nothing
        break;
    };
  }
  ImGui::End();
}

// guiDrawRendererProperties moved to vk_viewer_ui_renderer.cpp

void VkViewerUI::guiDrawSplatSetProperties()
{
  namespace PE = nvgui::PropertyEditor;

  if(ImGui::CollapsingHeader("Model Transform", ImGuiTreeNodeFlags_DefaultOpen))
  {
    PE::begin("##Transform");
    if(guiGetTransform(m_splatSetVk.scale, m_splatSetVk.rotation, m_splatSetVk.translation, m_splatSetVk.transform,
                       m_splatSetVk.transformInverse, false))
    {
      // delay update of Acceleration Structures if not using ray tracing
      m_requestDelayedUpdateSplatAs = true;
    }
    PE::end();
  }
  if(ImGui::CollapsingHeader("Splat Set Format in VRAM", ImGuiTreeNodeFlags_DefaultOpen))
  {
    if(PE::begin("##VRAM format"))
    {
      if(PE::entry(
             "Default settings", [&] { return ImGui::Button("Reset"); }, "resets to default settings"))
      {
        resetDataParameters();
        m_requestUpdateSplatData = true;
      }
      if(PE::entry(
             "Storage", [&] { return m_ui.enumCombobox(GUI_STORAGE, "##ID", &prmData.dataStorage); },
             "Selects between Data Buffers and Textures for storing model attributes, including:\n"
             "Position, Color and Opacity, Covariance Matrix\n"
             "and Spherical Harmonics (SH) Coefficients (for degrees higher than 0)"))
      {
        m_requestUpdateSplatData = true;
      }
      ImGui::BeginDisabled(m_splatSet.maxShDegree() == 0);
      if(PE::entry(
             "SH format", [&]() { return m_ui.enumCombobox(GUI_SH_FORMAT, "##ID", &prmData.shFormat); },
             "Selects storage format for SH coefficient, balancing precision and memory usage"))
      {
        m_requestUpdateSplatData = true;
      }
      ImGui::EndDisabled();
      PE::end();
    }
  }
  if(ImGui::CollapsingHeader("RTX acceleration structures", ImGuiTreeNodeFlags_DefaultOpen))
  {
    if(PE::begin("##VRAM format RTX"))
    {
      if(PE::entry(
             "Default settings", [&] { return ImGui::Button("Reset"); }, "resets to default settings"))
      {
        resetRtxDataParameters();
        m_requestUpdateSplatAs = true;
      }
      if(PE::Checkbox("Use AABBs", &prmRtxData.useAABBs,
                      "If on, uses AABBs for splats in BLAS instead of ICOSAHEDRON meshes."
                      "In this case the renderer will use the collision shader instead of "
                      "the ray/triangle intersection specialized hardware."))
        m_requestUpdateSplatAs = true;

      // We do not allow useAABBs without instances (prevent bvh with very bad properties leading to very low frame rate and device lost error)
      if(prmRtxData.useAABBs)
        prmRtxData.useTlasInstances = true;

      ImGui::BeginDisabled(prmRtxData.useAABBs);
      if(PE::Checkbox("Use TLAS instances", &prmRtxData.useTlasInstances,
                      "If on, uses one TLAS entry per splat and a small BLAS "
                      "with a unit Icosahedron. \nOtherwise use a single TLAS "
                      "entry and a huge BLAS containing all the transformed Icosahedrons."))
        m_requestUpdateSplatAs = true;
      ImGui::EndDisabled();

      if(PE::Checkbox("BLAS Compaction", &prmRtxData.compressBlas, "Bottom Level Acceleration structure compression."))
        m_requestUpdateSplatAs = true;

      if(m_splatSet.size() != 0 && !m_splatSetVk.rtxValid)
      {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.0f, 0.0f, 1.0f));
        PE::Text("Error", "RTX allocation failed");
        ImGui::PopStyleColor();
      }

      PE::end();
    }
  }
  if(ImGui::CollapsingHeader("Loading Options", ImGuiTreeNodeFlags_DefaultOpen))
  {
    if(PE::begin("##Loading Options"))
    {
      if(PE::entry(
             "Color Space",
             [&] {
               bool changed = false;
               changed |= ImGui::RadioButton("None", &prmScene.colorSpaceConversion, 0);
               ImGui::SameLine();
               changed |= ImGui::RadioButton("sRGB to Linear", &prmScene.colorSpaceConversion, 1);
               return changed;
             },
              "Select color space conversion for the next loaded PLY file.\n"
              "sRGB to Linear is required for ML-SHARP compatibility-exported files."))
       {
       }

      PE::Checkbox("Remove black splats", &prmScene.removeBlackSplats,
                    "If on, splats with (almost) zero color will be discarded during loading.\n"
                    "This can help reduce point count and improve performance without visible quality loss.");

      if(PE::entry(
             "Apply to scene", [&] { return ImGui::Button("Reload all files"); },
             "Reloads all radiance fields with the current loading options (Color Space and Black Splat Removal)"))
      {
        // Store current files
        m_reloadQueue.clear();
        for(const auto& field : m_radianceFields)
        {
          m_reloadQueue.push_back(field.filename);
        }

        if(!m_reloadQueue.empty())
        {
          // Clear scene
          vkDeviceWaitIdle(m_device);
          deinitAll();
          m_radianceFields.clear();
          m_splatSet.clear();

          // Start reloading first file
          prmScene.sceneToLoadFilename = m_reloadQueue.front();
          m_reloadQueue.erase(m_reloadQueue.begin());
          prmScene.addSceneToExisting = false;
          m_isReloading               = true;
          // Note: the actual load will be triggered by prmScene.sceneToLoadFilename in onUIRender
        }
      }

      PE::end();
    }
  }
}

void VkViewerUI::guiDrawMeshTransformProperties()
{
  namespace PE = nvgui::PropertyEditor;

  Instance& inst = m_meshSetVk.instances[m_selectedItemIndex];
  PE::begin("##Transform");
  if(guiGetTransform(inst.scale, inst.rotation, inst.translation, inst.transform, inst.transformInverse, false))
  {
    m_meshSetVk.rtxUpdateTopLevelAccelerationStructure();
  }
  PE::end();
}

void VkViewerUI::guiDrawMeshMaterialProperties()
{
  namespace PE = nvgui::PropertyEditor;

  const auto objIndex           = m_meshSetVk.instances[m_selectedItemIndex].objIndex;
  auto&      materials          = m_meshSetVk.meshes[objIndex].materials;
  bool       needMaterialUpdate = false;

  for(auto i = 0; i < materials.size(); ++i)
  {
    PE::begin("##Material");
    auto& material = materials[i];
    ImGui::PushID(i);
    PE::Text("Name", m_meshSetVk.meshes[objIndex].matNames[i]);
    needMaterialUpdate |= PE::entry(
        "Model", [&]() { return m_ui.enumCombobox(GUI_ILLUM_MODEL, "##ID", &material.illum); }, "TODO");
    needMaterialUpdate |= PE::ColorEdit3("ambient", glm::value_ptr(material.ambient));
    needMaterialUpdate |= PE::ColorEdit3("diffuse", glm::value_ptr(material.diffuse));
    needMaterialUpdate |= PE::ColorEdit3("specular", glm::value_ptr(material.specular));
    needMaterialUpdate |= PE::ColorEdit3("transmittance", glm::value_ptr(material.transmittance));
    // TODO implement in Shader
    //needMaterialUpdate |= PE::ColorEdit3("emission", glm::value_ptr(material.emission));
    needMaterialUpdate |= PE::SliderFloat("shininess", &material.shininess, 0.0f, 2000.0f);
    needMaterialUpdate |= PE::SliderFloat("ior", &material.ior, 1.0f, 3.0f);
    //needMaterialUpdate |= PE::DragFloat("dissolve", &material.dissolve);
    ImGui::PopID();
    PE::end();
  }
  if(needMaterialUpdate)
  {
    m_meshSetVk.updateObjMaterialsBuffer(objIndex);
  }
}

void VkViewerUI::guiDrawCameraProperties()
{
  namespace PE = nvgui::PropertyEditor;

  Camera camera = m_cameraSet.getCamera();
  if(m_selectedItemIndex > -1)  // we show a preset - "read only"
  {
    camera = m_cameraSet.getPreset(m_selectedItemIndex);
    if(m_selectedItemIndex > 0)
    {
      ImGui::Text("To modify a preset.");
      ImGui::Text("  1. Load the preset");
      ImGui::Text("  2. Modify the current camera");
      ImGui::Text("  3. Overwrite the preset with active camera");
    }
    else
      ImGui::Text("Default Preset cannot be modified.");
  }

  bool changed = false;

  if(ImGui::CollapsingHeader("Camera Intrinsics", ImGuiTreeNodeFlags_DefaultOpen))
  {
    ImGui::BeginDisabled(m_selectedItemIndex != -1 || cameraManip->isAnimated());
    if(PE::begin())
    {
      if(PE::entry(
             "Camera type", [&] { return m_ui.enumCombobox(GUI_CAMERA_TYPE, "##ID", &camera.model); },
             "Fisheye type may not be supported by all the Pipelines.\n"
             "The Camera type is not stored per camera for the time beeing."))
      {
        m_requestUpdateShaders = true;
        changed                = true;
      }

      PE::InputFloat2("Clip planes", glm::value_ptr(camera.clip));
      changed |= ImGui::IsItemDeactivatedAfterEdit();

      if(PE::SliderFloat("FOV", &camera.fov, 1.F, 179.F, "%.1f deg", ImGuiSliderFlags_Logarithmic, "Field of view in degrees"))
      {
        changed = true;
      }

      ImGui::BeginDisabled(prmSelectedPipeline != PIPELINE_RTX && prmSelectedPipeline != PIPELINE_HYBRID_3DGUT
                           && prmSelectedPipeline != PIPELINE_MESH_3DGUT);

      if(PE::Checkbox("Depth of Field", &camera.dofEnabled,
                      "Activates Depth of Field effect (DoF). Only works with 3DGRT, 3DGUT and hybrid 3DGUT/3GDRT.\n"
                      "Activating \"Temporal sampling\" in addition to DoF leads to better visual results."))
      {
        m_requestUpdateShaders = true;
        changed                = true;
      }
      ImGui::BeginDisabled(!camera.dofEnabled);
      if(PE::DragFloat("Focus distance", &camera.focusDist, 0.1F, 0.1F, 15.0F, "%.3f"))
      {
        resetFrameCounter();
        changed = true;
      }
      if(PE::SliderFloat("Aperture", &camera.aperture, 0.0F, 0.01F, "%.6f"))
      {
        resetFrameCounter();
        changed = true;
      }
      ImGui::EndDisabled();  // DoF

      ImGui::EndDisabled();  // Modifiable
    }
    PE::end();
    ImGui::EndDisabled();
  }
  if(ImGui::CollapsingHeader("Camera Extrinsics", ImGuiTreeNodeFlags_DefaultOpen))
  {
    ImGui::BeginDisabled(m_selectedItemIndex != -1 || cameraManip->isAnimated());
    if(PE::begin())
    {

      PE::InputFloat3("Eye", &camera.eye.x, "%.5f", 0, "Position of the Camera");
      changed |= ImGui::IsItemDeactivatedAfterEdit();
      PE::InputFloat3("Center", &camera.ctr.x, "%.5f", 0, "Center of camera interest");
      changed |= ImGui::IsItemDeactivatedAfterEdit();
      PE::InputFloat3("Up", &camera.up.x, "%.5f", 0, "Up vector interest");
      changed |= ImGui::IsItemDeactivatedAfterEdit();

      PE::end();
    }
    ImGui::EndDisabled();
  }

  // if changed it is necessarly the active camera
  if(changed)
    m_cameraSet.setCamera(camera);
}

void VkViewerUI::guiDrawNavigationProperties()
{

  namespace PE = nvgui::PropertyEditor;

  bool changed = false;

  ImGui::BeginDisabled(cameraManip->isAnimated());

  // Navigation Mode
  if(PE::begin())
  {
    auto mode     = cameraManip->getMode();
    auto speed    = cameraManip->getSpeed();
    auto duration = static_cast<float>(cameraManip->getAnimationDuration());

    changed |= PE::entry(
        "Navigation and Animation",
        [&] {
          int rmode = static_cast<int>(mode);
          changed |= ImGui::RadioButton("Examine", &rmode, nvutils::CameraManipulator::Examine);
          nvgui::tooltip("The camera orbit around a point of interest");
          changed |= ImGui::RadioButton("Fly", &rmode, nvutils::CameraManipulator::Fly);
          nvgui::tooltip("The camera is free and move toward the looking direction");
          changed |= ImGui::RadioButton("Walk", &rmode, nvutils::CameraManipulator::Walk);
          nvgui::tooltip("The camera is free but stay on a plane");
          cameraManip->setMode(static_cast<nvutils::CameraManipulator::Modes>(rmode));
          return changed;
        },
        "Camera Navigation Mode");

    changed |= PE::SliderFloat("Speed", &speed, 0.01F, 10.0F, "%.3f", 0, "Changing the default movement speed");
    changed |= PE::SliderFloat("Transition", &duration, 0.0F, 2.0F, "%.3f", 0,
                               "Nb seconds to move to new position when loading a camera preset");

    cameraManip->setSpeed(speed);
    cameraManip->setAnimationDuration(duration);

    PE::end();
  }

  ImGui::EndDisabled();
}

void VkViewerUI::guiDrawLightProperties()
{
  namespace PE = nvgui::PropertyEditor;

  bool needUpdate = false;

  auto& light = m_lightSet.getLight(m_selectedItemIndex);
  ImGui::Text("Light sources only affect meshes");
  ImGui::Text("Point lights have quadratic attenuation");
  PE::begin("##Light");
  if(PE::entry("Type", [&]() { return m_ui.enumCombobox(GUI_LIGHT_TYPE, "##ID", &light.type); }, "Type of light."))
  {
    needUpdate = true;
  }
  needUpdate |= PE::DragFloat3("Position", glm::value_ptr(light.position));
  needUpdate |= PE::DragFloat("Intensity", &light.intensity);
  PE::end();

  m_requestUpdateLightsBuffer |= needUpdate;
}

bool VkViewerUI::guiGetTransform(glm::vec3& scale,
                                          glm::vec3& rotation,
                                          glm::vec3& translation,
                                          glm::mat4& transform,
                                          glm::mat4& transformInv,
                                          bool       disabled /*=false*/)
{
  namespace PE = nvgui::PropertyEditor;

  bool updated = false;
  ImGui::BeginDisabled(disabled);
  updated |= PE::DragFloat3("Translate", glm::value_ptr(translation), 0.05f);
  updated |= PE::DragFloat3("Rotate", glm::value_ptr(rotation), 0.5f);
  updated |= PE::DragFloat3("Scale", glm::value_ptr(scale), 0.01f);
  ImGui::EndDisabled();

  if(updated)
  {
    computeTransform(scale, rotation, translation, transform, transformInv);
  }

  return updated;
}

void VkViewerUI::guiDrawRendererStatisticsWindow()
{
  if(ImGui::Begin("Rendering Statistics"))
  {
    const int32_t totalSplatCount = (uint32_t)m_splatSet.size();
    const int32_t rasterSplatCount =
        (prmRaster.sortingMethod != SORTING_GPU_SYNC_RADIX) ? totalSplatCount : m_indirectReadback.instanceCount;
    const uint32_t wgCount =
        (prmSelectedPipeline == PIPELINE_MESH || prmSelectedPipeline == PIPELINE_MESH_3DGUT) ?
            ((prmRaster.sortingMethod == SORTING_GPU_SYNC_RADIX) ?
                 m_indirectReadback.groupCountX :
                 (prmFrame.splatCount + prmRaster.meshShaderWorkgroupSize - 1) / prmRaster.meshShaderWorkgroupSize) :
            0;

    if(ImGui::BeginTable("Stats", 3, ImGuiTableFlags_BordersOuter))
    {
      ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthFixed, 230.0f);
      ImGui::TableSetupColumn("Size short", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Size Fill", ImGuiTableColumnFlags_WidthStretch);
      // ImGui::TableHeadersRow();
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("Total splats");
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatSize(totalSplatCount).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("%d", totalSplatCount);
      ImGui::TableNextRow();
      ImGui::BeginDisabled(prmSelectedPipeline == PIPELINE_RTX);
      ImGui::TableNextColumn();
      ImGui::Text("Sorted splats");
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatSize(rasterSplatCount).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("%d", rasterSplatCount);
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("Mesh shader work groups");
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatSize(wgCount).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("%d", wgCount);
      ImGui::TableNextRow();
      ImGui::EndDisabled();
      ImGui::EndTable();
    }
  }
  ImGui::End();
}


void VkViewerUI::guiDrawMemoryStatisticsWindow()
{
  ImGuiTableFlags itemFlags   = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
  ImGuiTableFlags totalFlags  = ImGuiTreeNodeFlags_DefaultOpen;

  if(ImGui::Begin("Memory Statistics"))
  {
    if(ImGui::BeginTable("Scene stats", 4, ImGuiTableFlags_RowBg))
    {
      // to draw horizontal line for specific rows.
      ImDrawList* draw_list = ImGui::GetWindowDrawList();

      ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Host used", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Device used", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Device allocated", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableHeadersRow();
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      bool open = ImGui::TreeNodeEx("Model data", totalFlags);
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMemorySize(m_splatSetVk.memoryStats.srcAll).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMemorySize(m_splatSetVk.memoryStats.odevAll).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMemorySize(m_splatSetVk.memoryStats.devAll).c_str());
      if(open)
      {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Separator();
        ImGui::TreeNodeEx("Centers", itemFlags);
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_splatSetVk.memoryStats.srcCenters).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_splatSetVk.memoryStats.odevCenters).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_splatSetVk.memoryStats.devCenters).c_str());
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TreeNodeEx("Covariances", itemFlags);
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_splatSetVk.memoryStats.srcCov).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_splatSetVk.memoryStats.odevCov).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_splatSetVk.memoryStats.devCov).c_str());
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TreeNodeEx("SH degree 0", itemFlags);
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_splatSetVk.memoryStats.srcSh0).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_splatSetVk.memoryStats.odevSh0).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_splatSetVk.memoryStats.devSh0).c_str());
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TreeNodeEx("SH degree 1,2,3", itemFlags);
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_splatSetVk.memoryStats.srcShOther).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_splatSetVk.memoryStats.odevShOther).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_splatSetVk.memoryStats.devShOther).c_str());
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TreeNodeEx("SH total", itemFlags);
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_splatSetVk.memoryStats.srcShAll).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_splatSetVk.memoryStats.odevShAll).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_splatSetVk.memoryStats.devShAll).c_str());
        // end if(open)
        ImGui::TreePop();
      }
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      open = ImGui::TreeNodeEx("Rasterization", totalFlags);
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMemorySize(m_renderMemoryStats.rasterHostTotal).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMemorySize(m_renderMemoryStats.rasterDeviceUsedTotal).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMemorySize(m_renderMemoryStats.rasterDeviceAllocTotal).c_str());
      if(open)
      {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Separator();
        ImGui::TreeNodeEx("UBO frame info", itemFlags);
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_renderMemoryStats.usedUboFrameInfo).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_renderMemoryStats.usedUboFrameInfo).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_renderMemoryStats.usedUboFrameInfo).c_str());
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TreeNodeEx("Indirect params", itemFlags);
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(0).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_renderMemoryStats.usedIndirect).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_renderMemoryStats.usedIndirect).c_str());
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TreeNodeEx("Distances", itemFlags);
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_renderMemoryStats.hostAllocDistances).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_renderMemoryStats.usedDistances).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_renderMemoryStats.allocDistances).c_str());
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TreeNodeEx("Indices", itemFlags);
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_renderMemoryStats.hostAllocIndices).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_renderMemoryStats.usedIndices).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_renderMemoryStats.allocIndices).c_str());
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TreeNodeEx("GPU sort", itemFlags);
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(0).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(prmRaster.sortingMethod != SORTING_GPU_SYNC_RADIX ? 0 : m_renderMemoryStats.allocVdrxInternal)
                              .c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(prmRaster.sortingMethod != SORTING_GPU_SYNC_RADIX ? 0 : m_renderMemoryStats.allocVdrxInternal)
                              .c_str());
        // end if(open)
        ImGui::TreePop();
      }
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      open = ImGui::TreeNodeEx("Ray tracing", totalFlags);
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMemorySize(m_renderMemoryStats.rtxHostTotal).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMemorySize(m_renderMemoryStats.rtxDeviceUsedTotal).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMemorySize(m_renderMemoryStats.rtxDeviceAllocTotal).c_str());
      if(open)
      {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Separator();
        ImGui::TreeNodeEx("TLAS", itemFlags);
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(0).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_renderMemoryStats.rtxUsedTlas).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_renderMemoryStats.rtxUsedTlas).c_str());
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TreeNodeEx("BLAS", itemFlags);
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(0).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_renderMemoryStats.rtxUsedBlas).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_renderMemoryStats.rtxUsedBlas).c_str());
        // end if(open)
        ImGui::TreePop();
      }
      ImGui::EndTable();
    }
    ImGui::Separator();
    if(ImGui::BeginTable("Total", 4, ImGuiTableFlags_None))
    {
      ImGui::TableSetupColumn("Rendering", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Host used", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Device used", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Device allocated", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableNextColumn();
      ImGui::Text("Total");
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMemorySize(m_splatSetVk.memoryStats.srcAll + m_renderMemoryStats.hostTotal).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMemorySize(m_splatSetVk.memoryStats.odevAll + m_renderMemoryStats.deviceUsedTotal).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMemorySize(m_splatSetVk.memoryStats.devAll + m_renderMemoryStats.deviceAllocTotal).c_str());
      ImGui::EndTable();
    }
  }
  ImGui::End();
}

void VkViewerUI::guiDrawFooterBar()
{
  //
  //ImGuiViewportP* viewport = (ImGuiViewportP*)(void*)ImGui::GetMainViewport();
  ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_MenuBar;
  float height = ImGui::GetFrameHeight();

  if(ImGui::BeginViewportSideBar("##MainStatusBar", NULL, ImGuiDir_Down, height, window_flags))
  {
    if(ImGui::BeginMenuBar())
    {
      ImGui::Text("Mouse ");
      ImGui::Text("%s", fmt::format("{} {}", prmFrame.cursor.x, prmFrame.cursor.y).c_str());
      ImGui::Text(" | Splat Id ");
      ImGui::Text("%s", std::to_string(m_indirectReadback.particleID).c_str());
      ImGui::Text(" | Splat Dist ");
      ImGui::Text("%s", std::to_string(m_indirectReadback.particleDist).c_str());

      /* DEBUG Feedback 
      ImGui::Text(" %s", "debug: ");
      ImGui::Text(" %s", std::to_string(m_indirectReadback.val1).c_str());
      ImGui::Text(" %s", std::to_string(m_indirectReadback.val2).c_str());
      ImGui::Text(" %s  ", std::to_string(m_indirectReadback.val3).c_str());
      ImGui::Text(" %s", std::to_string(m_indirectReadback.val4).c_str());
      ImGui::Text(" %s", std::to_string(m_indirectReadback.val5).c_str());
      ImGui::Text(" %s  ", std::to_string(m_indirectReadback.val6).c_str());
      ImGui::Text(" %s", std::to_string(m_indirectReadback.val7).c_str());
      */

      // temporal sampling progress bar
      {
        float       progress = 0.0;
        std::string buf      = "1/1";
        if(prmRtx.temporalSampling)
        {
          progress = (float)prmFrame.frameSampleId / (std::max(1, prmFrame.frameSampleMax));
          buf      = fmt::format("{}/{}", prmFrame.frameSampleId, prmFrame.frameSampleMax);
        }
        ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - 255);
        ImGui::Text("%s", "SPP");
        nvgui::tooltip("Samples Per Pixel");
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(0.4f, 0.7f, 0.0f, 1.0f));  // Green of course :-)
        ImGui::ProgressBar(progress, ImVec2(200.f, 0.f), buf.c_str());
        ImGui::PopStyleColor();
      }

      ImGui::EndMenuBar();
    }
    ImGui::End();
  }
}

void VkViewerUI::guiAddToRecentFiles(std::filesystem::path filePath, int historySize)
{
  // first check if filePath is absolute
  if(filePath.is_relative())
  {
    filePath = std::filesystem::absolute(filePath);
  }
  //
  auto it = std::find(m_recentFiles.begin(), m_recentFiles.end(), filePath);
  if(it != m_recentFiles.end())
  {
    m_recentFiles.erase(it);
  }
  m_recentFiles.insert(m_recentFiles.begin(), filePath);
  if(m_recentFiles.size() > historySize)
  {
    m_recentFiles.pop_back();
  }
}

void VkViewerUI::guiAddToRecentProjects(std::filesystem::path filePath, int historySize)
{
  // first check if filePath is absolute
  if(filePath.is_relative())
  {
    filePath = std::filesystem::absolute(filePath);
  }
  //
  auto it = std::find(m_recentProjects.begin(), m_recentProjects.end(), filePath);
  if(it != m_recentProjects.end())
  {
    m_recentProjects.erase(it);
  }
  m_recentProjects.insert(m_recentProjects.begin(), filePath);
  if(m_recentProjects.size() > historySize)
  {
    m_recentProjects.pop_back();
  }
}

void VkViewerUI::guiRegisterIniFileHandlers()
{
  // mandatory to work, see ImGui::DockContextInitialize as an example
  auto readOpen = [](ImGuiContext*, ImGuiSettingsHandler* handler, const char* name) -> void* {
    if(strcmp(name, "Data") != 0)
      return NULL;
    // Make sure we clear out our current recent vectors so we don't just keep adding to the list every time we load
    // This is if the .ini file is loaded twice, which happens in nvpro_core2
    auto* ui = static_cast<VkViewerUI*>(handler->UserData);
    if(strcmp(handler->TypeName, "RecentFiles") == 0)
    {
      ui->m_recentFiles.clear();
    }
    else if(strcmp(handler->TypeName, "RecentProjects") == 0)
    {
      ui->m_recentProjects.clear();
    }
    return (void*)1;
  };

  {
    // Save settings handler, not using capture so can be used as a function pointer
    auto saveRecentFilesToIni = [](ImGuiContext* ctx, ImGuiSettingsHandler* handler, ImGuiTextBuffer* buf) {
      auto* self = static_cast<VkViewerUI*>(handler->UserData);
      buf->appendf("[%s][Data]\n", handler->TypeName);
      for(const auto& file : self->m_recentFiles)
      {
        buf->appendf("File=%s\n", file.string().c_str());
      }
      buf->append("\n");
    };

    // Load settings handler, not using capture so can be used as a function pointer
    auto loadRecentFilesFromIni = [](ImGuiContext* ctx, ImGuiSettingsHandler* handler, void* entry, const char* line) {
      auto* self = static_cast<VkViewerUI*>(handler->UserData);
      if(strncmp(line, "File=", 5) == 0)
      {
        const char* filePath = line + 5;
        self->m_recentFiles.push_back(filePath);
      }
    };

    //
    ImGuiSettingsHandler iniHandler;
    iniHandler.TypeName   = "RecentFiles";
    iniHandler.TypeHash   = ImHashStr(iniHandler.TypeName);
    iniHandler.ReadOpenFn = readOpen;
    iniHandler.WriteAllFn = saveRecentFilesToIni;
    iniHandler.ReadLineFn = loadRecentFilesFromIni;
    iniHandler.UserData   = this;  // Pass the current instance to the handler
    ImGui::GetCurrentContext()->SettingsHandlers.push_back(iniHandler);
  }
  {
    // Save settings handler, not using capture so can be used as a function pointer
    auto saveRecentProjectsToIni = [](ImGuiContext* ctx, ImGuiSettingsHandler* handler, ImGuiTextBuffer* buf) {
      auto* self = static_cast<VkViewerUI*>(handler->UserData);
      buf->appendf("[%s][Data]\n", handler->TypeName);
      for(const auto& file : self->m_recentProjects)
      {
        buf->appendf("File=%s\n", file.string().c_str());
      }
      buf->append("\n");
    };

    // Load settings handler, not using capture so can be used as a function pointer
    auto loadRecentProjectsFromIni = [](ImGuiContext* ctx, ImGuiSettingsHandler* handler, void* entry, const char* line) {
      auto* self = static_cast<VkViewerUI*>(handler->UserData);
      if(strncmp(line, "File=", 5) == 0)
      {
        const char* filePath = line + 5;
        self->m_recentProjects.push_back(filePath);
      }
    };

    //
    ImGuiSettingsHandler iniHandler;
    iniHandler.TypeName   = "RecentProjects";
    iniHandler.TypeHash   = ImHashStr(iniHandler.TypeName);
    iniHandler.ReadOpenFn = readOpen;
    iniHandler.WriteAllFn = saveRecentProjectsToIni;
    iniHandler.ReadLineFn = loadRecentProjectsFromIni;
    iniHandler.UserData   = this;  // Pass the current instance to the handler
    ImGui::GetCurrentContext()->SettingsHandlers.push_back(iniHandler);
  }
}

///////////////////////////////////
// Loading and Saving Propjects

namespace fs = std::filesystem;

fs::path getRelativePath(const fs::path& from, const fs::path& to)
{
  fs::path relativePath;

  auto fromIter = from.begin();
  auto toIter   = to.begin();

  // Find common point
  while(fromIter != from.end() && toIter != to.end() && (*fromIter) == (*toIter))
  {
    ++fromIter;
    ++toIter;
  }

  // Add ".." for each remaining part in `from` path
  for(; fromIter != from.end(); ++fromIter)
  {
    relativePath /= "..";
  }

  // Add remaining part of `to` path
  for(; toIter != to.end(); ++toIter)
  {
    relativePath /= *toIter;
  }

  return relativePath;
}

std::filesystem::path makeAbsolutePath(const std::filesystem::path& base, const std::string& relativePath)
{
  return std::filesystem::absolute(base / relativePath);
}

// LOAD macros, loadProjectIfNeeded, and saveProject moved to vk_viewer_ui_project.cpp

void VkViewerUI::dumpSplat(uint32_t splatIdx)
{
  if(!(splatIdx >= 0 && splatIdx < m_splatSet.size()))
  {
    LOGE("Error: no splat to dump\n");
    return;
  }

  std::ofstream out("c:\\Temp\\debug_splat.ply");
  if(!out)
  {
    LOGE("Error: could not open file c:\\Temp\\debug_splat.ply\n");
    return;
  }

  // prints the header
  out << "ply" << std::endl;
  out << "format ascii 1.0" << std::endl;
  out << "element vertex 1" << std::endl;
  out << "property float x" << std::endl;
  out << "property float y" << std::endl;
  out << "property float z" << std::endl;
  out << "property float nx" << std::endl;
  out << "property float ny" << std::endl;
  out << "property float nz" << std::endl;
  for(auto i = 0; i < 3; ++i)
    out << "property float f_dc_" << i << std::endl;
  for(auto i = 0; i < 45; ++i)
    out << "property float f_rest_" << i << std::endl;
  out << "property float opacity" << std::endl;
  for(auto i = 0; i < 3; ++i)
    out << "property float scale_" << i << std::endl;
  for(auto i = 0; i < 4; ++i)
    out << "property float rot_" << i << std::endl;
  out << "end_header" << std::endl;

  // prints the splat values
  for(auto i = 0; i < 3; ++i)
    out << m_splatSet.positions[splatIdx * 3 + i] << " ";
  for(auto i = 0; i < 3; ++i)
    out << "0 ";  // no normals
  for(auto i = 0; i < 3; ++i)
    out << m_splatSet.f_dc[splatIdx * 3 + i] << " ";
  for(auto i = 0; i < 45; ++i)
    out << m_splatSet.f_rest[splatIdx * 45 + i] << " ";
  out << m_splatSet.opacity[splatIdx] << " ";
  for(auto i = 0; i < 3; ++i)
    out << m_splatSet.scale[splatIdx * 3 + i] << " ";
  for(auto i = 0; i < 4; ++i)
    out << m_splatSet.rotation[splatIdx * 4 + i] << " ";

  //
  out.close();

  //
  LOGI("Splat %u was dumped to c:\\Temp\\debug_splat.ply\n", splatIdx);
}

#ifdef WITH_COMFYUI
void VkViewerUI::guiDrawComfyUIWindow()
{
  ImGui::SetNextWindowSize(ImVec2(500, 450), ImGuiCond_FirstUseEver);
  if (!ImGui::Begin("ComfyUI 3D Generator", &m_showComfyUIWindow))
  {
    ImGui::End();
    return;
  }

  auto state = m_comfyClient ? m_comfyClient->getState() : ComfyUIClient::State::Disconnected;

  ImGui::SeparatorText("Connection");

  ImGui::SetNextItemWidth(200);
  ImGui::InputText("Host", m_comfyHost, sizeof(m_comfyHost));
  ImGui::SameLine();
  ImGui::SetNextItemWidth(80);
  ImGui::InputInt("Port", &m_comfyPort);

  bool isConnected = (state == ComfyUIClient::State::Connected || state == ComfyUIClient::State::Running);

  if (!isConnected)
  {
    if (ImGui::Button(ICON_MS_LINK " Connect"))
    {
      if (m_comfyClient)
      {
        m_comfyClient->connect(m_comfyHost, static_cast<uint16_t>(m_comfyPort));
      }
    }
  }
  else
  {
    if (ImGui::Button(ICON_MS_LINK_OFF " Disconnect"))
    {
      if (m_comfyClient)
      {
        m_comfyClient->disconnect();
      }
    }
  }

  ImGui::SameLine();
  switch (state)
  {
    case ComfyUIClient::State::Disconnected:
      ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Disconnected");
      break;
    case ComfyUIClient::State::Connecting:
      ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Connecting...");
      break;
    case ComfyUIClient::State::Connected:
      ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Connected");
      break;
    case ComfyUIClient::State::Running:
      ImGui::TextColored(ImVec4(0.0f, 0.8f, 1.0f, 1.0f), "Running...");
      break;
    case ComfyUIClient::State::Completed:
      ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.5f, 1.0f), "Completed");
      break;
    case ComfyUIClient::State::Error:
      ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "Error");
      break;
  }

  ImGui::SeparatorText("Workflow");

  std::string workflowStr = m_comfyWorkflowPath.string();
  char workflowBuf[512];
  strncpy(workflowBuf, workflowStr.c_str(), sizeof(workflowBuf) - 1);
  workflowBuf[sizeof(workflowBuf) - 1] = '\0';
  
  ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 80);
  if (ImGui::InputText("##workflow", workflowBuf, sizeof(workflowBuf)))
  {
    m_comfyWorkflowPath = workflowBuf;
  }
  ImGui::SameLine();
  if (ImGui::Button(ICON_MS_FOLDER_OPEN " Browse"))
  {
    auto path = nvgui::windowOpenFileDialog(m_app->getWindowHandle(), "Select Workflow", "JSON Files|*.json");
    if (!path.empty())
    {
      m_comfyWorkflowPath = path;
    }
  }

  ImGui::SeparatorText("Prompt");

  ImGui::Text("Positive Prompt:");
  ImGui::InputTextMultiline("##positive", m_comfyPrompt, sizeof(m_comfyPrompt), 
                            ImVec2(ImGui::GetContentRegionAvail().x, 80));

  ImGui::Text("Negative Prompt:");
  ImGui::InputTextMultiline("##negative", m_comfyNegativePrompt, sizeof(m_comfyNegativePrompt),
                            ImVec2(ImGui::GetContentRegionAvail().x, 50));

  ImGui::SeparatorText("Generate");

  bool canGenerate = isConnected && state != ComfyUIClient::State::Running;
  
  if (!canGenerate)
  {
    ImGui::BeginDisabled();
  }

  if (ImGui::Button(ICON_MS_AUTO_AWESOME " Generate 3D Model", ImVec2(ImGui::GetContentRegionAvail().x, 40)))
  {
    if (m_comfyClient && std::filesystem::exists(m_comfyWorkflowPath))
    {
      m_comfyStatusMessage = "Queueing workflow...";
      if (m_comfyClient->queueWorkflow(m_comfyWorkflowPath, m_comfyPrompt, m_comfyNegativePrompt))
      {
        m_comfyStatusMessage = "Workflow queued successfully";
      }
      else
      {
        m_comfyStatusMessage = "Failed: " + m_comfyClient->getLastError();
      }
    }
    else if (!std::filesystem::exists(m_comfyWorkflowPath))
    {
      m_comfyStatusMessage = "Error: Workflow file not found";
    }
  }

  if (!canGenerate)
  {
    ImGui::EndDisabled();
  }

  if (state == ComfyUIClient::State::Running && m_comfyClient)
  {
    int current = m_comfyClient->getProgressCurrent();
    int total = m_comfyClient->getProgressTotal();
    if (total > 0)
    {
      float progress = static_cast<float>(current) / static_cast<float>(total);
      ImGui::ProgressBar(progress, ImVec2(ImGui::GetContentRegionAvail().x, 0), 
                         (std::to_string(current) + "/" + std::to_string(total)).c_str());
    }
    else
    {
      ImGui::ProgressBar(-1.0f * static_cast<float>(ImGui::GetTime()), 
                         ImVec2(ImGui::GetContentRegionAvail().x, 0), "Processing...");
    }
  }

  if (!m_comfyStatusMessage.empty())
  {
    ImGui::TextWrapped("%s", m_comfyStatusMessage.c_str());
  }

  if (state == ComfyUIClient::State::Error && m_comfyClient)
  {
    ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "%s", m_comfyClient->getLastError().c_str());
  }

  ImGui::End();
}

void VkViewerUI::onComfyUIWorkflowComplete(const ComfyUIClient::WorkflowResult& result)
{
  if (result.success && !result.plyPath.empty())
  {
    m_comfyStatusMessage = "Success! Loading: " + result.plyPath;
    LOGI("ComfyUI workflow completed. PLY path: %s\n", result.plyPath.c_str());

    if (std::filesystem::exists(result.plyPath))
    {
      prmScene.sceneToLoadFilename = result.plyPath;
    }
    else
    {
      m_comfyStatusMessage = "Warning: PLY file not found at " + result.plyPath;
    }
  }
  else
  {
    m_comfyStatusMessage = "Failed: " + result.errorMessage;
  }
}
#endif  // WITH_COMFYUI









// guiDrawDepthStreamProperties moved to vk_viewer_ui_depth_stream.cpp

void VkViewerUI::guiDrawPerformancePanel()
{
    if (ImGui::Begin("Performance Telemetry")) {
        auto metrics = m_perfStats.getAllMetrics();
        
        if (ImGui::BeginTable("Metrics", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
            ImGui::TableSetupColumn("Name");
            ImGui::TableSetupColumn("Current");
            ImGui::TableSetupColumn("Avg");
            ImGui::TableSetupColumn("Min/Max");
            ImGui::TableHeadersRow();

            for (const auto& [name, metric] : metrics) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%s", name.c_str());
                
                ImGui::TableNextColumn();
                ImGui::Text("%.2f", metric.current);
                
                ImGui::TableNextColumn();
                ImGui::Text("%.2f", metric.avg);
                
                ImGui::TableNextColumn();
                ImGui::Text("%.2f / %.2f", metric.min, metric.max);
            }
            ImGui::EndTable();
        }
        
        for (const auto& [name, metric] : metrics) {
            if (!metric.historyForPlotting.empty()) {
                std::vector<float> values(metric.historyForPlotting.begin(), metric.historyForPlotting.end());
                ImGui::PlotLines(name.c_str(), values.data(), (int)values.size(), 0, nullptr, FLT_MAX, FLT_MAX, ImVec2(0, 50));
            }
        }

#ifdef WITH_OPENXR
        if (m_xrInitialized && m_xr && m_xr->isPerformanceMetricsSupported()) {
            m_xr->updatePerformanceMetrics();
            const auto& xrMetrics = m_xr->getPerformanceMetrics();
            
            if (xrMetrics.valid) {
                ImGui::Separator();
                ImGui::Text("Quest Performance Metrics (XR_META)");
                
                if (ImGui::BeginTable("XRMetrics", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                    ImGui::TableSetupColumn("Metric");
                    ImGui::TableSetupColumn("Value");
                    ImGui::TableHeadersRow();

                    auto addRow = [](const char* label, const char* fmt, auto value) {
                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::Text("%s", label);
                        ImGui::TableNextColumn();
                        ImGui::Text(fmt, value);
                    };

                    addRow("App CPU Frame", "%.2f ms", xrMetrics.appCpuFrameTimeMs);
                    addRow("App GPU Frame", "%.2f ms", xrMetrics.appGpuFrameTimeMs);
                    addRow("Motion-to-Photon", "%.2f ms", xrMetrics.motionToPhotonLatencyMs);
                    addRow("Compositor CPU", "%.2f ms", xrMetrics.compositorCpuFrameTimeMs);
                    addRow("Compositor GPU", "%.2f ms", xrMetrics.compositorGpuFrameTimeMs);
                    addRow("Dropped Frames", "%u", xrMetrics.droppedFrameCount);
                    addRow("SpaceWarp Mode", "%u", xrMetrics.spacewarpMode);
                    addRow("CPU Util (Avg)", "%.1f %%", xrMetrics.cpuUtilizationAvg);
                    addRow("CPU Util (Worst)", "%.1f %%", xrMetrics.cpuUtilizationWorst);
                    addRow("GPU Util", "%.1f %%", xrMetrics.gpuUtilization);

                    ImGui::EndTable();
                }
            }
        }
#endif
    }
    ImGui::End();
}

#ifdef WITH_OPENXR
// VR/XR functions moved to vk_viewer_ui_xr.cpp
#endif

// namespace vk_viewer (continued)


void VkViewerUI::guiDrawFileDialog()
{
  if(m_showFileDialog)
  {
    ImGui::SetNextWindowSize(ImVec2(500, 440), ImGuiCond_FirstUseEver);
    if(ImGui::Begin("Load Scene from Resources", &m_showFileDialog))
    {
      if(ImGui::BeginListBox("##files", ImVec2(-FLT_MIN, -FLT_MIN)))
      {
        for(size_t i = 0; i < m_fileList.size(); i++)
        {
          const bool is_selected = false;
          std::string filename = std::filesystem::path(m_fileList[i]).filename().string();
          if(ImGui::Selectable(filename.c_str(), is_selected))
          {
            prmScene.sceneToLoadFilename = m_fileList[i];
            prmScene.addSceneToExisting = false;
            m_showFileDialog = false;
          }
        }
        ImGui::EndListBox();
      }
    }
    ImGui::End();
  }
}

void VkViewerUI::guiDrawSupersplatDialog()
{
  if(m_showSupersplatDialog)
  {
    ImGui::SetNextWindowSize(ImVec2(800, 600), ImGuiCond_FirstUseEver);
    if(ImGui::Begin("Supersplat Browser", &m_showSupersplatDialog))
    {
      static char searchBuf[256] = "";
      if (m_supersplatSearch.size() < sizeof(searchBuf)) {
          strncpy(searchBuf, m_supersplatSearch.c_str(), sizeof(searchBuf) - 1);
          searchBuf[sizeof(searchBuf) - 1] = '\0';
      }
      bool triggerSearch = ImGui::InputText("Search", searchBuf, sizeof(searchBuf), ImGuiInputTextFlags_EnterReturnsTrue);
      ImGui::SameLine();
      if(ImGui::Button("Go"))
      {
          triggerSearch = true;
      }
      
      if(triggerSearch)
      {
        m_supersplatSearch = searchBuf;
        if(m_supersplatClient)
        {
          m_supersplatClient->fetchSceneList(m_supersplatSearch, [this](const std::vector<SupersplatClient::Scene>& scenes) {
            std::lock_guard<std::mutex> lock(m_thumbnailMutex);
            m_supersplatScenes = scenes;
            // Fetch thumbnails
            for (const auto& scene : scenes) {
              if (!scene.thumbnailUrl.empty()) {
                  m_supersplatClient->fetchThumbnail(scene.thumbnailUrl, 
                      [this, url=scene.thumbnailUrl](const std::vector<uint8_t>& data, int w, int h, int c) {
                          if (data.empty()) return;
                          std::lock_guard<std::mutex> lock(m_thumbnailMutex);
                          std::vector<uint8_t> rgba = data;
                          if (c == 3) {
                              rgba.resize(w * h * 4);
                              for (int i = w * h - 1; i >= 0; --i) {
                                  rgba[i * 4 + 3] = 255;
                                  rgba[i * 4 + 2] = data[i * 3 + 2];
                                  rgba[i * 4 + 1] = data[i * 3 + 1];
                                  rgba[i * 4 + 0] = data[i * 3 + 0];
                              }
                          }
                          m_pendingThumbnails.push_back({url, rgba, w, h});
                      });
              }
            }
          });
        }
      }
      else
      {
          m_supersplatSearch = searchBuf;
      }

      std::lock_guard<std::mutex> lock(m_thumbnailMutex);
      if(ImGui::BeginTable("Scenes", 4))
      {
        for(const auto& scene : m_supersplatScenes)
        {
          ImGui::TableNextColumn();
          ImGui::PushID(scene.id);
          
          ImTextureID texId = 0;
          if(m_thumbnailDescriptors.count(scene.thumbnailUrl))
            texId = (ImTextureID)m_thumbnailDescriptors[scene.thumbnailUrl];
            
          if(texId && ImGui::ImageButton("##img", texId, ImVec2(150, 100)))
          {
             if(!scene.viewUrl.empty())
             {
                prmScene.sceneToLoadFilename = scene.viewUrl;
                prmScene.addSceneToExisting = false;
                m_showSupersplatDialog = false;
             }
          }
          else if (!texId)
          {
             if (ImGui::Button(scene.title.c_str(), ImVec2(150, 100))) // Placeholder
             {
                 if(!scene.viewUrl.empty())
                 {
                    prmScene.sceneToLoadFilename = scene.viewUrl;
                    prmScene.addSceneToExisting = false;
                    m_showSupersplatDialog = false;
                 }
             }
          }
          
          ImGui::TextWrapped("%s", scene.title.c_str());
          if(scene.size > 0)
          {
            ImGui::TextDisabled("%s", formatMemorySize(scene.size).c_str());
          }
          ImGui::PopID();
        }
        ImGui::EndTable();
      }
    }
    ImGui::End();
  }
}

void VkViewerUI::createTextureFromRGBA(const std::vector<uint8_t>& data, int width, int height, nvvk::Image& texture, VkImageView& view)
{
    VkCommandBuffer cmd = m_app->createTempCmdBuffer();
    
    VkImageCreateInfo info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_R8G8B8A8_UNORM;
    info.extent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    
    m_alloc.createImage(texture, info);
    
    VkImageViewCreateInfo viewInfo = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image = texture.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;
    
    vkCreateImageView(m_device, &viewInfo, nullptr, &view);
    
    // Transition to TRANSFER_DST
    VkImageMemoryBarrier barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.image = texture.image;
    barrier.subresourceRange = viewInfo.subresourceRange;
    
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 
        0, 0, nullptr, 0, nullptr, 1, &barrier);
        
    // Upload data using staging buffer
    nvvk::Buffer staging;
    m_alloc.createBuffer(staging, data.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    void* mappedData = nullptr;
    vmaMapMemory(m_alloc, staging.allocation, &mappedData);
    memcpy(mappedData, data.data(), data.size());
    vmaUnmapMemory(m_alloc, staging.allocation);
    
    VkBufferImageCopy copyRegion{};
    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.imageSubresource.layerCount = 1;
    copyRegion.imageExtent = info.extent;
    
    vkCmdCopyBufferToImage(cmd, staging.buffer, texture.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);
    
    // Transition to SHADER_READ_ONLY
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 
        0, 0, nullptr, 0, nullptr, 1, &barrier);
    
    m_app->submitAndWaitTempCmdBuffer(cmd);
    
    m_alloc.destroyBuffer(staging);
}

void VkViewerUI::guiDrawVrMenu()
{
    // A simple window floating in front of the camera (conceptually)
    // For now just a standard ImGui window
    ImGui::SetNextWindowSize(ImVec2(300, 200), ImGuiCond_FirstUseEver);
    if(ImGui::Begin("VR Menu", &m_showVrMenu))
    {
        if(ImGui::Button("Load from Resources...", ImVec2(-1, 40)))
        {
            m_showFileDialog = true;
            // Populate file list
            m_fileList.clear();
            std::vector<std::filesystem::path> resourceDirs = getResourcesDirs();
            if (!resourceDirs.empty())
            {
              std::filesystem::path resourcesDir = resourceDirs[0];
              if (std::filesystem::exists(resourcesDir))
              {
                for (const auto& entry : std::filesystem::directory_iterator(resourcesDir))
                {
                  if (entry.is_regular_file())
                    m_fileList.push_back(entry.path().string());
                }
              }
            }
        }
        
        if(ImGui::Button("Supersplat Browser...", ImVec2(-1, 40)))
        {
            m_showSupersplatDialog = true;
            if (m_supersplatClient)
            {
                m_supersplatClient->fetchSceneList("", [this](const std::vector<SupersplatClient::Scene>& scenes) {
                    std::lock_guard<std::mutex> lock(m_thumbnailMutex);
                    m_supersplatScenes = scenes;
                    // Fetch thumbnails logic (duplicated)
                    for (const auto& scene : scenes) {
                      if (!scene.thumbnailUrl.empty()) {
                          m_supersplatClient->fetchThumbnail(scene.thumbnailUrl, 
                              [this, url=scene.thumbnailUrl](const std::vector<uint8_t>& data, int w, int h, int c) {
                                  if (data.empty()) return;
                                  std::lock_guard<std::mutex> lock(m_thumbnailMutex);
                                  std::vector<uint8_t> rgba = data;
                                  if (c == 3) {
                                      rgba.resize(w * h * 4);
                                      for (int i = w * h - 1; i >= 0; --i) {
                                          rgba[i * 4 + 3] = 255;
                                          rgba[i * 4 + 2] = data[i * 3 + 2];
                                          rgba[i * 4 + 1] = data[i * 3 + 1];
                                          rgba[i * 4 + 0] = data[i * 3 + 0];
                                      }
                                  }
                                  m_pendingThumbnails.push_back({url, rgba, w, h});
                              });
                      }
                    }
                });
            }
        }
        
        if(ImGui::Button("Close Menu", ImVec2(-1, 40)))
        {
            m_showVrMenu = false;
        }
    }
    ImGui::End();
}

// Include UI partial files (unity build pattern)
#include "vk_viewer_ui_renderer.cpp"
#include "vk_viewer_ui_depth.cpp"
#include "vk_viewer_ui_project.cpp"
#include "vk_viewer_ui_vr.cpp"

} // namespace vk_viewer
