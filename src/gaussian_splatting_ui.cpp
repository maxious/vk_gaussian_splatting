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

#include "gaussian_splatting_ui.h"
#include "vdz_loader.h"
#include "utilities.h"
#include <backends/imgui_impl_vulkan.h>
#include <imgui/imgui_internal.h>

namespace vk_gaussian_splatting {

GaussianSplattingUI::GaussianSplattingUI(nvutils::ProfilerManager*   profilerManager,
                                         nvutils::ParameterRegistry* parameterRegistry,
                                         bool*                       benchmarkEnabled)
    : GaussianSplatting(profilerManager, parameterRegistry)
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

GaussianSplattingUI::~GaussianSplattingUI(){
    // Nothing to do here
};

void GaussianSplattingUI::onAttach(nvapp::Application* app)
{
    GaussianSplatting::onAttach(app);

  // we hide the UI dy default in benchmark mode
  m_showUI = !(*m_pBenchmarkEnabled);

  // Detect FFmpeg capabilities once at startup
  VideoRenderer::initCapabilities();

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
  m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_VDZ_DEPTH, "VDZ Depth (grayscale)");
  m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_VDZ_MESH, "VDZ Depth Mesh (2.5D)");

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

void GaussianSplattingUI::onDetach()
{
#ifdef WITH_OPENXR
    destroyHandMeshes();
#endif
    GaussianSplatting::onDetach();
}

void GaussianSplattingUI::onResize(VkCommandBuffer cmd, const VkExtent2D& size)
{
  GaussianSplatting::onResize(cmd, size);
}

void GaussianSplattingUI::onPreRender()
{
#ifdef WITH_COMFYUI
  if (m_comfyClient)
  {
    m_comfyClient->update();
  }
#endif

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

  GaussianSplatting::onPreRender();
}

void GaussianSplattingUI::onRender(VkCommandBuffer cmd)
{
#ifdef WITH_OPENXR
  // Update hand meshes before rendering
  updateHandMeshes();
#endif

  GaussianSplatting::onRender(cmd);

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

void GaussianSplattingUI::onUIMenu()
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
    if(ImGui::MenuItem(ICON_MS_MOVIE " Open Video+Depth...", ""))
    {
      auto videoPath = nvgui::windowOpenFileDialog(m_app->getWindowHandle(), "Select Video File", 
                                                    "Video Files|*.mp4;*.avi;*.mov;*.mkv");
      if(!videoPath.empty())
      {
        auto vdzPath = nvgui::windowOpenFileDialog(m_app->getWindowHandle(), "Select VDZ Depth Sequence", 
                                                   "VDZ Files|*.vdz");
        if(!vdzPath.empty())
        {
          enableVideoDepthPlayback(videoPath.string(), vdzPath.string());
          prmRender.visualize = VISUALIZE_VDZ_MESH;
          prmFrame.visualize = prmRender.visualize;
          prmFrame.vdzUseVideoTexture = 1;
          m_requestUpdateShaders = true;
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
    ImGui::MenuItem(ICON_MS_VIDEOCAM " Video Export...", "", &m_showVideoExportWindow);
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

void GaussianSplattingUI::onFileDrop(const std::filesystem::path& filename)
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
  else if(extension == ".vdz")
    prmScene.depthFrameToLoadFilename = filename;
  else if(extension == ".vkgs")
    prmScene.projectToLoadFilename = filename;
  else if(extension == ".obj")
    prmScene.meshToImportFilename = filename;
  else
    LOGE("Error: unsupported file extension %s\n", extension.c_str());
}

void GaussianSplattingUI::onUIRender()
{
  // Handle pending frame save from video rendering (save G-buffer after render completes)
  if(m_pendingFrameSave)
  {
    VkImage    srcImage = m_gBuffers.getColorImage(COLOR_MAIN);
    VkExtent2D size     = {static_cast<uint32_t>(m_viewSize.x), static_cast<uint32_t>(m_viewSize.y)};
    saveFrameAsync(srcImage, size, m_pendingFramePath);
    m_pendingFrameSave = false;
  }

  // Video rendering progress
  if(m_videoRenderActive)
  {
    updateVideoRender();
  }

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
     && m_splatLoader.getStatus() == SplatLoaderAsync::State::STATE_READY)
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
  if(!prmScene.depthFrameToLoadFilename.empty())
  {
    DepthFrame depthFrame;
    if(VDZLoader::loadVDZFile(prmScene.depthFrameToLoadFilename, depthFrame))
    {
      LOGI("Depth frame loaded: %ux%u pixels\n", depthFrame.width, depthFrame.height);
      
      // Upload depth frame to GPU
      if(!m_depthManager)
      {
        m_depthManager = std::make_unique<DepthTextureManager>();
        m_depthManager->initialize(m_device, m_app->getPhysicalDevice(), m_app->getQueue(0).queue, &m_alloc);
      }
      if(m_depthManager)
      {
        VkCommandBuffer cmd = m_app->createTempCmdBuffer();
        m_depthManager->uploadDepthFrame(depthFrame, cmd);
        m_app->submitAndWaitTempCmdBuffer(cmd);
        
        // Auto-switch to depth visualization mode
        prmRender.visualize = VISUALIZE_VDZ_MESH;

        // Sync render parameter to frame parameter immediately to ensure renderer picks it up
        prmFrame.visualize = prmRender.visualize;

        prmFrame.vdzZScale = 1.0f;
        prmFrame.vdzZBias = 0.0f;
        prmFrame.vdzZGamma = 1.0f;
        prmFrame.vdzZMaxClip = depthFrame.zMax > 0.0f ? depthFrame.zMax : 10.0f;
        
        m_requestUpdateShaders = true;
        m_requestUpdateSplatData = true; 
        LOGI("Switched to VDZ depth visualization mode (zMax: %.2f)\n", prmFrame.vdzZMaxClip);
      }
      if(m_depthManager)
      {
        VkCommandBuffer cmd = m_app->createTempCmdBuffer();
        m_depthManager->uploadDepthFrame(depthFrame, cmd);
        m_app->submitAndWaitTempCmdBuffer(cmd);
        
        // Auto-switch to depth visualization mode
        prmRender.visualize = VISUALIZE_VDZ_MESH;

        // Sync render parameter to frame parameter immediately to ensure renderer picks it up
        prmFrame.visualize = prmRender.visualize;

        prmFrame.vdzZScale = 1.0f;
        prmFrame.vdzZBias = 0.0f;
        prmFrame.vdzZGamma = 1.0f;
        prmFrame.vdzZMaxClip = depthFrame.zMax > 0.0f ? depthFrame.zMax : 10.0f;
        
        m_requestUpdateShaders = true;
        m_requestUpdateSplatData = true; 
        LOGI("Switched to VDZ depth visualization mode (zMax: %.2f)\n", prmFrame.vdzZMaxClip);
      }
    }
    else
    {
      LOGE("Failed to load depth frame: %s\n", prmScene.depthFrameToLoadFilename.string().c_str());
    }
    
    // reset request
    prmScene.depthFrameToLoadFilename.clear();
  }

  if(!m_showUI)
    return;

  /////////////////
  // Draw the UI parts

guiDrawAssetsWindow();
    guiDrawPropertiesWindow();
    guiDrawRendererStatisticsWindow();
    guiDrawMemoryStatisticsWindow();
    
    // Animation controls
    if (m_animationController) {
        renderAnimationControls(true);
        renderTimelineControls();
        renderAudioControls();
        renderProgressDisplay();
    }
    
    // Animation controls
    if (m_animationController) {
        renderAnimationControls(true);
        renderTimelineControls();
        renderAudioControls();
        renderProgressDisplay();
    }

  guiDrawFooterBar();

  if(m_showVideoExportWindow)
  {
    guiDrawVideoExportWindow();
  }

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

void GaussianSplattingUI::guiDrawAssetsWindow()
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

void GaussianSplattingUI::guiDrawRendererTree()
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

void GaussianSplattingUI::guiDrawCameraTree()
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

void GaussianSplattingUI::guiDrawLightTree()
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

void GaussianSplattingUI::guiDrawRadianceFieldsTree()
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

void GaussianSplattingUI::guiDrawObjectTree()
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

void GaussianSplattingUI::guiDrawDepthStreamTree()
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
    if(m_videoDepthPlaybackMode && m_videoDecoder)
    {
      ImGuiTreeNodeFlags videoFlags = leaf_flags;
      if(m_selectedAsset == GUI_DEPTH_STREAM && m_selectedItemIndex == 1)
        videoFlags |= ImGuiTreeNodeFlags_Selected;
      
      int vw, vh;
      m_videoDecoder->getDimensions(vw, vh);
      std::string videoLabel = fmt::format(ICON_MS_VIDEOCAM " Video ({}x{})", vw, vh);
      ImGui::TreeNodeEx(videoLabel.c_str(), videoFlags);
      if(ImGui::IsItemClicked())
      {
        m_selectedAsset = GUI_DEPTH_STREAM;
        m_selectedItemIndex = 1;
      }
    }
#endif

    // Show depth layer
    if(m_vdzSequence && m_vdzSequence->isOpen())
    {
      ImGuiTreeNodeFlags depthFlags = leaf_flags;
      if(m_selectedAsset == GUI_DEPTH_STREAM && m_selectedItemIndex == 2)
        depthFlags |= ImGuiTreeNodeFlags_Selected;
      
      std::string depthLabel = fmt::format(ICON_MS_LANDSCAPE " Depth ({}x{}, {} frames)", 
                                           m_vdzSequence->getWidth(), 
                                           m_vdzSequence->getHeight(),
                                           m_vdzSequence->getFrameCount());
      ImGui::TreeNodeEx(depthLabel.c_str(), depthFlags);
      if(ImGui::IsItemClicked())
      {
        m_selectedAsset = GUI_DEPTH_STREAM;
        m_selectedItemIndex = 2;
      }
    }
    else if(m_depthClient)
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

void GaussianSplattingUI::guiDrawPropertiesWindow()
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

void GaussianSplattingUI::guiDrawRendererProperties()
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
                        (m_depthManager != nullptr) ||
                        (prmRender.visualize == VISUALIZE_VDZ_DEPTH) || 
                        (prmRender.visualize == VISUALIZE_VDZ_MESH);

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

  // VDZ Depth Mesh controls (shown when VDZ mesh mode is selected)
  if(prmRender.visualize == VISUALIZE_VDZ_MESH)
  {
    PE::begin("## VDZ Depth Mesh");
    
    // Video+Depth playback controls
    if(m_videoDepthPlaybackMode)
    {
      bool useVideo = prmFrame.vdzUseVideoTexture != 0;
      if(PE::Checkbox("Use Video Texture", &useVideo,
                      "When enabled, uses the video RGB texture.\n"
                      "When disabled, shows depth as a colormap."))
      {
        prmFrame.vdzUseVideoTexture = useVideo ? 1 : 0;
      }
      
      if(m_vdzSequence)
      {
        PE::Text("VDZ Frames", "%zu", m_vdzSequence->getFrameCount());
        PE::Text("Duration", "%.1f s", static_cast<float>(m_vdzSequence->getDurationMs()) / 1000.0f);
        size_t displayFrameIndex = (m_lastVdzFrameIndex == SIZE_MAX) ? 0 : m_lastVdzFrameIndex;
        PE::Text("Current Frame", "%zu / %zu", displayFrameIndex, m_vdzSequence->getFrameCount());
      }
      
      // Playback controls
      PE::entry("Playback", [this]() {
        bool changed = false;
        if(m_playbackPaused)
        {
          if(ImGui::Button("Play"))
          {
            m_playbackPaused = false;
#ifdef WITH_VIDEO_DECODER
            if(m_videoDecoder)
            {
              m_videoDecoder->resume();
            }
#endif
            m_playbackStartTime = std::chrono::steady_clock::now();
            changed = true;
          }
        }
        else
        {
          if(ImGui::Button("Pause"))
          {
            m_playbackPaused = true;
#ifdef WITH_VIDEO_DECODER
            if(m_videoDecoder)
            {
              m_videoDecoder->pause();
            }
#endif
            changed = true;
          }
        }
        ImGui::SameLine();
        if(ImGui::Button("Restart"))
        {
#ifdef WITH_VIDEO_DECODER
          if(m_videoDecoder)
          {
            // Seek to beginning - seekToTime handles restarting if thread stopped
            m_videoDecoder->seekToTime(0.0);
          }
#endif
          m_lastVdzFrameIndex = SIZE_MAX;
          m_playbackStartTime = std::chrono::steady_clock::now();
          m_playbackPaused = false;
          changed = true;
        }
        return changed;
      });
    }
    else
    {
      PE::entry("Load Video+Depth", [this]() {
        static std::filesystem::path videoPath;
        static std::filesystem::path vdzPath;
        
        if(ImGui::Button("Load Video..."))
        {
          videoPath = nvgui::windowOpenFileDialog(m_app->getWindowHandle(), "Select Video File", "Video Files|*.mp4;*.avi;*.mov;*.mkv");
        }
        ImGui::SameLine();
        ImGui::Text("%s", videoPath.empty() ? "(none)" : videoPath.filename().string().c_str());
        
        if(ImGui::Button("Load VDZ..."))
        {
          vdzPath = nvgui::windowOpenFileDialog(m_app->getWindowHandle(), "Select VDZ Depth Sequence", "VDZ Files|*.vdz");
        }
        ImGui::SameLine();
        ImGui::Text("%s", vdzPath.empty() ? "(none)" : vdzPath.filename().string().c_str());
        
        if(!videoPath.empty() && !vdzPath.empty())
        {
          if(ImGui::Button("Start Playback"))
          {
            enableVideoDepthPlayback(videoPath.string(), vdzPath.string());
            prmFrame.vdzUseVideoTexture = 1;
            videoPath.clear();
            vdzPath.clear();
          }
        }
        
        return false;
      });
    }
    
    PE::SliderFloat("Z Scale", &prmFrame.vdzZScale, 0.0f, 10.0f, "%.2f", 0,
                    "Depth scale multiplier - controls how depth values affect vertex displacement");
    PE::SliderFloat("Z Bias", &prmFrame.vdzZBias, -5.0f, 5.0f, "%.2f", 0,
                    "Global Z offset added after scaling");
    PE::SliderFloat("Z Gamma", &prmFrame.vdzZGamma, 0.1f, 5.0f, "%.2f", 0,
                    "Gamma correction for depth (depth = pow(depth, gamma))");
    PE::SliderFloat("Z Max Clip", &prmFrame.vdzZMaxClip, 0.0f, 10.0f, "%.2f", 0,
                    "Maximum depth clipping threshold");
    PE::SliderFloat("Plane Scale", &prmFrame.vdzPlaneScale, 0.1f, 10.0f, "%.2f", 0,
                    "Scale of the view-aligned plane");
    PE::SliderFloat("Aspect Ratio", &prmFrame.vdzAspect, 0.5f, 3.0f, "%.3f", 0,
                    "Aspect ratio (width/height) of the depth texture");
    PE::SliderFloat("Edge Threshold", &prmFrame.vdzEdgeThreshold, 0.0f, 1.0f, "%.3f", 0,
                    "Depth gradient threshold for edge detection.\n"
                    "Fragments with depth discontinuities above this threshold are discarded.\n"
                    "0 = disabled, higher values = more aggressive edge culling");

    bool worldSpace = prmFrame.vdzWorldSpaceMode != 0;
    if(PE::Checkbox("World Space / VR Mode", &worldSpace,
                    "Detach mesh from camera and place it in the world.\n"
                    "Allows 6DOF movement and VR viewing.\n"
                    "When enabled, the mesh freezes at the current camera position."))
    {
      prmFrame.vdzWorldSpaceMode = worldSpace ? 1 : 0;
      m_requestUpdateShaders = true;
    }

    PE::end();
  }

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
  PE::begin("## DLSS-RR Denoising");
  bool wasEnabled = m_dlssRREnabled;
  if(PE::Checkbox("Enable DLSS-RR", &m_dlssRREnabled, 
                  "Enable NVIDIA DLSS Ray Reconstruction denoiser for RTX mode.\n"
                  "Requires RTX GPU with DLSS support."))
  {
    if(m_dlssRREnabled && !m_dlssRRInitialized)
    {
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

        int parametric = prmRtxData.useAABBs ? PARTICLE_FORMAT_PARAMETRIC : PARTICLE_FORMAT_ICOSAHEDRON;

        if(PE::entry(
               "Particles format", [&]() { return m_ui.enumCombobox(GUI_PARTICLE_FORMAT, "##ID", &parametric); },
               "This is a convenience shortcut to switch the Radiance Field use AABB property.\n"
               "Note that activating parametric will force the use of TLAS instance.\n"))
        {
          if(parametric == PARTICLE_FORMAT_ICOSAHEDRON)
          {
            prmRtxData.useAABBs = false;
          }
          if(parametric == PARTICLE_FORMAT_PARAMETRIC)
          {
            prmRtxData.useAABBs         = true;
            prmRtxData.useTlasInstances = true;
          }
          m_requestUpdateSplatData = true;
        }

        if(PE::Checkbox("Adaptive clamp", &prmRtx.kernelAdaptiveClamping))
          m_requestUpdateSplatData = true;

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
  }
  ImGui::EndTabBar();
}

void GaussianSplattingUI::guiDrawSplatSetProperties()
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

void GaussianSplattingUI::guiDrawMeshTransformProperties()
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

void GaussianSplattingUI::guiDrawMeshMaterialProperties()
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

void GaussianSplattingUI::guiDrawCameraProperties()
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

void GaussianSplattingUI::guiDrawNavigationProperties()
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

void GaussianSplattingUI::guiDrawLightProperties()
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

bool GaussianSplattingUI::guiGetTransform(glm::vec3& scale,
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

void GaussianSplattingUI::guiDrawRendererStatisticsWindow()
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
    ImGui::End();
  }
}


void GaussianSplattingUI::guiDrawMemoryStatisticsWindow()
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

void GaussianSplattingUI::guiDrawFooterBar()
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

void GaussianSplattingUI::guiAddToRecentFiles(std::filesystem::path filePath, int historySize)
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

void GaussianSplattingUI::guiAddToRecentProjects(std::filesystem::path filePath, int historySize)
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

void GaussianSplattingUI::guiRegisterIniFileHandlers()
{
  // mandatory to work, see ImGui::DockContextInitialize as an example
  auto readOpen = [](ImGuiContext*, ImGuiSettingsHandler* handler, const char* name) -> void* {
    if(strcmp(name, "Data") != 0)
      return NULL;
    // Make sure we clear out our current recent vectors so we don't just keep adding to the list every time we load
    // This is if the .ini file is loaded twice, which happens in nvpro_core2
    auto* ui = static_cast<GaussianSplattingUI*>(handler->UserData);
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
      auto* self = static_cast<GaussianSplattingUI*>(handler->UserData);
      buf->appendf("[%s][Data]\n", handler->TypeName);
      for(const auto& file : self->m_recentFiles)
      {
        buf->appendf("File=%s\n", file.string().c_str());
      }
      buf->append("\n");
    };

    // Load settings handler, not using capture so can be used as a function pointer
    auto loadRecentFilesFromIni = [](ImGuiContext* ctx, ImGuiSettingsHandler* handler, void* entry, const char* line) {
      auto* self = static_cast<GaussianSplattingUI*>(handler->UserData);
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
      auto* self = static_cast<GaussianSplattingUI*>(handler->UserData);
      buf->appendf("[%s][Data]\n", handler->TypeName);
      for(const auto& file : self->m_recentProjects)
      {
        buf->appendf("File=%s\n", file.string().c_str());
      }
      buf->append("\n");
    };

    // Load settings handler, not using capture so can be used as a function pointer
    auto loadRecentProjectsFromIni = [](ImGuiContext* ctx, ImGuiSettingsHandler* handler, void* entry, const char* line) {
      auto* self = static_cast<GaussianSplattingUI*>(handler->UserData);
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

// some macros to fetch jsoin values only if exist and affect to val

#define LOAD1(val, item, name)                                                                                         \
  if((item).contains(name))                                                                                            \
  (val) = (item)[name]

#define LOAD2(val, item, name)                                                                                         \
  if((item).contains(name))                                                                                            \
  (val) = {(item)[name][0], (item)[name][1]}

#define LOAD3(val, item, name)                                                                                         \
  if((item).contains(name))                                                                                            \
  (val) = {(item)[name][0], (item)[name][1], (item)[name][2]}

#define LOAD4_FROM3(val, item, name)                                                                                   \
  if((item).contains(name))                                                                                            \
  (val) = glm::vec4((item)[name][0], (item)[name][1], (item)[name][2], 0.0f)

// This method is multi pass
bool GaussianSplattingUI::loadProjectIfNeeded()
{
  // Nothing to load
  if(prmScene.projectToLoadFilename.empty())
    return true;

  auto path = prmScene.projectToLoadFilename.string();

  // load the json and set loading status
  if(!loadingProject)
  {
    if(!m_radianceFields.empty())
      ImGui::OpenPopup("Load .vkg project file ?");

    // Always center this window when appearing
    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

    bool doReset = true;

    if(ImGui::BeginPopupModal("Load .vkg project file ?", NULL, ImGuiWindowFlags_AlwaysAutoResize))
    {
      doReset = false;

      ImGui::Text("The current project will be entirely replaced.\nThis operation cannot be undone!");
      ImGui::Separator();

      if(ImGui::Button("OK", ImVec2(120, 0)))
      {
        doReset = true;
        ImGui::CloseCurrentPopup();
      }
      ImGui::SetItemDefaultFocus();
      ImGui::SameLine();
      if(ImGui::Button("Cancel", ImVec2(120, 0)))
      {
        // cancel any request leading to a reset
        prmScene.sceneToLoadFilename   = "";
        prmScene.projectToLoadFilename = "";
        ImGui::CloseCurrentPopup();
      }
      ImGui::EndPopup();
    }

    if(doReset)
    {
      LOGI("Opening project file %s\n", path.c_str());

      std::ifstream i(path);
      if(!i.is_open())
      {
        LOGE("Error : unable to open project file %s\n", path.c_str());
        prmScene.projectToLoadFilename = "";
        return false;
      }

      try
      {
        i >> data;
      }
      catch(...)
      {
        LOGE("Error : invalid project file %s\n", path.c_str());
        prmScene.projectToLoadFilename = "";
        return false;
      }
      i.close();

      loadingProject = true;

      // Initiate SplatSet loading
      if(!data["splats"].empty())
      {
        const auto& item             = data["splats"][0];
        prmScene.sceneToLoadFilename = makeAbsolutePath(std::filesystem::path(path).parent_path(), item["path"]);
        
        // Warn if project has multiple splat sets (only first will be loaded initially)
        if(data["splats"].size() > 1)
        {
          LOGW("Project contains %zu radiance fields. Only the first will be loaded initially. "
               "Use File > Add to load additional files.\n", data["splats"].size());
        }
      }
    }

    // Will do the rest of the work on next call when splatset is loaded
    return true;
  }

  // we skip until the splat set is being loaded
  if(m_splatLoader.getStatus() != SplatLoaderAsync::State::STATE_READY)
    return true;

  // we finalize
  guiAddToRecentProjects(prmScene.projectToLoadFilename);
  loadingProject                 = false;
  prmScene.projectToLoadFilename = "";

  try
  {
    // Renderer
    if(data.contains("renderer"))
    {
      const auto& item = data["renderer"];

      if(item.contains("vsync"))
        m_app->setVsync(item["vsync"]);

      LOAD1(prmSelectedPipeline, item, "pipeline");

      LOAD1(prmRender.maxShDegree, item, "maxShDegree");
      LOAD1(prmRender.opacityGaussianDisabled, item, "opacityGaussianDisabled");
      LOAD1(prmRender.showShOnly, item, "showShOnly");
      LOAD1(prmRender.visualize, item, "visualize");
      LOAD1(prmRender.wireframe, item, "wireframe");

      LOAD1(prmRaster.cpuLazySort, item, "cpuLazySort");
      LOAD1(prmRaster.distShaderWorkgroupSize, item, "distShaderWorkgroupSize");
      LOAD1(prmRaster.fragmentBarycentric, item, "fragmentBarycentric");
      LOAD1(prmRaster.frustumCulling, item, "frustumCulling");
      LOAD1(prmRaster.meshShaderWorkgroupSize, item, "meshShaderWorkgroupSize");
      LOAD1(prmRaster.pointCloudModeEnabled, item, "pointCloudModeEnabled");
      LOAD1(prmRaster.sortingMethod, item, "sortingMethod");

      LOAD1(prmRtx.temporalSampling, item, "temporalSampling");
      LOAD1(prmFrame.frameSampleMax, item, "temporalSamplesCount");
      LOAD1(prmRtx.kernelAdaptiveClamping, item, "kernelAdaptiveClamping");
      LOAD1(prmRtx.kernelDegree, item, "kernelDegree");
      LOAD1(prmRtx.kernelMinResponse, item, "kernelMinResponse");
      LOAD1(prmRtx.payloadArraySize, item, "payloadArraySize");
    }
    // Splat global options
    if(data.contains("splatsGlobals"))
    {
      const auto& item = data["splatsGlobals"];

      LOAD1(prmData.dataStorage, item, "dataStorage");
      LOAD1(prmData.shFormat, item, "shFormat");

      LOAD1(prmRtxData.compressBlas, item, "compressBlas");
      LOAD1(prmRtxData.useAABBs, item, "useAABBs");
      LOAD1(prmRtxData.useTlasInstances, item, "useTlasInstances");

      m_requestUpdateSplatData = true;
      m_requestUpdateSplatAs   = true;
    }
    // Parse splat settings
    if(data.contains("splats"))
    {
      if(!data["splats"].empty())
      {
        const auto& item = data["splats"][0];
        LOAD3(m_splatSetVk.translation, item, "position");
        LOAD3(m_splatSetVk.rotation, item, "rotation");
        LOAD3(m_splatSetVk.scale, item, "scale");

        computeTransform(m_splatSetVk.scale, m_splatSetVk.rotation, m_splatSetVk.translation, m_splatSetVk.transform,
                         m_splatSetVk.transformInverse);

        // delay update of Acceleration Structures if not using ray tracing
        m_requestDelayedUpdateSplatAs = true;
      }
    }

    // Load all the meshes
    if(data.contains("meshes"))
    {
      auto meshId = 0;
      for(const auto& item : data["meshes"])
      {
        std::string relPath;
        LOAD1(relPath, item, "path");
        if(relPath.empty())
          continue;

        auto meshPath = makeAbsolutePath(std::filesystem::path(path).parent_path(), relPath);
        if(!m_meshSetVk.loadModel(meshPath.string()))
        {
          meshId++;
          continue;
        }
        // Access to newly created mesh/instance
        auto& instance = m_meshSetVk.instances.back();
        auto& mesh     = m_meshSetVk.meshes[instance.objIndex];

        // Transform
        LOAD3(instance.translation, item, "position");
        LOAD3(instance.rotation, item, "rotation");
        LOAD3(instance.scale, item, "scale");
        computeTransform(instance.scale, instance.rotation, instance.translation, instance.transform, instance.transformInverse);

        // Materials
        if(item.contains("materials"))
        {
          auto matId = 0;
          for(const auto& matItem : item["materials"])
          {
            auto& mat = mesh.materials[matId];
            LOAD4_FROM3(mat.ambient, matItem, "ambient");
            LOAD4_FROM3(mat.diffuse, matItem, "diffuse");
            LOAD1(mat.illum, matItem, "illum");
            LOAD1(mat.ior, matItem, "ior");
            LOAD1(mat.shininess, matItem, "shininess");
            LOAD4_FROM3(mat.specular, matItem, "specular");
            LOAD4_FROM3(mat.transmittance, matItem, "transmittance");

            matId++;
          }
          m_meshSetVk.updateObjMaterialsBuffer(meshId);
        }

        meshId++;
      }
      m_requestUpdateMeshData = true;
      m_requestUpdateShaders  = true;
    }

    // Parse camera
    if(data.contains("camera"))
    {
      auto&  item = data["camera"];
      Camera cam;
      LOAD1(cam.model, item, "model");
      LOAD3(cam.ctr, item, "ctr");
      LOAD3(cam.eye, item, "eye");
      LOAD3(cam.up, item, "up");
      LOAD1(cam.fov, item, "fov");
      LOAD1(cam.dofEnabled, item, "dofEnabled");
      LOAD1(cam.focusDist, item, "focusDist");
      LOAD1(cam.aperture, item, "aperture");
      m_cameraSet.setCamera(cam);
    }
    // Parse camera presets
    if(data.contains("cameras"))
    {
      for(const auto& item : data["cameras"])
      {
        Camera cam;
        LOAD1(cam.model, item, "model");
        LOAD3(cam.ctr, item, "ctr");
        LOAD3(cam.eye, item, "eye");
        LOAD3(cam.up, item, "up");
        LOAD1(cam.fov, item, "fov");
        LOAD1(cam.dofEnabled, item, "dofEnabled");
        LOAD1(cam.focusDist, item, "focusDist");
        LOAD1(cam.aperture, item, "aperture");
        m_cameraSet.createPreset(cam);
      }
    }
    // Parse lights
    if(data.contains("lights"))
    {
      bool defaultLight = true;
      for(const auto& item : data["lights"])
      {
        // A default light already exists, we only modify it
        uint64_t id = 0;
        if(!defaultLight)
        {
          id = m_lightSet.createLight();
        }
        auto& light = m_lightSet.getLight(id);
        LOAD1(light.type, item, "type");
        LOAD3(light.position, item, "position");
        LOAD1(light.intensity, item, "intensity");
        defaultLight = false;
      }
      m_requestUpdateLightsBuffer = true;
    }

    return true;
  }
  catch(...)
  {
    return false;
  }
}

bool GaussianSplattingUI::saveProject(std::string path)
{
  std::ofstream o(path);
  if(!o.is_open())
    return false;

  try
  {
    json data;

    // Renderer
    {
      json item;

      item["vsync"] = m_app->isVsync();

      item["pipeline"] = prmSelectedPipeline;

      item["maxShDegree"]             = prmRender.maxShDegree;
      item["opacityGaussianDisabled"] = prmRender.opacityGaussianDisabled;
      item["showShOnly"]              = prmRender.showShOnly;
      item["visualize"]               = prmRender.visualize;
      item["wireframe"]               = prmRender.wireframe;

      item["cpuLazySort"]             = prmRaster.cpuLazySort;
      item["distShaderWorkgroupSize"] = prmRaster.distShaderWorkgroupSize;
      item["fragmentBarycentric"]     = prmRaster.fragmentBarycentric;
      item["frustumCulling"]          = prmRaster.frustumCulling;
      item["meshShaderWorkgroupSize"] = prmRaster.meshShaderWorkgroupSize;
      item["pointCloudModeEnabled"]   = prmRaster.pointCloudModeEnabled;
      item["sortingMethod"]           = prmRaster.sortingMethod;

      item["temporalSampling"]       = prmRtx.temporalSampling;
      item["temporalSamplesCount"]   = prmFrame.frameSampleMax;
      item["kernelAdaptiveClamping"] = prmRtx.kernelAdaptiveClamping;
      item["kernelDegree"]           = prmRtx.kernelDegree;
      item["kernelMinResponse"]      = prmRtx.kernelMinResponse;
      item["payloadArraySize"]       = prmRtx.payloadArraySize;

      data["renderer"] = item;
    }

    // Active Camera
    {
      const auto& cam = m_cameraSet.getCamera();
      json        item;
      item["model"]      = cam.model;
      item["ctr"]        = {cam.ctr.x, cam.ctr.y, cam.ctr.z};
      item["eye"]        = {cam.eye.x, cam.eye.y, cam.eye.z};
      item["up"]         = {cam.up.x, cam.up.y, cam.up.z};
      item["fov"]        = cam.fov;
      item["dofEnabled"] = cam.dofEnabled;
      item["focusDist"]  = cam.focusDist;
      item["aperture"]   = cam.aperture;

      data["camera"] = item;
    }

    // Camera presets
    data["cameras"] = json::array();
    for(auto camId = 0; camId < m_cameraSet.size(); ++camId)
    {
      auto cam = m_cameraSet.getPreset(camId);

      json item;
      item["model"]      = cam.model;
      item["ctr"]        = {cam.ctr.x, cam.ctr.y, cam.ctr.z};
      item["eye"]        = {cam.eye.x, cam.eye.y, cam.eye.z};
      item["up"]         = {cam.up.x, cam.up.y, cam.up.z};
      item["fov"]        = cam.fov;
      item["dofEnabled"] = cam.dofEnabled;
      item["focusDist"]  = cam.focusDist;
      item["aperture"]   = cam.aperture;

      data["cameras"].push_back(item);
    }

    // Lights
    data["lights"] = json::array();
    for(auto lightId = 0; lightId < m_lightSet.numLights; ++lightId)
    {
      const auto& light = m_lightSet.getLight(lightId);

      json item;
      item["type"]      = light.type;
      item["position"]  = {light.position.x, light.position.y, light.position.z};
      item["intensity"] = light.intensity;

      data["lights"].push_back(item);
    }

    // Splat global options
    {
      json item;
      item["dataStorage"] = prmData.dataStorage;
      item["shFormat"]    = prmData.shFormat;

      item["compressBlas"]     = prmRtxData.compressBlas;
      item["useAABBs"]         = prmRtxData.useAABBs;
      item["useTlasInstances"] = prmRtxData.useTlasInstances;

      data["splatsGlobals"] = item;
    }

    // Splat sets - save all radiance fields
    data["splats"] = json::array();
    for(const auto& field : m_radianceFields)
    {
      json item;
      item["path"]     = getRelativePath(std::filesystem::path(path).parent_path(), field.filename);
      item["position"] = {m_splatSetVk.translation.x, m_splatSetVk.translation.y, m_splatSetVk.translation.z};
      item["rotation"] = {m_splatSetVk.rotation.x, m_splatSetVk.rotation.y, m_splatSetVk.rotation.z};
      item["scale"]    = {m_splatSetVk.scale.x, m_splatSetVk.scale.y, m_splatSetVk.scale.z};

      data["splats"].push_back(item);
    }

    // Meshes
    data["meshes"] = json::array();
    for(auto instId = 0; instId < m_meshSetVk.instances.size(); ++instId)
    {
      const auto& instance = m_meshSetVk.instances[instId];
      const auto& mesh     = m_meshSetVk.meshes[instance.objIndex];

      json item;
      item["path"] = getRelativePath(std::filesystem::path(path).parent_path(), mesh.path);
      item["name"] = mesh.name;

      // Transform
      item["position"] = {instance.translation.x, instance.translation.y, instance.translation.z};
      item["rotation"] = {instance.rotation.x, instance.rotation.y, instance.rotation.z};
      item["scale"]    = {instance.scale.x, instance.scale.y, instance.scale.z};

      // Material override
      item["materials"] = json::array();

      for(auto matId = 0; matId < mesh.matNames.size(); ++matId)
      {
        json matItem;

        const auto& name = mesh.matNames[matId];
        const auto& mat  = mesh.materials[matId];

        matItem["name"]          = name;
        matItem["ambient"]       = {mat.ambient.x, mat.ambient.y, mat.ambient.z};
        matItem["diffuse"]       = {mat.diffuse.x, mat.diffuse.y, mat.diffuse.z};
        matItem["illum"]         = mat.illum;
        matItem["ior"]           = mat.ior;
        matItem["shininess"]     = mat.shininess;
        matItem["specular"]      = {mat.specular.x, mat.specular.y, mat.specular.z};
        matItem["transmittance"] = {mat.transmittance.x, mat.transmittance.y, mat.transmittance.z};

        item["materials"].push_back(matItem);
      }

      data["meshes"].push_back(item);
    }

    o << std::setw(4) << data << std::endl;
    o.close();
    return true;
  }
  catch(...)
  {
    return false;
  }
}

void GaussianSplattingUI::dumpSplat(uint32_t splatIdx)
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
void GaussianSplattingUI::guiDrawComfyUIWindow()
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

void GaussianSplattingUI::onComfyUIWorkflowComplete(const ComfyUIClient::WorkflowResult& result)
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

void GaussianSplattingUI::guiDrawVideoExportWindow()
{
  namespace PE = nvgui::PropertyEditor;

  ImGui::SetNextWindowSize(ImVec2(450, 600), ImGuiCond_FirstUseEver);
  if(!ImGui::Begin("Video Export", &m_showVideoExportWindow))
  {
    ImGui::End();
    return;
  }

  bool isRendering = m_videoRenderer.isRendering();

  ImGui::BeginDisabled(isRendering);

  if(ImGui::CollapsingHeader("Trajectory", ImGuiTreeNodeFlags_DefaultOpen))
  {
    PE::begin("##Trajectory");

    static const char* trajectoryTypes[] = {"Orbit", "Swipe", "Rotate + Zoom", "Shake"};
    int                trajectoryIdx     = static_cast<int>(m_videoSettings.trajectory.type);
    if(trajectoryIdx > 3) trajectoryIdx = 0;  // Clamp if invalid
    if(PE::entry(
           "Type", [&]() { return ImGui::Combo("##TrajType", &trajectoryIdx, trajectoryTypes, IM_ARRAYSIZE(trajectoryTypes)); },
           "Camera movement pattern"))
    {
      m_videoSettings.trajectory.type = static_cast<TrajectoryType>(trajectoryIdx);
    }

    PE::SliderFloat("Orbit Radius", &m_videoSettings.trajectory.orbitRadius, 0.01f, 5.0f, "%.2f m", 0,
                    "Lateral camera movement range");

    if(m_videoSettings.trajectory.type == TrajectoryType::ROTATE_FORWARD)
    {
      PE::SliderFloat("Zoom Range", &m_videoSettings.trajectory.zoomRange, 0.0f, 2.0f, "%.2f m", 0,
                      "Forward/backward movement range");
    }

    PE::SliderInt("Orbits/Cycles", &m_videoSettings.trajectory.numOrbits, 1, 10, "%d", 0, "Number of complete cycles");

    PE::Checkbox("Look at Center", &m_videoSettings.trajectory.lookAtCenter, "Keep camera pointed at scene center");

    if(m_videoSettings.trajectory.type == TrajectoryType::SWIPE)
    {
      PE::Checkbox("Ping-Pong", &m_videoSettings.trajectory.pingPong, "Return to start position smoothly");
    }

    PE::end();
  }

  if(ImGui::CollapsingHeader("Video Settings", ImGuiTreeNodeFlags_DefaultOpen))
  {
    PE::begin("##VideoSettings");

    static const char* frameRates[] = {"24 fps", "30 fps", "60 fps", "120 fps"};
    static int         frameRateValues[] = {24, 30, 60, 120};
    int                frameRateIdx = 1;
    for(int i = 0; i < 4; ++i)
    {
      if(frameRateValues[i] == m_videoSettings.frameRate)
        frameRateIdx = i;
    }
    if(PE::entry(
           "Frame Rate", [&]() { return ImGui::Combo("##FPS", &frameRateIdx, frameRates, IM_ARRAYSIZE(frameRates)); },
           "Output video frame rate"))
    {
      m_videoSettings.frameRate = frameRateValues[frameRateIdx];
    }

    PE::SliderFloat("Duration", &m_videoSettings.durationSec, 1.0f, 120.0f, "%.1f sec", 0, "Total video length");

    int totalFrames = m_videoSettings.getTotalFrames();
    PE::Text("Total Frames", "%d", totalFrames);

    PE::end();
  }

  if(ImGui::CollapsingHeader("Resolution", ImGuiTreeNodeFlags_DefaultOpen))
  {
    PE::begin("##Resolution");

    m_videoSettings.width  = static_cast<int>(m_viewSize.x);
    m_videoSettings.height = static_cast<int>(m_viewSize.y);

    PE::Text("Size", "%d x %d", m_videoSettings.width, m_videoSettings.height);
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Resize viewport to change resolution");

    PE::end();
  }

  if(ImGui::CollapsingHeader("Stereo VR (SBS)", ImGuiTreeNodeFlags_DefaultOpen))
  {
    PE::begin("##StereoVR");

    PE::Checkbox("Enable SBS Stereo", &m_videoSettings.enableSBS, "Render side-by-side stereo for VR");

    ImGui::BeginDisabled(!m_videoSettings.enableSBS);

    float ipdMM = m_videoSettings.stereoIPD * 1000.0f;
    if(PE::SliderFloat("IPD", &ipdMM, 50.0f, 75.0f, "%.1f mm", 0, "Inter-pupillary distance"))
    {
      m_videoSettings.stereoIPD = ipdMM / 1000.0f;
    }

    PE::SliderFloat("Convergence", &m_videoSettings.stereoConvergence, 0.1f, 10.0f, "%.2f m", 0,
                    "Distance where stereo images overlap perfectly");

    PE::Checkbox("Off-Axis Projection", &m_videoSettings.stereoOffAxis, "Use asymmetric frustum for proper stereo");

    ImGui::EndDisabled();

    PE::end();
  }

  if(ImGui::CollapsingHeader("Output", ImGuiTreeNodeFlags_DefaultOpen))
  {
    PE::begin("##Output");

    static char outputDir[512] = "";
    if(outputDir[0] == '\0')
    {
      auto defaultPath = std::filesystem::current_path() / "video_output";
#ifdef _WIN32
      strncpy_s(outputDir, sizeof(outputDir), defaultPath.string().c_str(), _TRUNCATE);
#else
      snprintf(outputDir, sizeof(outputDir), "%s", defaultPath.string().c_str());
#endif
      m_videoSettings.outputDir = defaultPath;
    }

    if(PE::entry(
           "Directory",
           [&]()
           {
             bool changed = ImGui::InputText("##OutDir", outputDir, sizeof(outputDir));
             return changed;
           },
           "Enter output directory path"))
    {
      m_videoSettings.outputDir = outputDir;
    }

    static char outputName[256] = "video";
    if(PE::InputText("Filename", outputName, sizeof(outputName)))
    {
      m_videoSettings.outputName = outputName;
    }

    static const char* formats[]    = {"TGA", "HDR (Radiance .hdr)"};
    int                formatIdx    = static_cast<int>(m_videoSettings.outputFormat);
    if(PE::entry(
           "Frame Format", [&]() { return ImGui::Combo("##Format", &formatIdx, formats, IM_ARRAYSIZE(formats)); }, "Frame output format"))
    {
      m_videoSettings.outputFormat = static_cast<VideoOutputFormat>(formatIdx);
    }

    if(m_videoSettings.outputFormat == VideoOutputFormat::FORMAT_HDR)
    {
      const auto& caps = VideoRenderer::getCapabilities();
      if(!caps.hdr10Available)
      {
        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.3f, 1.0f), ICON_MS_WARNING " FFmpeg 6+ required for HDR10 video");
      }
      else
      {
        ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), ICON_MS_CHECK_CIRCLE " HDR10 encoding (BT.2020/PQ)");
      }
    }

    static const char* codecs[]     = {"NVENC HEVC HQ", "NVENC H.264 HQ", "NVENC HEVC Lossless", "NVENC H.264 Lossless",
                                       "H.264 Lossless (CPU)", "H.265 Lossless (CPU)", "ProRes 4444"};
    int                codecIdx     = static_cast<int>(m_videoSettings.codec);
    if(PE::entry(
           "Codec", [&]() { return ImGui::Combo("##Codec", &codecIdx, codecs, IM_ARRAYSIZE(codecs)); }, "Video encoding codec"))
    {
      m_videoSettings.codec = static_cast<VideoCodec>(codecIdx);
    }

    const auto& caps = VideoRenderer::getCapabilities();
    if(caps.available)
    {
      if(caps.nvencAvailable)
      {
        ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), ICON_MS_CHECK_CIRCLE " FFmpeg + NVENC available");
      }
      else
      {
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f), ICON_MS_CHECK_CIRCLE " FFmpeg available (no NVENC)");
      }
    }
    else
    {
      ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.3f, 1.0f), ICON_MS_WARNING " FFmpeg not found - frames only");
    }

    PE::Checkbox("Encode Video", &m_videoSettings.encodeVideo, "Automatically encode frames to video");
    PE::Checkbox("Delete Frames After", &m_videoSettings.deleteFramesAfterEncode, "Remove PNG files after encoding");

    PE::end();
  }

  ImGui::EndDisabled();

  ImGui::Separator();

  if(isRendering)
  {
    auto progress = m_videoRenderer.getProgress();

    ImGui::ProgressBar(progress.progressPct / 100.0f, ImVec2(-1, 0),
                       (std::to_string(progress.currentFrame) + "/" + std::to_string(progress.totalFrames)).c_str());

    ImGui::TextWrapped("%s", progress.statusMessage.c_str());

    if(!progress.errorMessage.empty())
    {
      ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "%s", progress.errorMessage.c_str());
    }

    if(ImGui::Button("Cancel", ImVec2(-1, 0)))
    {
      m_videoRenderer.cancelRender();
      m_videoRenderActive = false;
    }
  }
  else
  {
    auto progress = m_videoRenderer.getProgress();
    if(progress.state == VideoRenderState::STATE_COMPLETED)
    {
      ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), ICON_MS_CHECK_CIRCLE " %s", progress.statusMessage.c_str());
    }
    else if(progress.state == VideoRenderState::STATE_ERROR)
    {
      ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), ICON_MS_ERROR " %s", progress.errorMessage.c_str());
    }

    bool sceneLoaded = !m_radianceFields.empty();
    ImGui::BeginDisabled(!sceneLoaded);

    if(ImGui::Button(ICON_MS_MOVIE " Start Render", ImVec2(-1, 30)))
    {
      startVideoRender();
    }

    ImGui::EndDisabled();

    if(!sceneLoaded)
    {
      ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.3f, 1.0f), "Load a scene first");
    }
  }

  ImGui::End();
}

void GaussianSplattingUI::startVideoRender()
{
  Camera startCam = m_cameraSet.getCamera();

  m_videoSettings.trajectory.numFrames = m_videoSettings.getTotalFrames();

  // Disable vsync for faster rendering
  m_savedVsync = m_app->isVsync();
  m_app->setVsync(false);

  if(m_videoSettings.enableSBS)
  {
    m_renderSBS           = true;
    m_stereoSeparation    = m_videoSettings.stereoIPD;
    m_stereoConvergence   = m_videoSettings.stereoConvergence;
    m_stereoOffAxisProj   = m_videoSettings.stereoOffAxis;
  }

  m_videoRenderer.startRender(
      m_videoSettings, startCam, {},
      [this](const Camera& cam, int frameIndex)
      {
        m_cameraSet.setCamera(cam, true);

        if(m_videoSettings.enableSBS)
        {
          m_renderSBS = true;
        }
      },
      [this](const std::filesystem::path& framePath)
      {
        // Queue frame save - will be processed next frame after render completes
        m_pendingFramePath = framePath;
        m_pendingFrameSave = true;
      });

  m_videoRenderActive = true;
}

void GaussianSplattingUI::updateVideoRender()
{
  if(!m_videoRenderActive)
    return;

  // Wait for pending frame save before advancing
  if(m_pendingFrameSave)
    return;

  auto progress = m_videoRenderer.getProgress();

  if(progress.state == VideoRenderState::STATE_COMPLETED || progress.state == VideoRenderState::STATE_ERROR ||
     progress.state == VideoRenderState::STATE_CANCELLED)
  {
    m_videoRenderActive = false;

    // Restore vsync
    m_app->setVsync(m_savedVsync);

    if(m_videoSettings.enableSBS)
    {
      m_renderSBS = false;
    }
    return;
  }

  if(progress.state == VideoRenderState::STATE_RENDERING)
  {
    m_videoRenderer.renderNextFrame();
  }
  else if(progress.state == VideoRenderState::STATE_WAITING_FRAMES)
  {
    // Wait for async frame saver to finish before encoding
    if(!m_asyncFrameSaver.hasPendingFrames())
    {
      m_videoRenderer.checkFramesComplete();
    }
  }
}

void GaussianSplattingUI::saveFrameAsync(VkImage srcImage, VkExtent2D size, const std::filesystem::path& path)
{
  VkDevice         device         = m_app->getDevice();
  VkPhysicalDevice physicalDevice = m_app->getPhysicalDevice();
  VkImage          dstImage       = {};
  VkDeviceMemory   dstImageMemory = {};

  bool isHDR = (path.extension() == ".hdr");
  VkFormat format = isHDR ? VK_FORMAT_R32G32B32A32_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM;

  VkCommandBuffer cmd = m_app->createTempCmdBuffer();
  nvvk::imageToLinear(cmd, device, physicalDevice, srcImage, size, dstImage, dstImageMemory, format);
  m_app->submitAndWaitTempCmdBuffer(cmd);

  VkImageSubresource  subResource{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0};
  VkSubresourceLayout subResourceLayout;
  vkGetImageSubresourceLayout(device, dstImage, &subResource, &subResourceLayout);

  const char* data = nullptr;
  vkMapMemory(device, dstImageMemory, 0, VK_WHOLE_SIZE, 0, (void**)&data);
  data += subResourceLayout.offset;

  size_t bytesPerPixel = isHDR ? sizeof(float) * 4 : sizeof(uint8_t) * 4;
  size_t rowSize       = size.width * bytesPerPixel;

  std::vector<uint8_t> pixelData(size.width * size.height * bytesPerPixel);
  for(uint32_t y = 0; y < size.height; y++)
  {
    memcpy(pixelData.data() + y * rowSize, data + y * subResourceLayout.rowPitch, rowSize);
  }

  vkUnmapMemory(device, dstImageMemory);
  vkFreeMemory(device, dstImageMemory, nullptr);
  vkDestroyImage(device, dstImage, nullptr);

  m_asyncFrameSaver.queueFrame(pixelData.data(), size.width, size.height, isHDR, path);
}

void GaussianSplattingUI::guiDrawDepthStreamProperties()
{
  namespace PE = nvgui::PropertyEditor;

  if(ImGui::CollapsingHeader("Depth Streaming", ImGuiTreeNodeFlags_DefaultOpen))
  {
    PE::begin("##Depth Streaming");

    // Connection settings
    static char hostBuffer[256] = "192.168.1.200";
    PE::entry("Host", [&]() {
      return ImGui::InputText("##Host", hostBuffer, sizeof(hostBuffer));
    });

    static int port = 8000;
    PE::entry("Port", [&]() {
      return ImGui::InputInt("##Port", &port, 1, 100, ImGuiInputTextFlags_CharsDecimal);
    });

    static std::filesystem::path videoPath;
    static bool backendConnected = false;
    static bool connectionAttempted = false;
    static bool connectionFailed = false;
    static DepthStreamClient::SessionInfo currentSession;
    static bool uploadInProgress = false;
    static bool uploadFailed = false;

    if(connectionAttempted)
    {
      if(backendConnected)
      {
        ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "✓ Connected to backend");
      }
      else if(connectionFailed)
      {
        ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "✗ Failed to connect to backend");
      }
    }

    if(uploadInProgress)
    {
      ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "⟳ Uploading video...");
    }
    else if(uploadFailed)
    {
      ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "✗ Upload failed");
    }
    else if(m_enableDepthRendering)
    {
      ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "✓ Streaming active - Session: %s", currentSession.sessionId.c_str());
    }

    if(backendConnected)
    {

      PE::entry("Video File", [&]() {
        std::string displayText = videoPath.empty() ? "No file selected" : videoPath.filename().string();
        ImGui::Text("%s", displayText.c_str());

        if(ImGui::Button("Select Video/Image File..."))
        {
          auto path = nvgui::windowOpenFileDialog(m_app->getWindowHandle(), "Select depth video/image file",
                                                  "Video/Image Files|*.mp4;*.avi;*.mov;*.mkv;*.jpg;*.jpeg;*.png|All Files|*.*");
          if(!path.empty())
          {
            videoPath = path;
            return true;
          }
        }
        return false;
      });

      // Upload/Connect button
      PE::entry("Upload Video", [&]() {
        if(m_enableDepthRendering)
        {
          if(ImGui::Button("Disconnect"))
          {
            m_enableDepthRendering = false;
            backendConnected = false;
            uploadFailed = false;
            uploadInProgress = false;
            if (m_depthClient) {
              m_depthClient->disconnectWebSocket();
            }
          }
        }
        else
        {
          if(ImGui::Button("Upload & Start") && !videoPath.empty() && !uploadInProgress)
          {
            uploadInProgress = true;
            uploadFailed = false;

            std::string ext = videoPath.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            bool isImage = (ext == ".jpg" || ext == ".jpeg" || ext == ".png");

            if (isImage)
            {
                // Image processing logic
                std::vector<uint8_t> plyData;
                
                // Use processImagePath if file is local and backend is on localhost, otherwise upload
                bool success = false;
                if (std::string(hostBuffer) == "127.0.0.1" || std::string(hostBuffer) == "localhost")
                {
                    success = m_depthClient->processImagePath(videoPath, plyData);
                }
                else
                {
                    success = m_depthClient->uploadImage(videoPath, plyData);
                }

                if (success && !plyData.empty())
                {
                    // Write PLY data to temp file for loading via existing file-based API
                    std::filesystem::path tempPlyPath = std::filesystem::temp_directory_path() / "temp_gs_image.ply";
                    std::ofstream out(tempPlyPath, std::ios::binary);
                    out.write(reinterpret_cast<const char*>(plyData.data()), plyData.size());
                    out.close();

                    prmScene.sceneToLoadFilename = tempPlyPath;
                    prmScene.addSceneToExisting = false;
                    uploadInProgress = false;
                    
                    LOGI("Loaded 3DGS from image: %s\n", videoPath.string().c_str());
                }
                else
                {
                    LOGE("Failed to process image\n");
                    uploadFailed = true;
                    uploadInProgress = false;
                }
            }
            else
            {
                // Video processing logic
                if (m_depthClient && m_depthClient->uploadVideo(videoPath, currentSession))
                {
                  if (m_depthClient->connectWebSocket(currentSession.sessionId))
                  {
                    enableDepthRendering(hostBuffer, port, videoPath.string());
                    uploadInProgress = false;
                  }
                  else
                  {
                    LOGE("Failed to connect WebSocket\n");
                    uploadFailed = true;
                    uploadInProgress = false;
                  }
                }
                else
                {
                  LOGE("Failed to upload video\n");
                  uploadFailed = true;
                  uploadInProgress = false;
                }
            }
          }
        }
        return false;
      });
    }
    else
    {
      // Connect button
      PE::entry("Connect", [&]() {
        if(ImGui::Button("Connect to Backend"))
        {
          if (!m_depthClient) {
            m_depthClient = std::make_unique<DepthStreamClient>();
          }
          m_depthClient->setBackendAddress(hostBuffer, port);
          connectionAttempted = true;
          backendConnected = m_depthClient->testConnection();
          connectionFailed = !backendConnected;
        }
        return false;
      });
    }

    if(m_enableDepthRendering)
    {
      PE::entry("Depth Scale", [&]() {
        return ImGui::DragFloat("##Scale", &m_depthScale, 0.01f, 0.1f, 10.0f);
      });

      PE::entry("Depth Bias", [&]() {
        return ImGui::DragFloat("##Bias", &m_depthBias, 0.01f, -5.0f, 5.0f);
      });
    }

    PE::end();
  }
}

void GaussianSplattingUI::guiDrawPerformancePanel()
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
void GaussianSplattingUI::onWristButtonPressed()
{
    m_showVrMenu = !m_showVrMenu;
}

void GaussianSplattingUI::onXrInitialized()
{
    // Initialize hand meshes now that XR session is ready with hand trackers
    if (initHandMeshes()) {
        LOGI("Hand meshes initialized successfully\n");
    }
}

bool GaussianSplattingUI::initHandMeshes()
{
    if (!m_xr || !m_xr->handsSupported())
        return false;

    // Define vertex structure for GPU
    struct HandVertex {
        glm::vec3 position;
        glm::vec3 normal;
        glm::vec2 uv;
        glm::ivec4 blendIndices;
        glm::vec4 blendWeights;
    };

    // Initialize each hand mesh
    for (int handIdx = 0; handIdx < 2; ++handIdx) {
        GsOpenXr::Hand hand = (handIdx == 0) ? GsOpenXr::Hand::Left : GsOpenXr::Hand::Right;
        GaussianSplattingUI::HandMeshVk& mesh = (handIdx == 0) ? m_leftHandMesh : m_rightHandMesh;

        XrHandTrackerEXT tracker = m_xr->getHandTracker(hand);
        if (tracker == XR_NULL_HANDLE) {
            LOGW("Hand tracker not available for %s hand\n", hand == GsOpenXr::Hand::Left ? "left" : "right");
            continue;
        }

        // First call to get buffer sizes
        XrHandTrackingMeshFB handMesh{XR_TYPE_HAND_TRACKING_MESH_FB};
        XrResult result = m_xr->getHandMeshFB(tracker, &handMesh);
        if (result != XR_SUCCESS) {
            LOGW("Failed to get hand mesh info for %s hand: %d\n", hand == GsOpenXr::Hand::Left ? "left" : "right", (int)result);
            continue;
        }

        // Allocate CPU buffers
        mesh.positions.resize(handMesh.vertexCountOutput);
        mesh.normals.resize(handMesh.vertexCountOutput);
        mesh.uvs.resize(handMesh.vertexCountOutput);
        mesh.blendIndices.resize(handMesh.vertexCountOutput);
        mesh.blendWeights.resize(handMesh.vertexCountOutput);
        mesh.indices.resize(handMesh.indexCountOutput);

        // Set capacities and pointers for second call
        handMesh.vertexCapacityInput = static_cast<uint32_t>(mesh.positions.size());
        handMesh.indexCapacityInput = static_cast<uint32_t>(mesh.indices.size());
        handMesh.jointCapacityInput = XR_HAND_JOINT_COUNT_EXT;
        handMesh.vertexPositions = mesh.positions.data();
        handMesh.vertexNormals = mesh.normals.data();
        handMesh.vertexUVs = mesh.uvs.data();
        handMesh.vertexBlendIndices = mesh.blendIndices.data();
        handMesh.vertexBlendWeights = mesh.blendWeights.data();
        handMesh.indices = reinterpret_cast<int16_t*>(mesh.indices.data());
        handMesh.jointBindPoses = mesh.jointBindPoses.data();
        handMesh.jointRadii = mesh.jointRadii.data();
        handMesh.jointParents = mesh.jointParents.data();

        // Second call to fill data
        result = m_xr->getHandMeshFB(tracker, &handMesh);
        if (result != XR_SUCCESS) {
            LOGW("Failed to get hand mesh data for %s hand: %d\n", hand == GsOpenXr::Hand::Left ? "left" : "right", (int)result);
            continue;
        }

        // Copy bind poses
        std::copy(handMesh.jointBindPoses, handMesh.jointBindPoses + XR_HAND_JOINT_COUNT_EXT, mesh.jointBindPoses.begin());

        // Create GPU vertex buffer
        std::vector<HandVertex> vertices(handMesh.vertexCountOutput);
        for (uint32_t i = 0; i < handMesh.vertexCountOutput; ++i) {
            vertices[i].position = glm::vec3(mesh.positions[i].x, mesh.positions[i].y, mesh.positions[i].z);
            vertices[i].normal = glm::vec3(mesh.normals[i].x, mesh.normals[i].y, mesh.normals[i].z);
            vertices[i].uv = glm::vec2(mesh.uvs[i].x, mesh.uvs[i].y);
            vertices[i].blendIndices = glm::ivec4(mesh.blendIndices[i].x, mesh.blendIndices[i].y, mesh.blendIndices[i].z, mesh.blendIndices[i].w);
            vertices[i].blendWeights = glm::vec4(mesh.blendWeights[i].x, mesh.blendWeights[i].y, mesh.blendWeights[i].z, mesh.blendWeights[i].w);
        }

        VkDeviceSize vertexBufferSize = vertices.size() * sizeof(HandVertex);
        m_alloc.createBuffer(mesh.vertexBuffer, vertexBufferSize,
                                         VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                         VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        m_uploader.appendBuffer(mesh.vertexBuffer, 0, std::span(vertices));
        NVVK_DBG_NAME(mesh.vertexBuffer.buffer);

        // Create GPU index buffer
        VkDeviceSize indexBufferSize = mesh.indices.size() * sizeof(uint16_t);
        m_alloc.createBuffer(mesh.indexBuffer, indexBufferSize,
                                         VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                         VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        m_uploader.appendBuffer(mesh.indexBuffer, 0, std::span(mesh.indices));
        NVVK_DBG_NAME(mesh.indexBuffer.buffer);

        // Create joint matrices buffer (dynamic, updated each frame)
        // Use CPU-visible memory with persistent mapping for per-frame updates
        VkDeviceSize jointBufferSize = XR_HAND_JOINT_COUNT_EXT * sizeof(glm::mat4);
        m_alloc.createBuffer(mesh.jointMatricesBuffer, jointBufferSize,
                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                         VMA_MEMORY_USAGE_CPU_TO_GPU,
                                         VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
        NVVK_DBG_NAME(mesh.jointMatricesBuffer.buffer);

        mesh.initialized = true;
        LOGI("Initialized hand mesh for %s hand: %d vertices, %d indices\n",
             hand == GsOpenXr::Hand::Left ? "left" : "right", (int)handMesh.vertexCountOutput, (int)handMesh.indexCountOutput);

    }

    // Upload the appended buffer data using a temporary command buffer
    VkCommandBuffer cmd = m_app->createTempCmdBuffer();
    m_uploader.cmdUploadAppended(cmd);
    m_app->submitAndWaitTempCmdBuffer(cmd);

    // Delay rendering for a few frames to ensure all resources are synchronized
    m_handMeshReadyFrameDelay = 5;
    
    return true;
}

void GaussianSplattingUI::destroyHandMeshes()
{
    for (int hand = 0; hand < 2; ++hand) {
        GaussianSplattingUI::HandMeshVk& mesh = (hand == 0) ? m_leftHandMesh : m_rightHandMesh;

        m_alloc.destroyBuffer(mesh.vertexBuffer);
        m_alloc.destroyBuffer(mesh.indexBuffer);
        m_alloc.destroyBuffer(mesh.jointMatricesBuffer);

        mesh.initialized = false;
    }
}

void GaussianSplattingUI::updateHandMeshes()
{
    if (!m_xr || !m_xr->handsSupported())
        return;

    // Helper to convert XrPosef to glm::mat4
    auto poseToMatrix = [](const XrPosef& pose) -> glm::mat4 {
        glm::quat q(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
        glm::vec3 t(pose.position.x, pose.position.y, pose.position.z);
        return glm::translate(glm::mat4(1.0f), t) * glm::mat4_cast(q);
    };

    for (int handIdx = 0; handIdx < 2; ++handIdx) {
        GsOpenXr::Hand hand = (handIdx == 0) ? GsOpenXr::Hand::Left : GsOpenXr::Hand::Right;
        GaussianSplattingUI::HandMeshVk& mesh = (handIdx == 0) ? m_leftHandMesh : m_rightHandMesh;

        if (!mesh.initialized)
            continue;

        const auto& handInput = m_xr->getHandInput(hand);
        if (!handInput.tracked)
            continue;

        // Compute joint matrices following Meta's skinning approach
        // Get wrist as root for relative transforms
        glm::mat4 wristMatrix = poseToMatrix(handInput.jointPoses[XR_HAND_JOINT_WRIST_EXT]);
        glm::mat4 wristInverse = glm::inverse(wristMatrix);

        for (uint32_t i = 0; i < XR_HAND_JOINT_COUNT_EXT; ++i) {
            const XrPosef& currentPose = handInput.jointPoses[i];
            const XrPosef& bindPose = mesh.jointBindPoses[i];

            glm::mat4 currentMatrix = poseToMatrix(currentPose);
            glm::mat4 bindMatrix = poseToMatrix(bindPose);
            glm::mat4 inverseBind = glm::inverse(bindMatrix);

            // Transform relative to wrist, then apply inverse bind pose
            glm::mat4 modelFromRoot = wristInverse * currentMatrix;
            mesh.jointMatrices[i] = modelFromRoot * inverseBind;
        }

        // Upload joint matrices to GPU (persistently mapped buffer, direct memcpy)
        if (mesh.jointMatricesBuffer.mapping) {
            memcpy(mesh.jointMatricesBuffer.mapping, mesh.jointMatrices.data(), XR_HAND_JOINT_COUNT_EXT * sizeof(glm::mat4));
        }
    }
}

void GaussianSplattingUI::renderHandMesh(VkCommandBuffer cmd, const GaussianSplattingUI::HandMeshVk& mesh, const glm::mat4& wristTransform)
{
    if (!mesh.initialized || !mesh.visible)
        return;

    // Validate all required resources exist
    if (m_graphicsPipelineHandMesh == VK_NULL_HANDLE) {
        LOGD("[Hand] renderHandMesh: pipeline is null\n");
        return;
    }
    if (m_descriptorSet == VK_NULL_HANDLE || m_pipelineLayout == VK_NULL_HANDLE) {
        LOGD("[Hand] renderHandMesh: descriptor set or pipeline layout is null\n");
        return;
    }
    if (mesh.jointMatricesBuffer.buffer == VK_NULL_HANDLE) {
        LOGD("[Hand] renderHandMesh: joint matrices buffer is null\n");
        return;
    }
    if (mesh.vertexBuffer.buffer == VK_NULL_HANDLE || mesh.indexBuffer.buffer == VK_NULL_HANDLE) {
        LOGD("[Hand] renderHandMesh: vertex or index buffer is null\n");
        return;
    }

    // Update descriptor set with this hand's joint matrices buffer
    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = mesh.jointMatricesBuffer.buffer;
    bufferInfo.offset = 0;
    bufferInfo.range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = m_descriptorSet;
    write.dstBinding = BINDING_JOINT_MATRICES;
    write.dstArrayElement = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.descriptorCount = 1;
    write.pBufferInfo = &bufferInfo;

    vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);

    // Bind hand mesh pipeline
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipelineHandMesh);

    // Rebind descriptor set after updating (with dynamic offsets for UBO)
    uint32_t dynamicOffsets[] = {0, 0};  // frameInfo UBO offset, indirect buffer offset
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &m_descriptorSet, 2, dynamicOffsets);

    // Enable depth test and write
    vkCmdSetDepthTestEnable(cmd, VK_TRUE);
    vkCmdSetDepthWriteEnable(cmd, VK_TRUE);

    // Push constants for model matrix (wrist transform positions the hand in world space)
    shaderio::PushConstant pc{};
    pc.modelMatrix = wristTransform;
    pc.modelMatrixInverse = glm::inverse(wristTransform);
    pc.modelMatrixRotScaleInverse = glm::inverse(glm::mat4(glm::mat3(wristTransform)));
    pc.objIndex = 0;
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);

    // Bind vertex buffer
    VkBuffer vertexBuffers[] = {mesh.vertexBuffer.buffer};
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffers, offsets);

    // Bind index buffer
    vkCmdBindIndexBuffer(cmd, mesh.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT16);

    // Draw the mesh
    vkCmdDrawIndexed(cmd, static_cast<uint32_t>(mesh.indices.size()), 1, 0, 0, 0);
}

void GaussianSplattingUI::onRenderMultiviewExtra(VkCommandBuffer cmd)
{
    if (!m_xr || !m_xr->handsSupported() || !m_xrInitialized || m_descriptorSet == VK_NULL_HANDLE)
        return;
    if (m_handMeshReadyFrameDelay > 0)
        return;
    if (m_graphicsPipelineHandMeshMultiview == VK_NULL_HANDLE)
        return;

    auto poseToMatrix = [](const XrPosef& pose) -> glm::mat4 {
        glm::quat q(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
        glm::vec3 t(pose.position.x, pose.position.y, pose.position.z);
        return glm::translate(glm::mat4(1.0f), t) * glm::mat4_cast(q);
    };

    const auto& leftHand = m_xr->getHandInput(GsOpenXr::Hand::Left);
    if (leftHand.tracked) {
        glm::mat4 wristTransform = poseToMatrix(leftHand.jointPoses[XR_HAND_JOINT_WRIST_EXT]);
        renderHandMeshMultiview(cmd, m_leftHandMesh, wristTransform);
    }
    else if (m_leftHandMesh.initialized && m_debugForceRenderHands) {
        glm::mat4 debugTransform = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -0.5f));
        renderHandMeshMultiview(cmd, m_leftHandMesh, debugTransform);
    }

    const auto& rightHand = m_xr->getHandInput(GsOpenXr::Hand::Right);
    if (rightHand.tracked) {
        glm::mat4 wristTransform = poseToMatrix(rightHand.jointPoses[XR_HAND_JOINT_WRIST_EXT]);
        renderHandMeshMultiview(cmd, m_rightHandMesh, wristTransform);
    }
    else if (m_rightHandMesh.initialized && m_debugForceRenderHands) {
        glm::mat4 debugTransform = glm::translate(glm::mat4(1.0f), glm::vec3(0.2f, 0.0f, -0.5f));
        renderHandMeshMultiview(cmd, m_rightHandMesh, debugTransform);
    }
}

void GaussianSplattingUI::renderHandMeshMultiview(VkCommandBuffer cmd, const GaussianSplattingUI::HandMeshVk& mesh, const glm::mat4& wristTransform)
{
    if (!mesh.initialized || !mesh.visible)
        return;

    if (m_graphicsPipelineHandMeshMultiview == VK_NULL_HANDLE)
        return;
    if (m_descriptorSet == VK_NULL_HANDLE || m_pipelineLayout == VK_NULL_HANDLE)
        return;
    if (mesh.jointMatricesBuffer.buffer == VK_NULL_HANDLE)
        return;
    if (mesh.vertexBuffer.buffer == VK_NULL_HANDLE || mesh.indexBuffer.buffer == VK_NULL_HANDLE)
        return;

    // Update descriptor set with this hand's joint matrices buffer
    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = mesh.jointMatricesBuffer.buffer;
    bufferInfo.offset = 0;
    bufferInfo.range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = m_descriptorSet;
    write.dstBinding = BINDING_JOINT_MATRICES;
    write.dstArrayElement = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.descriptorCount = 1;
    write.pBufferInfo = &bufferInfo;

    vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);

    // Bind hand mesh multiview pipeline
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipelineHandMeshMultiview);

    // Rebind descriptor set after updating (with dynamic offsets for UBO)
    uint32_t dynamicOffsets[] = {m_lastFrameInfoOffset, 0};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &m_descriptorSet, 2, dynamicOffsets);

    // Enable depth test and write
    vkCmdSetDepthTestEnable(cmd, VK_TRUE);
    vkCmdSetDepthWriteEnable(cmd, VK_TRUE);

    // Push constants for model matrix
    shaderio::PushConstant pc{};
    pc.modelMatrix = wristTransform;
    pc.modelMatrixInverse = glm::inverse(wristTransform);
    pc.modelMatrixRotScaleInverse = glm::inverse(glm::mat4(glm::mat3(wristTransform)));
    pc.objIndex = 0;
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);

    // Bind vertex buffer
    VkBuffer vertexBuffers[] = {mesh.vertexBuffer.buffer};
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffers, offsets);

    // Bind index buffer
    vkCmdBindIndexBuffer(cmd, mesh.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT16);

    // Draw the mesh
    vkCmdDrawIndexed(cmd, static_cast<uint32_t>(mesh.indices.size()), 1, 0, 0, 0);
}
#endif

// namespace vk_gaussian_splatting (continued)


void GaussianSplattingUI::guiDrawFileDialog()
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

void GaussianSplattingUI::guiDrawSupersplatDialog()
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

void GaussianSplattingUI::createTextureFromRGBA(const std::vector<uint8_t>& data, int width, int height, nvvk::Image& texture, VkImageView& view)
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

void GaussianSplattingUI::guiDrawVrMenu()
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

} // namespace vk_gaussian_splatting
