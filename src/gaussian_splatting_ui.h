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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _GAUSSIAN_SPLATTING_UI_H_
#define _GAUSSIAN_SPLATTING_UI_H_

// Include winsock2 first to avoid winsock.h conflicts
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#endif

#include <iostream>
#include <string>
#include <array>
#include <chrono>
#include <filesystem>
#include <span>
// TODO: include Igmlui before Vulkan
// Or undef Status before including imgui
// need to solve this issue
#include <imgui/imgui.h>
//
#include <vulkan/vulkan_core.h>
// mathematics
#include <glm/vec3.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/transform.hpp>
// threading
#include <thread>
#include <condition_variable>
#include <mutex>
// GPU radix sort (header-only library, uses volk)
#define VRDX_USE_VOLK
#include <vk_radix_sort.h>
//
#include <nvvk/context.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/pipeline.hpp>

#include <nvutils/logger.hpp>
#include <nvutils/file_operations.hpp>
#include <nvutils/alignment.hpp>

#include <nvvk/helpers.hpp>
#include <nvvk/gbuffers.hpp>
#include <nvvk/resources.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/staging.hpp>
#include <nvvk/validation_settings.hpp>
#include <nvvk/sampler_pool.hpp>
#include <nvvk/default_structs.hpp>
#include <nvvk/profiler_vk.hpp>
#include <nvvk/acceleration_structures.hpp>
#include <nvvk/descriptors.hpp>
#include <nvvk/sbt_generator.hpp>

#include <nvvkglsl/glsl.hpp>

#include <nvapp/application.hpp>
#include <nvapp/elem_camera.hpp>
#include <nvapp/elem_profiler.hpp>
#include <nvapp/elem_sequencer.hpp>
#include <nvapp/elem_default_title.hpp>
#include <nvapp/elem_default_menu.hpp>
//
#include <nvgui/axis.hpp>
#include <nvgui/enum_registry.hpp>
#include <nvgui/property_editor.hpp>
#include <nvgui/file_dialog.hpp>
//
#include <nvgpu_monitor/elem_gpu_monitor.hpp>

// Shared between host and device
#include "shaderio.h"

#include "utilities.h"
#include "splat_set.h"
#include "splat_set_vk.h"
#include "splat_loader_async.h"
#include "splat_sorter_async.h"
#include "mesh_set_vk.h"
#include "light_set_vk.h"
#include "camera_set.h"
#include "camera_trajectory.h"
#include "video_renderer.h"
#include "gaussian_splatting.h"
#include "async_frame_saver.h"

#ifdef WITH_COMFYUI
#include "comfyui_client.h"
#endif

// Json
#include <tinygltf/json.hpp>
using nlohmann::json;

#include "perf_stats.h"

#include "perf_stats.h"

namespace vk_gaussian_splatting {

class GaussianSplattingUI : public GaussianSplatting
{
public:  // Methods specializing IAppElement
  GaussianSplattingUI(nvutils::ProfilerManager* profilerManager, nvutils::ParameterRegistry* parameterRegistry, bool* benchmarkEnabled);

  ~GaussianSplattingUI() override;

  void onAttach(nvapp::Application* app) override;

  void onDetach() override;

  void onResize(VkCommandBuffer cmd, const VkExtent2D& size) override;

  void onPreRender() override;

  void onRender(VkCommandBuffer cmd) override;

  void onUIRender();

  void onUIMenu();

  void onFileDrop(const std::filesystem::path& filename);

  // handle recent files save/load at imgui level
  void guiRegisterIniFileHandlers();

private:
  void guiDrawAssetsWindow(void);
  void guiDrawRendererTree();
  void guiDrawCameraTree();
  void guiDrawLightTree();
  void guiDrawRadianceFieldsTree();
  void guiDrawObjectTree();
  void guiDrawDepthStreamTree();

  void guiDrawVideoExportWindow();
  void startVideoRender();
  void updateVideoRender();

  void guiDrawPropertiesWindow(void);
  void guiDrawRendererProperties();
  void guiDrawSplatSetProperties();
  void guiDrawMeshTransformProperties();
  void guiDrawMeshMaterialProperties();
  void guiDrawCameraProperties();
  void guiDrawNavigationProperties();
  void guiDrawLightProperties();
  void guiDrawDepthStreamProperties();
  void guiDrawPerformancePanel();

  void guiDrawRendererStatisticsWindow();

  void guiDrawMemoryStatisticsWindow(void);

  void guiDrawFooterBar(void);

  bool guiGetTransform(glm::vec3& scale, glm::vec3& rotation, glm::vec3& translation, glm::mat4& transform, glm::mat4& transformInv, bool disabled /*=false*/);

  // methods to handle recent files in file menu
  void guiAddToRecentFiles(std::filesystem::path filePath, int historySize = 20);
  void guiAddToRecentProjects(std::filesystem::path filePath, int historySize = 20);

  bool loadProjectIfNeeded();
  bool saveProject(std::string path);

private:
  // hide/show ui elements
  bool                                                  m_showUI = true;
  PerfStats                                             m_perfStats;
  std::shared_ptr<nvapp::ElementProfiler::ViewSettings> m_profilerViewSettings;

  // benchmark mode (enabled by command line), loadings will be synchronous and vsync off
  bool* m_pBenchmarkEnabled = {};
  // screenshot file name (used by benchmark)
  std::filesystem::path m_screenshotFilename;

  // Recent files list
  std::vector<std::filesystem::path> m_recentFiles;

  // Recent projects list
  std::vector<std::filesystem::path> m_recentProjects;

  // for multiple choice selectors in the UI
  enum GuiEnums
  {
    GUI_STORAGE,              // model storage in VRAM (in texture or buffer)
    GUI_SORTING,              // the sorting method to use
    GUI_PIPELINE,             // the rendering pipeline to use
    GUI_CAMERA_TYPE,          // type of camera
    GUI_FRUSTUM_CULLING,      // where to perform frustum culling (or disabled)
    GUI_SH_FORMAT,            // data format for storage of SH in VRAM
    GUI_PARTICLE_FORMAT,      // Particle tracing mode for RTX
    GUI_KERNEL_DEGREE,        // Kernel degree for RTX
    GUI_VISUALIZE,            // visualization mode
    GUI_ILLUM_MODEL,          // TODO rename, "illumination" model is not the proper name
    GUI_DIST_SHADER_WG_SIZE,  // Distance shader workgroup size
    GUI_MESH_SHADER_WG_SIZE,  // Mesh shader workgroup size
    GUI_RAY_HIT_PER_PASS,     // Max number of ray hits stored per pass (payload array size)
    GUI_TEMPORAL_SAMPLING,    // Temporal sampling mode
    GUI_LIGHT_TYPE,           // Type of light
    GUI_EXTENT_METHOD         // extent projection method
  };

  // UI utility for choice (a.k.a. "combo") menus
  nvgui::EnumRegistry m_ui;

  // which property to display in the property editor
  enum
  {
    GUI_NONE,
    GUI_RENDERER,
    GUI_CAMERA,
    GUI_LIGHT,
    GUI_SPLATSET,
    GUI_MESH,
    GUI_DEPTH_STREAM,
  } m_selectedAsset = GUI_RENDERER;

  bool m_showDepthPerformance{false};
  bool m_objListUpdated = false;
  const float TREE_INDENT      = 16.0f;
  
  // Radiance field deletion request
  bool   m_requestDeleteRadianceField = false;
  size_t m_radianceFieldToDelete = 0;

  // Reload queue for multiple files
  std::vector<std::filesystem::path> m_reloadQueue;
  bool                               m_isReloading = false;

  // Project loading
  bool loadingProject = false;
  json data;

  // Debuging
  void dumpSplat(uint32_t splatIdx);

  // Video Export
  bool                  m_showSuperSplatUrlPopup = false;
  bool                  m_showVideoExportWindow = false;
  VideoRenderer         m_videoRenderer;
  VideoRenderSettings   m_videoSettings;
  bool                  m_videoRenderActive   = false;
  bool                  m_pendingFrameSave    = false;
  std::filesystem::path m_pendingFramePath;
  bool                  m_savedVsync          = true;
  AsyncFrameSaver       m_asyncFrameSaver;

  void saveFrameAsync(VkImage srcImage, VkExtent2D size, const std::filesystem::path& path);

#ifdef WITH_COMFYUI
  // ComfyUI integration
  void guiDrawComfyUIWindow();
  void onComfyUIWorkflowComplete(const ComfyUIClient::WorkflowResult& result);

  std::unique_ptr<ComfyUIClient> m_comfyClient;
  bool                           m_showComfyUIWindow = true;
  char                           m_comfyPrompt[4096] = "a cute cat sitting on a table, photorealistic, 8k";
  char                           m_comfyNegativePrompt[2048] = "ugly, low quality, blurry";
  char                           m_comfyHost[256] = "192.168.1.200";
  int                            m_comfyPort = 8188;
  std::filesystem::path          m_comfyWorkflowPath = "S:\\ComfyUI\\user\\default\\workflows\\z_image_turbo_3d.json";
  std::string                    m_comfyStatusMessage;
#endif

  // Hand mesh rendering
  struct HandMeshVk {
    // CPU mesh data from OpenXR
    std::vector<XrVector3f> positions;
    std::vector<XrVector3f> normals;
    std::vector<XrVector2f> uvs;
    std::vector<XrVector4sFB> blendIndices;
    std::vector<XrVector4f> blendWeights;
    std::vector<uint16_t> indices;

    // Vulkan GPU resources
    VkBuffer vertexBuffer = VK_NULL_HANDLE;
    VkBuffer indexBuffer = VK_NULL_HANDLE;
    VkBuffer jointMatricesBuffer = VK_NULL_HANDLE;
    VmaAllocation vertexAllocation = VK_NULL_HANDLE;
    VmaAllocation indexAllocation = VK_NULL_HANDLE;
    VmaAllocation jointAllocation = VK_NULL_HANDLE;

    // Joint matrices for skinning (updated each frame)
    std::array<glm::mat4, XR_HAND_JOINT_COUNT_EXT> jointMatrices{};

    // Rendering state
    bool initialized = false;
    bool visible = true;
  };

  HandMeshVk m_leftHandMesh;
  HandMeshVk m_rightHandMesh;

  // Hand mesh rendering functions
  bool initHandMeshes();
  void destroyHandMeshes();
  void updateHandMeshes();
  void renderHandMesh(VkCommandBuffer cmd, const GaussianSplattingUI::HandMeshVk& mesh, const glm::mat4& viewProj);

  // Wrist button for file picker
  struct WristButton {
    bool visible = true;
    bool pressed = false;
    float radius = 0.03f;
    glm::vec3 localOffset{0.0f, 0.0f, 0.05f};
  } m_wristButton;

  // File picker state
  bool m_showFilePicker = false;

  // Wrist button handler
  void onWristButtonPressed() override;
};

}  // namespace vk_gaussian_splatting

#endif
