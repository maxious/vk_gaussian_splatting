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

#ifndef _VK_VIEWER_UI_H_
#define _VK_VIEWER_UI_H_

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
#include <unordered_map>
#include <memory>
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
#include "vk_viewer.h"

#ifdef WITH_COMFYUI
#include "comfyui_client.h"
#endif

#include "supersplat_client.h"

// Json
#include <tinygltf/json.hpp>
using nlohmann::json;

#include "perf_stats.h"

#include "perf_stats.h"

namespace vk_viewer {

class VkViewerUI : public VkViewer
{
public:  // Methods specializing IAppElement
  VkViewerUI(nvutils::ProfilerManager* profilerManager, nvutils::ParameterRegistry* parameterRegistry, bool* benchmarkEnabled);

  ~VkViewerUI() override;

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
  // Auto-screenshot: delay in seconds before taking screenshot and exiting
  float m_autoScreenshotDelay = 0.0f;
  float m_autoScreenshotTimer = 0.0f;
  bool  m_autoScreenshotPending = false;
  int   m_autoScreenshotExitCountdown = -1;  // Frames to wait after screenshot before exit

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

#ifdef WITH_OPENXR
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
    nvvk::Buffer vertexBuffer;
    nvvk::Buffer indexBuffer;
    nvvk::Buffer jointMatricesBuffer;

    // Joint matrices for skinning (updated each frame)
    std::array<glm::mat4, XR_HAND_JOINT_COUNT_EXT> jointMatrices{};
    // Bind poses from the mesh
    std::array<XrPosef, XR_HAND_JOINT_COUNT_EXT> jointBindPoses{};
    std::array<float, XR_HAND_JOINT_COUNT_EXT> jointRadii{};
    std::array<XrHandJointEXT, XR_HAND_JOINT_COUNT_EXT> jointParents{};

    // Rendering state
    bool initialized = false;
    bool visible = true;
  };

  HandMeshVk m_leftHandMesh;
  HandMeshVk m_rightHandMesh;
  bool m_debugForceRenderHands = true;  // Debug: force render hands even when not tracked
  int m_handMeshReadyFrameDelay = 0;    // Delay rendering for a few frames after init

  // Hand mesh rendering functions
  bool initHandMeshes();
  void destroyHandMeshes();
  void updateHandMeshes();
  void renderHandMesh(VkCommandBuffer cmd, const VkViewerUI::HandMeshVk& mesh, const glm::mat4& wristTransform);
  void onRenderMultiviewExtra(VkCommandBuffer cmd) override;
  void renderHandMeshMultiview(VkCommandBuffer cmd, const VkViewerUI::HandMeshVk& mesh, const glm::mat4& wristTransform);

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
  void onXrInitialized() override;
#endif

  void guiDrawFileDialog();
  void guiDrawSupersplatDialog();
  void createTextureFromRGBA(const std::vector<uint8_t>& data, int width, int height, nvvk::Image& texture, VkImageView& view);

  std::unique_ptr<SupersplatClient> m_supersplatClient;
  bool m_showFileDialog = false;
  bool m_showSupersplatDialog = false;
  std::vector<std::string> m_fileList;
  std::vector<SupersplatClient::Scene> m_supersplatScenes;
  std::string m_supersplatSearch;
  std::unordered_map<std::string, nvvk::Image> m_thumbnailTextures;
  std::unordered_map<std::string, VkImageView> m_thumbnailViews;
  std::unordered_map<std::string, VkDescriptorSet> m_thumbnailDescriptors;
  std::mutex m_thumbnailMutex;
  struct PendingThumbnail { std::string url; std::vector<uint8_t> data; int w, h; };
  std::vector<PendingThumbnail> m_pendingThumbnails;

  bool m_showVrMenu = false;
  void guiDrawVrMenu();
};

}  // namespace vk_viewer

#endif
