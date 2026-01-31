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

#ifndef _VK_VIEWER_H_
#define _VK_VIEWER_H_

#include "depth_stream_client.h"
#include "depth_to_vk.h"
#include "backend_process_manager.h"
#ifdef WITH_VIDEO_DECODER
#include "video_decoder.h"
#include "depth_video_loader.h"
#include "hls_depth_player.h"
#endif

#include <iostream>

#include <string>
#include <array>
#include <chrono>
#include <filesystem>
#include <span>
// Important: include Igmlui before Vulkan
// Or undef "Status" before including imgui
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
#include <nvutils/logger.hpp>
#include <nvutils/file_operations.hpp>
#include <nvutils/alignment.hpp>

#include <nvvk/context.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/pipeline.hpp>
#include <nvvk/physical_device.hpp>
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
#include <nvslang/slang.hpp>

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

#include "parameters.h"
#include "utilities.h"
#include "splat_set.h"
#include "splat_set_vk.h"
#include "splat_loader_async.h"
#include "splat_sorter_async.h"
#include "mesh_set_vk.h"
#include "light_set_vk.h"
#include "vdz_mesh.h"
#include "camera_set.h"
#include "render_context.h"
#include "lcc_tile_manager.h"

#ifdef WITH_OPENXR
#include "gs_openxr.hpp"
#endif

#ifdef WITH_DLSS_RR
#include "gs_dlss_rr.hpp"
#endif

namespace vk_viewer {

// Forward declarations for animation
class AnimationController;
class AnimationUI;




class VkViewer : public nvapp::IAppElement
{
public:
  // Benchmarking, print extended info
  // invoked by parameter sequencer
  void benchmarkAdvance();

#ifdef WITH_OPENXR
  // Query OpenXR required Vulkan extensions (call before Vulkan context creation)
  bool queryOpenXrVulkanExtensions(std::vector<std::string>& outInstanceExtensions,
                                   std::vector<std::string>& outDeviceExtensions);
#endif

    void enableDepthRendering(const std::string& host, int port, const std::string& videoPath);
    void enableDepthVideoPlayback(const std::string& metadataPath);
    void enableHlsPlayback(const std::string& hlsPlaylistPath);
    void updateDepthRendering(VkCommandBuffer cmd);
    bool isDepthVideoPlaying() const;

  public:
    // Camera manipulator


  // public so that it can be accessed by main
  std::shared_ptr<nvutils::CameraManipulator> cameraManip{};

  // VDZ mesh accessor for hybrid rendering mode switching
  VdzMesh& getVdzMesh() { return m_vdzMesh; }

protected:
  VkViewer(nvutils::ProfilerManager* profilerManager, nvutils::ParameterRegistry* parameterRegistry);

  virtual ~VkViewer();

  virtual void onAttach(nvapp::Application* app);

  virtual void onDetach();

  virtual void onResize(VkCommandBuffer cmd, const VkExtent2D& size);

  virtual void onPreRender();

  // reset frame counter for temporal accumulated multi-sampling
  // will cause a restart of the frame construction
  inline void resetFrameCounter() { prmFrame.frameSampleId = -1; }

  virtual void onRender(VkCommandBuffer cmd);

  // reset the rendering settings that can
  // be modified by the user interface
  inline void resetRenderSettings()
  {
    resetFrameParameters();
    resetRenderParameters();
    resetRasterParameters();
    resetRtxParameters();
  }

  // Initializes all that is related to the scene based
  // on current parameters. VRAM Data, shaders, pipelines.
  // Invoked on scene load success.
  bool initAll();

  // Denitializes all that is related to the scene.
  // VRAM Data, shaders, pipelines.
  // Invoked on scene close or on exit.
  void deinitAll();

  // free scene (splat set) from RAM
  void deinitScene();

  // LCC tiled streaming methods
  bool initializeLccStreaming(const std::filesystem::path& scenePath);
  void shutdownLccStreaming();
  void updateLccStreaming(const glm::mat4& viewProj, const glm::vec3& cameraPos, float dt);
  bool isLccStreamingActive() const { return m_lccTileManager != nullptr && m_lccTileManager->isActive(); }

#ifdef WITH_OPENXR
protected:
  // Initialize OpenXR runtime and session
  void initializeOpenXR();
  // Shutdown OpenXR
  void shutdownOpenXR();
  // Process controller input for locomotion
  void updateXrLocomotion(float deltaTime);

  // Virtual function for derived classes to handle wrist button
  virtual void onWristButtonPressed() {}
  // Virtual function called after XR is initialized (for hand mesh init, etc.)
  virtual void onXrInitialized() {}

  // Locomotion settings
  float m_xrMoveSpeed       = 2.0f;   // meters per second
  float m_xrSprintMultiplier = 2.5f;  // speed multiplier when sprinting
  float m_xrSnapTurnAngle   = 45.0f;  // degrees per snap turn
  float m_xrSmoothTurnSpeed = 90.0f;  // degrees per second for smooth turn
  bool  m_xrUseSmoothTurn   = false;  // use smooth turn instead of snap turn

  // Hand drag state for pinch-to-drag scene movement
  struct HandDragState {
    bool active = false;
    GsOpenXr::Hand hand;
    glm::vec3 grabStartHandPos{0};
    glm::vec3 grabStartWorldOffset{0};
  } m_handDrag;



private:
  // Copy rendered image to XR swapchain
  void copyToXrSwapchain(VkCommandBuffer cmd);

  std::chrono::steady_clock::time_point m_xrLastFrameTime;
  bool  m_xrFirstFrame      = true;
#endif

private:
  bool m_attached = false;
  // init the raster pipelines
  void initPipelines();

  // deinit raster and rtx pipelines TODO move rtx in separate method
  void deinitPipelines();

  void initRendererBuffers();

  void deinitRendererBuffers();

  // Chunk-based hierarchical frustum culling
  void computeChunkBounds();                           // Compute chunk AABBs on CPU after loading
  void initChunkCullingBuffers();                      // Create GPU buffers for chunk culling
  void deinitChunkCullingBuffers();                    // Destroy chunk culling buffers
  void initChunkCullingPipeline();                     // Create chunk culling compute pipeline
  void deinitChunkCullingPipeline();                   // Destroy chunk culling pipeline
  void extractFrustumPlanes(const glm::mat4& viewProj); // Extract 6 frustum planes from viewProj matrix
  void dispatchChunkCulling(VkCommandBuffer cmd);      // Dispatch chunk culling compute shader

  void updateSlangMacros(void);

  bool compileSlangShader(const std::string& filename, VkShaderModule& module);

  bool initShaders(void);

  void deinitShaders(void);

  /////////////
  // Rendering orchestration (refactored from monolithic onRender)
  
  // Frame lifecycle helpers
  bool beginFrame(FrameRenderContext& ctx);
  void buildContentState(FrameRenderContext& ctx);
  void buildViews(FrameRenderContext& ctx);
  void prepareSceneForFrame(FrameRenderContext& ctx);
  void clearMainTargets(FrameRenderContext& ctx);
  void renderMultiviewPath(FrameRenderContext& ctx);
  void renderPerViewPath(FrameRenderContext& ctx);
  void renderSingleView(FrameRenderContext& ctx, const RenderView& view);
  void finalizeFrame(FrameRenderContext& ctx);
  
  // RTX-specific frame paths
  void renderRtxFrame(FrameRenderContext& ctx);
  void renderRtxStereoFrame(FrameRenderContext& ctx);

  /////////////
  // Rendering submethods

  // process eventual update requests comming from UI or benchmark
  // that requires to be performed before a new rendering after a DeviceWaitIdle
  void processUpdateRequests(void);

  // Updates frame information uniform buffer and frame camera info
  void updateAndUploadFrameInfoUBO(VkCommandBuffer cmd, const uint32_t splatCount);
  void updateAndUploadFrameInfoUBO(VkCommandBuffer cmd, const uint32_t splatCount, const glm::mat4& view,
                                   const glm::mat4& proj, const glm::vec3& eye, const glm::vec2& viewport,
                                   const glm::vec2& viewportOffset = glm::vec2(0.0f),
                                   const glm::vec2& stereoShift = glm::vec2(0.0f));

  void tryConsumeAndUploadCpuSortingResult(VkCommandBuffer cmd, const uint32_t splatCount);

  void processSortingOnGPU(VkCommandBuffer cmd, const uint32_t splatCount, const glm::mat4& viewMatrix, bool skipRadixSort = false);

  void drawSplatPrimitives(VkCommandBuffer cmd, const uint32_t splatCount, const glm::mat4* viewMatrix = nullptr);

  void drawMeshPrimitives(VkCommandBuffer cmd);

  void drawVdzMesh(VkCommandBuffer cmd);

  // for statistics display in the UI
  // copy form m_indirectReadbackHost updated at previous frame to m_indirectReadback
  void collectReadBackValuesIfNeeded(void);
  // for statistics display in the UI
  // read back updated indirect parameters from m_indirect into m_indirectReadbackHost
  void readBackIndirectParametersIfNeeded(VkCommandBuffer cmd);

  void updateRenderingMemoryStatistics(VkCommandBuffer cmd, const uint32_t splatCount);

  //////////////
  // RTX specific

  // updates the frame counter and returns true if a new raytracing pass is needed
  bool updateFrameCounter();

  void initRtDescriptorSet();
  void updateRtDescriptorSet();
  void initRtPipeline();
  void raytrace(const VkCommandBuffer& cmdBuf, bool meshDepthOnly = false,
                glm::ivec2 viewportOffset = {0, 0}, glm::ivec2 viewportSize = {0, 0});
  // VK_KHR_multiview optimized raytrace for stereo rendering (mobile VR)
  void raytraceMultiview(const VkCommandBuffer& cmdBuf, bool meshDepthOnly,
                         const glm::mat4& leftViewMat, const glm::mat4& leftProjMat,
                         const glm::mat4& rightViewMat, const glm::mat4& rightProjMat,
                         const glm::vec3& leftEyePos, const glm::vec3& rightEyePos,
                         glm::ivec2 viewportSize = {0, 0});

  //////////////
  // Post processing

  void initDescriptorSetPostProcessing();
  void updateDescriptorSetPostProcessing();
  void initPipelinePostProcessing();
  void postProcess(VkCommandBuffer cmd);



protected:
  // Loaded radiance fields metadata (each file appears as a separate entry)
  std::vector<RadianceFieldEntry> m_radianceFields;
  
  // Helper to get first loaded scene filename (for backward compatibility)
  std::filesystem::path getLoadedSceneFilename() const
  {
    return m_radianceFields.empty() ? std::filesystem::path{} : m_radianceFields[0].filename;
  }
  
  // Pending filename being loaded (stored during async load)
  std::filesystem::path m_pendingLoadFilename;

  // scene loader
  SplatLoaderAsync m_splatLoader;
  // Temporary storage for newly loaded splat data before merging
  SplatSet m_splatSetPending = {};
  // 3DGS/3DGRT model in RAM (merged from all radiance fields)
  SplatSet m_splatSet = {};
  // 3DGS/3DGRT model in VRAM
  SplatSetVk m_splatSetVk = {};
  // Set of meshes in VRAM
  MeshSetVk m_meshSetVk = {};
  // VDZ depth-displaced mesh for depth visualization
  VdzMesh m_vdzMesh = {};
  // Set of lights in RAM and VRAM
  LightSetVk m_lightSet = {};
  // Set of cameras in RAM
  CameraSet m_cameraSet = {};

  // LCC tiled streaming manager (nullptr when not streaming)
  std::unique_ptr<LccTileManager> m_lccTileManager;
  // LCC metadata (scale ranges for GPU decompression)
  LccLoader::LccMeta m_lccMeta{};
  // Raw packed LCC data for GPU-side decompression
  std::vector<uint8_t> m_lccPackedData;
  uint32_t m_lccPackedSplatCount = 0;
  // Temporary SplatSet for streaming (filled each frame with visible tiles)
  SplatSet m_streamingSplatSet;

  // Index of the item selected in a root node of scene graph or -1 if none
  int64_t m_selectedItemIndex = -1;
  // Index of the last camera loaded
  uint64_t m_lastLoadedCamera = 0;

  // Push constant for rasterizer
  shaderio::PushConstant m_pcRaster{};

  // counting benchmark steps
  int m_benchmarkId = 0;

  // trigger a rebuild of the data in VRAM (textures or buffers) at next frame
  // also triggers shaders and pipeline rebuild
  bool m_requestUpdateSplatData = false;
  // trigger a rebuild of the splat set RTX Acceleration Structure at next frame
  bool m_requestUpdateSplatAs = false;
  // request delayed update of Acceleration Structures if not using ray tracing
  bool m_requestDelayedUpdateSplatAs = false;
  // trigger a rebuild of the shaders and pipelines at next frame
  bool m_requestUpdateShaders = false;
  // trigger the reinit of mesh acceleration structures at next frame
  bool m_requestUpdateMeshData = false;
  // trigger the update of light buffer at next frame
  bool m_requestUpdateLightsBuffer = false;
  // trigger the deletion of the selected mesh object
  bool m_requestDeleteSelectedMesh = false;

  // SBS Stereo
  bool  m_renderSBS           = false;
  float m_stereoSeparation    = 0.063f;  // 63mm default IPD (in meters)
  float m_stereoConvergence   = 1.0f;    // Convergence distance in meters (where stereo images overlap)
  bool  m_stereoOffAxisProj   = true;    // Use off-axis (asymmetric) frustum projection

  // OpenXR HMD support
#ifdef WITH_OPENXR
  std::unique_ptr<GsOpenXr> m_xr;
  bool  m_useXrHmd        = false;  // Enable XR HMD rendering
  bool  m_xrInitialized   = false;  // Whether XR was successfully initialized
  bool  m_xrResizedThisFrame = false;  // Skip rendering frame after XR GBuffer resize
  VkImage m_xrColorImage  = VK_NULL_HANDLE;  // Current XR color swapchain image
  VkImage m_xrDepthImage  = VK_NULL_HANDLE;  // Current XR depth swapchain image
  VkImage m_xrMotionImage = VK_NULL_HANDLE;  // Current XR motion swapchain image

  
  // Multiview resources for stereo rendering (2-layer images)
  nvvk::Image   m_xrMultiviewColor{};       // 2-layer color image for multiview
  nvvk::Image   m_xrMultiviewDepth{};       // 2-layer depth image for multiview
  VkImageView   m_xrMultiviewColorView = VK_NULL_HANDLE;  // Array view (both layers)
  VkImageView   m_xrMultiviewDepthView = VK_NULL_HANDLE;  // Array view (both layers)
  VkExtent2D    m_xrMultiviewExtent{};      // Per-eye extent for multiview
  bool          m_xrMultiviewInitialized = false;
  
  // Environment Depth resources
  // Map from swapchain index to image view
  std::map<uint32_t, VkImageView> m_envDepthImageViews;
  
  void initXrMultiviewResources(VkCommandBuffer cmd, VkExtent2D perEyeExtent);
  void deinitXrMultiviewResources();
  void renderMultiviewRaster(VkCommandBuffer cmd, uint32_t splatCount);
  
  // Virtual hook for rendering additional content in the multiview render pass
  // Called after splats are rendered but before vkCmdEndRendering
  virtual void onRenderMultiviewExtra(VkCommandBuffer cmd) {}
#endif

  nvapp::Application*         m_app{nullptr};
  nvutils::ProfilerManager*   m_profilerManager;
  nvutils::ParameterRegistry* m_parameterRegistry;
  nvvk::StagingUploader       m_uploader{};     // utility to upload buffers to device
  nvvk::SamplerPool           m_samplerPool{};  // The sampler pool, used to create texture samplers
  VkSampler                   m_sampler{};      // texture sampler (nearest)
  nvvk::ResourceAllocator     m_alloc;
  nvvk::PhysicalDeviceInfo    m_physicalDeviceInfo;

  nvutils::ProfilerTimeline* m_profilerTimeline{};
  nvvk::ProfilerGpuTimer     m_profilerGpuTimer;

  glm::vec2         m_viewSize    = {0, 0};
  VkFormat          m_colorFormat = VK_FORMAT_R8G8B8A8_UNORM;    // Color format of the image
  VkFormat          m_depthFormat = VK_FORMAT_UNDEFINED;         // Depth format of the depth buffer
  VkClearColorValue m_clearColor  = {{0.0F, 0.0F, 0.0F, 0.0F}};  // Clear color
  VkDevice          m_device      = VK_NULL_HANDLE;              // Convenient sortcut to device

  // Convenient enum to dereference color buffers in GBuffers
  enum
  {
    COLOR_MAIN = 0,
    COLOR_AUX1 = 1,
    COLOR_MOTION = 2,
#ifdef WITH_DLSS_RR
    // DLSS-RR G-buffer outputs (at render resolution)
    COLOR_DLSS_DIFFUSE_ALBEDO  = 3,   // RGB diffuse albedo
    COLOR_DLSS_SPECULAR_ALBEDO = 4,   // RGB specular albedo
    COLOR_DLSS_NORMAL_ROUGH    = 5,   // RGB normal + A roughness
    COLOR_DLSS_MOTION          = 2,   // RG motion vectors (aliased to COLOR_MOTION)
    COLOR_DLSS_LINEAR_DEPTH    = 6,   // R linear depth
    COLOR_DLSS_SPEC_HIT_DIST   = 7,   // R specular hit distance
    COLOR_DLSS_OUTPUT          = 8,   // DLSS-RR output (at output resolution)
#endif
  };

  // G-Buffers: 2 color buffers + 1 depth buffer (+ DLSS-RR buffers when enabled)
  nvvk::GBuffer m_gBuffers;

#ifdef WITH_DLSS_RR
  // DLSS-RR denoiser
  std::unique_ptr<NgxContext> m_ngxContext;
  std::unique_ptr<GsDlssRR>   m_dlssRR;
  bool                        m_dlssRREnabled     = false;  // User toggle
  bool                        m_dlssRRInitialized = false;
  uint32_t                    m_dlssRRFrameIndex  = 0;
  bool                        m_dlssRRNeedsReset  = true;   // Reset temporal history
  
  // DLSS-RR quality preset
  NVSDK_NGX_PerfQuality_Value m_dlssRRQuality = NVSDK_NGX_PerfQuality_Value_MaxQuality;
  
  void initializeDlssRR();
  void shutdownDlssRR();
  void updateDlssRRDescriptorSet();
#endif

  // camera info for current frame, updated by onRender
  glm::vec3 m_eye{};
  glm::vec3 m_center{};
  glm::vec3 m_up{};

  // IndirectParams structure defined in shaderio.h
  nvvk::Buffer             m_indirect;              // indirect parameter buffer
  VkDeviceSize             m_indirectStride = 0;    // Stride for double-buffering
  uint32_t                 m_frameIndex = 0;        // Current frame index (0 or 1)
  uint32_t                 m_currentFrameInfoOffset = 0; // Current offset in FrameInfo buffer for dynamic update
  uint32_t                 m_lastFrameInfoOffset = 0; // Last used offset for binding (to avoid race conditions)
  VkDeviceSize             m_frameInfoStride = 0;   // Stride for FrameInfo UBO dynamic alignment
  nvvk::Buffer             m_indirectReadbackHost;  // buffer for readback

  shaderio::IndirectParams m_indirectReadback;      // readback values
  bool m_canCollectReadback = false;  // tells wether readback will be available in Host buffer at next frame

  // TODO maybe move that in SplatSetVK next to icosa, and the associated init/deinit
  nvvk::Buffer m_quadVertices;  // Buffer of vertices for the splat quad
  nvvk::Buffer m_quadIndices;   // Buffer of indices for the splat quad


  SplatSorterAsync      m_cpuSorter;                   // CPU async sorting
  std::vector<uint32_t> m_splatIndices;                // the array of cpu sorted indices to use for rendering
  VrdxSorter            m_gpuSorter = VK_NULL_HANDLE;  // GPU radix sort

  // Temporal stability: skip GPU sorting when camera is stationary
  glm::vec3 m_lastSortCameraPosition{0.0f};
  glm::vec3 m_lastSortCameraDirection{0.0f};
  bool      m_lastSortValid = false;
  bool      m_sortSkippedThisFrame = false;  // For debug UI display

  // buffers used by GPU and/or CPU sort
  nvvk::Buffer m_splatIndicesHost;      // Buffer of splat indices on host for transfers (used by CPU sort)
  nvvk::Buffer m_splatIndicesDevice;    // Buffer of splat indices on device (used by CPU and GPU sort)
  nvvk::Buffer m_splatDistancesDevice;  // Buffer of splat indices on device (used by CPU and GPU sort)
  nvvk::Buffer m_vrdxStorageDevice;     // Used internally by VrdxSorter, GPU sort
  uint32_t     m_rendererBufferCapacity = 0;  // Current capacity of sorting buffers (for grow-only resizing)

  // macro definitions shared by all shaders
  std::vector<std::pair<std::string, std::string>> m_shaderMacros;
  // used to load and compile shaders
  nvslang::SlangCompiler m_slangCompiler{};

  // Chunk-based hierarchical frustum culling
  nvvk::Buffer             m_chunkBoundsBuffer;           // Per-chunk AABB bounds (ChunkBounds[numChunks])
  nvvk::Buffer             m_visibleChunksBuffer;         // Output: visible chunk indices
  nvvk::Buffer             m_visibleChunkCountBuffer;     // Output: count of visible chunks (atomic counter)
  uint32_t                 m_numChunks = 0;               // Total number of chunks
  std::vector<shaderio::ChunkBounds> m_chunkBoundsCpu;    // CPU-side chunk bounds for upload
  VkPipeline               m_computePipelineChunkCull = VK_NULL_HANDLE;  // Chunk culling compute pipeline
  VkDescriptorSetLayout    m_chunkCullDescriptorSetLayout = VK_NULL_HANDLE;
  VkDescriptorSet          m_chunkCullDescriptorSet = VK_NULL_HANDLE;
  VkDescriptorPool         m_chunkCullDescriptorPool = VK_NULL_HANDLE;
  VkPipelineLayout         m_chunkCullPipelineLayout = VK_NULL_HANDLE;
  
  // The different shaders that are used in the pipelines
  struct Shaders
  {
    // 3D Gaussians Raster
    VkShaderModule distShader{};
    VkShaderModule chunkCullShader{};  // Chunk culling compute shader
    VkShaderModule meshShader{};
    VkShaderModule vertexShader{};
    VkShaderModule fragmentShader{};
    VkShaderModule threedgutMeshShader{};
    VkShaderModule threedgutFragmentShader{};
    // 3D Meshes raster
    VkShaderModule meshVertexShader{};
    VkShaderModule meshFragmentShader{};
    // for RTX
    VkShaderModule rtxRgenShader{};    // The ray generator
    VkShaderModule rtxRmissShader{};   // The miss shader
    VkShaderModule rtxRmiss2Shader{};  // For shadows (no support yet)
    VkShaderModule rtxRchitShader{};   // Closest Hit
    VkShaderModule rtxRahitShader{};   // Any Hit
    VkShaderModule rtxRintShader{};    // Interrsection
    // Post processings
    VkShaderModule postComputeShader{};
    // VDZ depth mesh rendering
    VkShaderModule vdzMeshVertexShader{};
    VkShaderModule vdzMeshFragmentShader{};
    // VDZ hybrid rendering (mesh + POM)
    VkShaderModule vdzHybridVertexShader{};
    VkShaderModule vdzHybridFragmentShader{};
    // Hand mesh rendering (XR)
    VkShaderModule handMeshVertexShader{};
    VkShaderModule handMeshFragmentShader{};

    // Utility storage to process shaders in loop
    std::vector<VkShaderModule*> modules{};
    // true if all the shaders are succesfully build
    bool valid = false;
  } m_shaders;

  // 3D Gaussians Pipelines
  VkPipeline m_computePipelineGsDistCull = VK_NULL_HANDLE;  // The compute pipeline to compute gaussian splats distances to eye and cull
  VkPipeline m_graphicsPipelineGsVert = VK_NULL_HANDLE;  // The graphic pipeline to rasterize gaussian splats using vertex shaders
  VkPipeline m_graphicsPipelineGsMesh = VK_NULL_HANDLE;  // The graphic pipeline to rasterize gaussian splats using mesh shaders
  VkPipeline m_graphicsPipeline3dgutMesh = VK_NULL_HANDLE;  // The graphic pipeline to rasterize 3DGUT splats using mesh shaders
  // 3D Meshes Pipelines
  VkPipeline m_graphicsPipelineMesh = VK_NULL_HANDLE;  // The graphic pipeline to rasterize meshes
    // VDZ depth mesh pipeline
    VkPipeline m_graphicsPipelineVdzMesh = VK_NULL_HANDLE;
    // VDZ hybrid pipeline (mesh + POM)
    VkPipeline m_graphicsPipelineVdzHybrid = VK_NULL_HANDLE;
  // Hand mesh pipeline (XR skinned hands)
  VkPipeline m_graphicsPipelineHandMesh = VK_NULL_HANDLE;
#ifdef WITH_OPENXR
  // Multiview variants of raster pipelines (viewMask = 0x3 for stereo)
  VkPipeline m_graphicsPipelineGsVertMultiview = VK_NULL_HANDLE;
  VkPipeline m_graphicsPipelineGsMeshMultiview = VK_NULL_HANDLE;
  VkPipeline m_graphicsPipeline3dgutMeshMultiview = VK_NULL_HANDLE;
  VkPipeline m_graphicsPipelineHandMeshMultiview = VK_NULL_HANDLE;
#endif

  // Common to 3D meshes and 3D Gaussians pipeline
  VkPipelineLayout      m_pipelineLayout      = VK_NULL_HANDLE;  // Raster Pipelines layout
  VkDescriptorSetLayout m_descriptorSetLayout = VK_NULL_HANDLE;  // Descriptor set layout
  VkDescriptorSet       m_descriptorSet       = VK_NULL_HANDLE;  // Raster Descriptor set
  VkDescriptorPool      m_descriptorPool      = VK_NULL_HANDLE;  // Raster Descriptor pool

  nvvk::Buffer m_frameInfoBuffer;  // uniform buffer to store frame parameters defined by global variable prmFrame

  // Rendering (sorting and splatting) related memory usage statistics
  struct RenderMemoryStats
  {
    // Rasterization

    uint64_t usedUboFrameInfo = 0;  // used = alloc all the time
    uint64_t usedIndirect     = 0;  // used = alloc all the time, for the active pipeline

    uint64_t hostAllocDistances = 0;  // used = alloc
    uint64_t hostAllocIndices   = 0;  // used = alloc

    uint64_t allocIndices      = 0;
    uint64_t usedIndices       = 0;
    uint64_t allocDistances    = 0;
    uint64_t usedDistances     = 0;
    uint64_t allocVdrxInternal = 0;  // used is unknown

    uint64_t rasterHostTotal        = 0;
    uint64_t rasterDeviceUsedTotal  = 0;
    uint64_t rasterDeviceAllocTotal = 0;

    // RTX
    uint64_t rtxUsedTlas = 0;
    uint64_t rtxUsedBlas = 0;

    uint64_t rtxHostTotal        = 0;
    uint64_t rtxDeviceUsedTotal  = 0;
    uint64_t rtxDeviceAllocTotal = 0;

    // Totals
    uint64_t hostTotal        = 0;
    uint64_t deviceUsedTotal  = 0;
    uint64_t deviceAllocTotal = 0;

  } m_renderMemoryStats;

  /////////////////////////
  // RTX specific

  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  VkPhysicalDeviceAccelerationStructurePropertiesKHR m_accelStructProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};

  std::vector<VkRayTracingShaderGroupCreateInfoKHR> m_rtShaderGroups;
  VkPipelineLayout                                  m_rtPipelineLayout = VK_NULL_HANDLE;
  VkPipeline m_rtPipeline = VK_NULL_HANDLE;  // The RTX pipeline to ray trace gaussian splats and meshes

  nvvk::DescriptorBindings m_rtDescriptorBindings  = {};
  VkDescriptorSetLayout    m_rtDescriptorSetLayout = VK_NULL_HANDLE;
  VkDescriptorSet          m_rtDescriptorSet       = VK_NULL_HANDLE;
  VkDescriptorPool         m_rtDescriptorPool      = VK_NULL_HANDLE;

  nvvk::Buffer m_payloadDevice;

  nvvk::Buffer m_rtSBTBuffer;  // common to GS and Mesh
  // The 4 SBT regions (raygen, miss, chit, call in this order)
  nvvk::SBTGenerator::Regions m_sbtRegions{};  // common to GS and Mesh

  shaderio::PushConstantRay m_pcRay{};  // Push constant for ray tracer

  std::unique_ptr<DepthStreamClient>   m_depthClient;
  std::unique_ptr<DepthTextureManager> m_depthManager;
  bool m_enableDepthRendering = false;
  bool m_videoDepthPlaybackMode = false;
  glm::mat4 m_vdzModelMatrix{1.0f};
  bool m_vdzWorldSpaceInitialized = false;
  float m_depthScale = 1.0f;
  float m_depthBias = 0.0f;

  // Parallax rendering state
  glm::vec2 m_lastMousePos = {-1.0f, -1.0f};  // Last mouse position for drag detection
  bool m_parallaxDragActive = false;           // Whether parallax drag is active
  float m_parallaxSensitivity = 0.001f;        // Sensitivity for parallax offset

  // VR depth video control state
  glm::vec2 m_vrPlaneTilt{0.0f, 0.0f};         // Current plane tilt (pitch, yaw) in VR
  glm::vec2 m_vrPlaneTiltTarget{0.0f, 0.0f};   // Target tilt (spring target)
  glm::vec2 m_vrViewpointOffset{0.0f, 0.0f};   // Viewpoint offset for parallax in VR
  float m_vrParallaxScale = 1.0f;              // Parallax scale from VR right stick X
  float m_vrViewpointDistance = 0.0f;          // Viewpoint distance from plane (VR right stick Y)

  DepthStreamClient::ClientStats m_depthStats{};
  int m_depthFrameCounter = 0;  // For selective debug logging
  
  size_t m_lastVdzFrameIndex = SIZE_MAX;
  std::chrono::steady_clock::time_point m_playbackStartTime;
  bool m_playbackPaused = false;
  double m_playbackTimeOffset = 0.0;

#ifdef WITH_VIDEO_DECODER
  std::unique_ptr<VideoDecoder> m_videoDecoder;
  std::unique_ptr<VideoDepthPlaybackManager> m_videoDepthManager;

  struct VideoTexture {
    nvvk::Image image;
    VkImageView view = VK_NULL_HANDLE;
    uint32_t width = 0;
    uint32_t height = 0;
  } m_videoTexture;
#endif

  // HLS Depth Player for HLS streaming playback
#ifdef WITH_VIDEO_DECODER
  std::unique_ptr<HlsDepthPlayer> m_hlsPlayer;
  bool m_hlsPlaybackMode = false;
  HlsDepthMetadata m_hlsMetadata{};
#endif

  // Backend process manager for spawning/stopping local backend
  std::unique_ptr<BackendProcessManager> m_backendManager;
  bool m_localBackendStarted = false;

  // Dummy texture for binding initialization (1x1 2D array)
  struct
  {
    nvvk::Image image;
    VkImageView view = VK_NULL_HANDLE;
  } m_dummyTextureArray;

  ///////////////////////////////
  // Post processing



  VkPipeline       m_computePipelinePostProcess = VK_NULL_HANDLE;
  VkPipelineLayout m_pipelineLayoutPostProcess  = VK_NULL_HANDLE;

  nvvk::DescriptorBindings m_descriptorBindingsPostProcess{};
  VkDescriptorSetLayout    m_descriptorSetLayoutPostProcess = VK_NULL_HANDLE;
  VkDescriptorSet          m_descriptorSetPostProcess       = VK_NULL_HANDLE;
  VkDescriptorPool         m_descriptorPoolPostProcess      = VK_NULL_HANDLE;

  // PLY Sequence Animation
  std::shared_ptr<AnimationController> m_animationController;
  std::unique_ptr<AnimationUI>         m_animationUI;
  bool                                  m_isAnimationPlaying = false;

  // Async loading
  std::filesystem::path                 m_pendingSequencePath;
  std::thread                           m_sequenceLoadThread;
  std::mutex                            m_sequenceLoadMutex;

  void enablePlySequencePlayback(const std::filesystem::path& dirPath);
  void loadPlySequenceAsync(const std::filesystem::path& dirPath);
  void updateAnimation(float deltaTime);
  bool isAnimationActive() const { return m_isAnimationPlaying && m_animationController != nullptr; }

};

}  // namespace vk_viewer

#endif
