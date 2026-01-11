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

#ifndef RENDER_CONTEXT_H_
#define RENDER_CONTEXT_H_

#include <vector>
#include <vulkan/vulkan_core.h>
#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

namespace vk_viewer {

enum class OutputMode
{
  DesktopMono,       // Single desktop window, mono view
  DesktopStereoSBS,  // Desktop window with side-by-side stereo
  XrDualPass,        // OpenXR with dual-pass per-eye rendering
  XrMultiview        // OpenXR with VK_KHR_multiview single-pass stereo
};

enum class RenderContentMode
{
  SplatsOnly,     // Render only Gaussian splats
  DepthOnly,      // Render only depth video mesh
  SplatsAndDepth  // Render both splats and depth content
};

struct RenderView
{
  glm::mat4  view{1.0f};
  glm::mat4  proj{1.0f};
  glm::vec3  eye{0.0f};
  glm::vec2  stereoShift{0.0f};  // Principal point shift for off-axis stereo (in pixels)
  VkViewport viewport{};
  VkRect2D   scissor{};
  uint32_t   viewIndex = 0;  // 0 for mono/left, 1 for right eye
};

struct FrameRenderContext
{
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  uint32_t        frameIndex = 0;
  uint32_t        splatCount = 0;
  uint32_t        colorBufferId = 0;
  glm::vec2       clipPlanes{0.1f, 1000.0f};
  glm::vec2       viewSize{0.0f};

  // XR state
  bool xrFrameActive = false;
  bool xrShouldRender = false;
  bool useXrMultiview = false;

  // Content flags
  bool hasSplats = false;
  bool hasDepthContent = false;
  bool hasMeshes = false;
  bool shadersValid = false;

  // Rendering modes
  RenderContentMode contentMode = RenderContentMode::SplatsOnly;
  OutputMode        outputMode = OutputMode::DesktopMono;

  // Camera state (center eye for sorting)
  glm::mat4 viewMatrix{1.0f};
  glm::mat4 projMatrix{1.0f};
  glm::vec3 eyePosition{0.0f};
  glm::vec3 centerPosition{0.0f};
  glm::vec3 upVector{0.0f, 1.0f, 0.0f};

  // Collected views for rendering
  std::vector<RenderView> views;

  // RTX-specific flags
  bool raytraceMeshDepth = false;
  bool useRtxPipeline = false;

  // Environment depth (XR passthrough)
  glm::mat4 envDepthViewMatrices[2];
  glm::mat4 envDepthProjMatrices[2];
};

}  // namespace vk_viewer

#endif  // RENDER_CONTEXT_H_
