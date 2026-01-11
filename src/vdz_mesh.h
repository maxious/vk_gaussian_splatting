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

#ifndef VDZ_MESH_H
#define VDZ_MESH_H

#include <vulkan/vulkan_core.h>
#include <nvvk/resources.hpp>
#include <nvvk/resource_allocator.hpp>
#include <glm/glm.hpp>
#include <vector>

namespace vk_viewer {

// Depth rendering mode for VDZ content
enum class VdzRenderMode
{
    DividedMesh = 0,  // Original divided mesh approach (128x72 default)
    PomOnly = 1,      // POM only on simple quad
    Hybrid = 2        // Low-res divided mesh + POM for best quality
};

struct VdzMeshVertex
{
  glm::vec3 position;
  glm::vec2 uv;
};

class VdzMesh
{
public:
  bool initialize(VkDevice device, nvvk::ResourceAllocator* alloc, uint32_t gridWidth = 128, uint32_t gridHeight = 72);
  void cleanup();

  // Reinitialize with new grid size (useful for switching modes)
  bool reinitialize(uint32_t gridWidth, uint32_t gridHeight);

  // Generate mesh for hybrid mode (lower resolution, combined with POM)
  void generateHybridGrid(uint32_t width, uint32_t height);

  // Generate simple quad for POM-only mode
  void generateQuad();

  VkBuffer getVertexBuffer() const { return m_vertexBuffer.buffer; }
  VkBuffer getIndexBuffer() const { return m_indexBuffer.buffer; }
  uint32_t getIndexCount() const { return m_indexCount; }
  uint32_t getVertexCount() const { return m_vertexCount; }

  // Get current mesh dimensions
  uint32_t getGridWidth() const { return m_gridWidth; }
  uint32_t getGridHeight() const { return m_gridHeight; }
  VdzRenderMode getRenderMode() const { return m_renderMode; }

  static VkVertexInputBindingDescription getBindingDescription();
  static std::vector<VkVertexInputAttributeDescription> getAttributeDescriptions();

private:
  void generateGrid(uint32_t width, uint32_t height);

  VkDevice                  m_device = VK_NULL_HANDLE;
  nvvk::ResourceAllocator*  m_alloc  = nullptr;

  nvvk::Buffer              m_vertexBuffer;
  nvvk::Buffer              m_indexBuffer;
  uint32_t                  m_indexCount  = 0;
  uint32_t                  m_vertexCount = 0;
  uint32_t                  m_gridWidth   = 128;  // Store current grid dimensions
  uint32_t                  m_gridHeight  = 72;
  VdzRenderMode             m_renderMode  = VdzRenderMode::DividedMesh;

  std::vector<VdzMeshVertex> m_vertices;
  std::vector<uint32_t>      m_indices;
};

}  // namespace vk_viewer

#endif  // VDZ_MESH_H
