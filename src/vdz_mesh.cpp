/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless REQUIRED BY APPLICABLE LAW OR AGREED TO IN WRITING, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "vdz_mesh.h"
#include <nvutils/logger.hpp>
#include <nvvk/resources.hpp>
#include <nvvk/check_error.hpp>
#include <cstring>

namespace vk_viewer {

bool VdzMesh::initialize(VkDevice device, nvvk::ResourceAllocator* alloc, uint32_t gridWidth, uint32_t gridHeight)
{
  m_device = device;
  m_alloc  = alloc;
  m_gridWidth = gridWidth;
  m_gridHeight = gridHeight;
  m_renderMode = VdzRenderMode::DividedMesh;

  generateGrid(gridWidth, gridHeight);

  VkBufferUsageFlags vertexUsage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
  VkBufferUsageFlags indexUsage  = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;

  VkDeviceSize vertexSize = m_vertices.size() * sizeof(VdzMeshVertex);
  VkDeviceSize indexSize  = m_indices.size() * sizeof(uint32_t);

  NVVK_CHECK(m_alloc->createBuffer(m_vertexBuffer, vertexSize, vertexUsage,
                                   VMA_MEMORY_USAGE_AUTO_PREFER_HOST, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT));
  NVVK_CHECK(m_alloc->createBuffer(m_indexBuffer, indexSize, indexUsage,
                                   VMA_MEMORY_USAGE_AUTO_PREFER_HOST, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT));

  if(m_vertexBuffer.mapping && m_indexBuffer.mapping)
  {
    std::memcpy(m_vertexBuffer.mapping, m_vertices.data(), vertexSize);
    std::memcpy(m_indexBuffer.mapping, m_indices.data(), indexSize);
  }
  else
  {
    LOGE("Failed to map VDZ mesh buffers\n");
    return false;
  }

  m_vertexCount = static_cast<uint32_t>(m_vertices.size());
  m_indexCount  = static_cast<uint32_t>(m_indices.size());

  LOGI("VDZ Mesh initialized: %u vertices, %u indices (%ux%u grid, mode=%d)\n", 
       m_vertexCount, m_indexCount, gridWidth, gridHeight, (int)m_renderMode);

  return true;
}

bool VdzMesh::reinitialize(uint32_t gridWidth, uint32_t gridHeight)
{
  // Only regenerate if dimensions changed
  if (m_gridWidth == gridWidth && m_gridHeight == gridHeight && !m_vertices.empty())
  {
    return true;  // No change needed
  }

  // Generate new mesh data
  m_gridWidth = gridWidth;
  m_gridHeight = gridHeight;

  if (m_renderMode == VdzRenderMode::PomOnly)
  {
    generateQuad();
  }
  else
  {
    // DividedMesh or Hybrid - both use grid mesh
    generateGrid(gridWidth, gridHeight);
  }

  // Reallocate buffers if they exist
  if (m_vertexBuffer.buffer != VK_NULL_HANDLE)
  {
    m_alloc->destroyBuffer(m_vertexBuffer);
  }
  if (m_indexBuffer.buffer != VK_NULL_HANDLE)
  {
    m_alloc->destroyBuffer(m_indexBuffer);
  }

  VkBufferUsageFlags vertexUsage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
  VkBufferUsageFlags indexUsage  = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;

  VkDeviceSize vertexSize = m_vertices.size() * sizeof(VdzMeshVertex);
  VkDeviceSize indexSize  = m_indices.size() * sizeof(uint32_t);

  NVVK_CHECK(m_alloc->createBuffer(m_vertexBuffer, vertexSize, vertexUsage,
                                   VMA_MEMORY_USAGE_AUTO_PREFER_HOST, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT));
  NVVK_CHECK(m_alloc->createBuffer(m_indexBuffer, indexSize, indexUsage,
                                   VMA_MEMORY_USAGE_AUTO_PREFER_HOST, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT));

  if(m_vertexBuffer.mapping && m_indexBuffer.mapping)
  {
    std::memcpy(m_vertexBuffer.mapping, m_vertices.data(), vertexSize);
    std::memcpy(m_indexBuffer.mapping, m_indices.data(), indexSize);
  }
  else
  {
    LOGE("Failed to map VDZ mesh buffers during reinit\n");
    return false;
  }

  m_vertexCount = static_cast<uint32_t>(m_vertices.size());
  m_indexCount  = static_cast<uint32_t>(m_indices.size());

  LOGI("VDZ Mesh reinitialized: %u vertices, %u indices (%ux%u grid, mode=%d)\n", 
       m_vertexCount, m_indexCount, gridWidth, gridHeight, (int)m_renderMode);

  return true;
}

void VdzMesh::cleanup()
{
  if(m_alloc)
  {
    m_alloc->destroyBuffer(m_vertexBuffer);
    m_alloc->destroyBuffer(m_indexBuffer);
  }
  m_vertices.clear();
  m_indices.clear();
  m_vertexCount = 0;
  m_indexCount = 0;
  m_gridWidth = 0;
  m_gridHeight = 0;
}

void VdzMesh::generateGrid(uint32_t width, uint32_t height)
{
  m_vertices.clear();
  m_indices.clear();
  m_renderMode = VdzRenderMode::DividedMesh;

  m_vertices.reserve(width * height);
  m_indices.reserve((width - 1) * (height - 1) * 6);

  for(uint32_t y = 0; y < height; ++y)
  {
    for(uint32_t x = 0; x < width; ++x)
    {
      float u = static_cast<float>(x) / static_cast<float>(width - 1);
      float v = static_cast<float>(y) / static_cast<float>(height - 1);

      VdzMeshVertex vertex;
      vertex.position = glm::vec3(u, v, 0.0f);
      vertex.uv       = glm::vec2(u, v);
      m_vertices.push_back(vertex);
    }
  }

  for(uint32_t y = 0; y < height - 1; ++y)
  {
    for(uint32_t x = 0; x < width - 1; ++x)
    {
      uint32_t topLeft     = y * width + x;
      uint32_t topRight    = topLeft + 1;
      uint32_t bottomLeft  = (y + 1) * width + x;
      uint32_t bottomRight = bottomLeft + 1;

      m_indices.push_back(topLeft);
      m_indices.push_back(bottomLeft);
      m_indices.push_back(topRight);

      m_indices.push_back(topRight);
      m_indices.push_back(bottomLeft);
      m_indices.push_back(bottomRight);
    }
  }
}

void VdzMesh::generateHybridGrid(uint32_t width, uint32_t height)
{
  m_vertices.clear();
  m_indices.clear();
  m_renderMode = VdzRenderMode::Hybrid;
  m_gridWidth = width;
  m_gridHeight = height;

  // Hybrid mode uses a lower resolution grid (e.g., 32x18 instead of 128x72)
  // The mesh provides coarse depth structure, POM handles fine detail
  m_vertices.reserve(width * height);
  m_indices.reserve((width - 1) * (height - 1) * 6);

  for(uint32_t y = 0; y < height; ++y)
  {
    for(uint32_t x = 0; x < width; ++x)
    {
      float u = static_cast<float>(x) / static_cast<float>(width - 1);
      float v = static_cast<float>(y) / static_cast<float>(height - 1);

      VdzMeshVertex vertex;
      vertex.position = glm::vec3(u, v, 0.0f);
      vertex.uv       = glm::vec2(u, v);
      m_vertices.push_back(vertex);
    }
  }

  for(uint32_t y = 0; y < height - 1; ++y)
  {
    for(uint32_t x = 0; x < width - 1; ++x)
    {
      uint32_t topLeft     = y * width + x;
      uint32_t topRight    = topLeft + 1;
      uint32_t bottomLeft  = (y + 1) * width + x;
      uint32_t bottomRight = bottomLeft + 1;

      m_indices.push_back(topLeft);
      m_indices.push_back(bottomLeft);
      m_indices.push_back(topRight);

      m_indices.push_back(topRight);
      m_indices.push_back(bottomLeft);
      m_indices.push_back(bottomRight);
    }
  }
}

void VdzMesh::generateQuad()
{
  m_vertices.clear();
  m_indices.clear();
  m_renderMode = VdzRenderMode::PomOnly;

  // Simple quad (2 triangles) for POM-only mode
  // Fullscreen quad covering [0,1] x [0,1] in UV space
  m_vertices = {
    {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},  // bottom-left
    {{1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},  // bottom-right
    {{1.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},  // top-right
    {{0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},  // top-left
  };
  m_indices = {0, 1, 2, 0, 2, 3};

  m_gridWidth = 2;
  m_gridHeight = 2;
}

VkVertexInputBindingDescription VdzMesh::getBindingDescription()
{
  VkVertexInputBindingDescription bindingDescription{};
  bindingDescription.binding   = 0;
  bindingDescription.stride    = sizeof(VdzMeshVertex);
  bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
  return bindingDescription;
}

std::vector<VkVertexInputAttributeDescription> VdzMesh::getAttributeDescriptions()
{
  std::vector<VkVertexInputAttributeDescription> attributeDescriptions(2);

  attributeDescriptions[0].binding  = 0;
  attributeDescriptions[0].location = 0;
  attributeDescriptions[0].format   = VK_FORMAT_R32G32B32_SFLOAT;
  attributeDescriptions[0].offset   = offsetof(VdzMeshVertex, position);

  attributeDescriptions[1].binding  = 0;
  attributeDescriptions[1].location = 1;
  attributeDescriptions[1].format   = VK_FORMAT_R32G32_SFLOAT;
  attributeDescriptions[1].offset   = offsetof(VdzMeshVertex, uv);

  return attributeDescriptions;
}

}  // namespace vk_viewer
