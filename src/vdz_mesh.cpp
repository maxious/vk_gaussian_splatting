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

#include "vdz_mesh.h"
#include <nvutils/logger.hpp>
#include <nvvk/resources.hpp>
#include <nvvk/check_error.hpp>
#include <cstring>

namespace vk_gaussian_splatting {

bool VdzMesh::initialize(VkDevice device, nvvk::ResourceAllocator* alloc, uint32_t gridWidth, uint32_t gridHeight)
{
  m_device = device;
  m_alloc  = alloc;

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

  LOGI("VDZ Mesh initialized: %u vertices, %u indices (%ux%u grid)\n", 
       m_vertexCount, m_indexCount, gridWidth, gridHeight);

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
  m_indexCount  = 0;
}

void VdzMesh::generateGrid(uint32_t width, uint32_t height)
{
  m_vertices.clear();
  m_indices.clear();

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

}  // namespace vk_gaussian_splatting
