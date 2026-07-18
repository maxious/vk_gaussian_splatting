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

#include "splat_set_vk.h"
#include "shaderio.h"
#include "utilities.h"

#include <chrono>
#include <cinttypes>

// mathematics
#include <glm/vec3.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/packing.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/transform.hpp>

#include <nvutils/logger.hpp>
#include <nvutils/timers.hpp>

#include <nvvk/debug_util.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/default_structs.hpp>

namespace vk_viewer {

// Utility function to compute the texture size according to the size of the data to be stored
// By default use map of 4K Width and 1K height then adjust the height according to the data size
glm::ivec2 computeDataTextureSize(int elementsPerTexel, int elementsPerSplat, int maxSplatCount, glm::ivec2 texSize = {4096, 1024})
{
  while(texSize.x * texSize.y * elementsPerTexel < maxSplatCount * elementsPerSplat)
    texSize.y *= 2;
  return texSize;
};

// quantize a float onto a uint8
uint8_t toUint8(float v, float rangeMin, float rangeMax)
{
  float normalized = (v - rangeMin) / (rangeMax - rangeMin);
  return static_cast<uint8_t>(std::clamp(std::round(normalized * 255.0f), 0.0f, 255.0f));
};

// Returns the size in bytes for a given format enum
int formatSize(uint32_t format)
{
  if(format == FORMAT_FLOAT32)
    return 4;
  if(format == FORMAT_FLOAT16)
    return 2;
  if(format == FORMAT_UINT8)
    return 1;
  return 0;
}

// convert SH coef to given format on the flight and store into dstBuffer
void storeSh(int format, float* srcBuffer, uint64_t srcIndex, void* dstBuffer, uint64_t dstIndex)
{
  if(format == FORMAT_FLOAT32)
    static_cast<float*>(dstBuffer)[dstIndex] = srcBuffer[srcIndex];
  else if(format == FORMAT_FLOAT16)
    static_cast<uint16_t*>(dstBuffer)[dstIndex] = glm::packHalf1x16(srcBuffer[srcIndex]);
  else if(format == FORMAT_UINT8)
    static_cast<uint8_t*>(dstBuffer)[dstIndex] = toUint8(srcBuffer[srcIndex], -1., 1.);
}

///////////////////
// class definition

void SplatSetVk::initDataStorage(SplatSet& splatSet, uint32_t storage, uint32_t format)
{
  // TODO check if properly deinit before anything

  // store the parameters for further usage
  m_storage = storage;
  m_format  = format;

  if(m_storage == STORAGE_BUFFERS)
  {
    initDataBuffers(splatSet);
  }
  else if(m_storage == STORAGE_TEXTURES)
  {
    initDataTextures(splatSet);
  }
  else if(m_storage == STORAGE_LCC_PACKED)
  {
    // LCC packed storage is initialized via initLccPackedStorage()
    LOGI("STORAGE_LCC_PACKED mode: use initLccPackedStorage() for raw data upload\n");
  }
  else
    LOGE("Invalid storage format");
}

void SplatSetVk::initLccPackedStorage(const uint8_t* rawData, uint32_t splatCount)
{
  if(!rawData || splatCount == 0)
  {
    LOGE("Invalid LCC packed data\n");
    return;
  }

  auto startTime = std::chrono::high_resolution_clock::now();

  // LCC Portable format: 32 bytes per splat
  constexpr size_t LCC_SPLAT_SIZE = 32;
  const size_t totalBytes = static_cast<size_t>(splatCount) * LCC_SPLAT_SIZE;

  VkCommandBuffer cmd = m_app->createTempCmdBuffer();

  // Host buffer flags (for staging)
  VkBufferUsageFlagBits2   hostBufferUsageFlags = VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT;
  VmaMemoryUsage           hostMemoryUsageFlags = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
  VmaAllocationCreateFlags hostAllocCreateFlags = VMA_ALLOCATION_CREATE_MAPPED_BIT
                                                  | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;

  // Device buffer flags
  VkBufferUsageFlagBits2 deviceBufferUsageFlags =
      VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT
      | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT;
  VmaMemoryUsage deviceMemoryUsageFlags = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

  // Create host staging buffer
  nvvk::Buffer hostBuffer;
  m_alloc->createBuffer(hostBuffer, totalBytes, hostBufferUsageFlags, hostMemoryUsageFlags, hostAllocCreateFlags);

  // Create device buffer for LCC packed data
  m_alloc->createBuffer(lccPackedBuffer, totalBytes, deviceBufferUsageFlags, deviceMemoryUsageFlags);

  // Copy raw data to host staging buffer
  memcpy(hostBuffer.mapping, rawData, totalBytes);

  // Copy from host to device
  VkBufferCopy copyRegion{.srcOffset = 0, .dstOffset = 0, .size = totalBytes};
  vkCmdCopyBuffer(cmd, hostBuffer.buffer, lccPackedBuffer.buffer, 1, &copyRegion);

  // Add barrier for shader reads
  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, 0, 1, &barrier, 0,
                       nullptr, 0, nullptr);

  m_app->submitAndWaitTempCmdBuffer(cmd);

  // Free staging buffer
  m_alloc->destroyBuffer(hostBuffer);

  // Update memory stats
  memoryStats.devAll = static_cast<uint32_t>(totalBytes);
  memoryStats.odevAll = static_cast<uint32_t>(totalBytes);

  auto endTime = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
  LOGI("LCC packed storage initialized: %u splats, %zu bytes in %lld ms\n",
       splatCount, totalBytes, static_cast<long long>(duration.count()));

  m_storage = STORAGE_LCC_PACKED;
}

void SplatSetVk::deinitDataStorage()
{
  if(m_storage == STORAGE_BUFFERS)
  {
    deinitDataBuffers();
  }
  else if(m_storage == STORAGE_TEXTURES)
  {
    deinitDataTextures();
  }
  else if(m_storage == STORAGE_LCC_PACKED)
  {
    // Deinit LCC packed buffer
    if(lccPackedBuffer.buffer != VK_NULL_HANDLE)
    {
      m_alloc->destroyBuffer(lccPackedBuffer);
      lccPackedBuffer = {};
    }
  }
  else
    LOGE("Invalid storage format");
}

///////////////////
// using data buffers to store splatset in VRAM

void SplatSetVk::initDataBuffers(SplatSet& splatSet)
{
  if(false)
  {
    // dump splat info for debug
    uint32_t splatId = 4178424;
    if(splatId < splatSet.size())
    {

      LOGD("pos: %f %f %f  0 1 0  f_dc: %f %f %f\n", splatSet.positions[splatId * 3 + 0],
           splatSet.positions[splatId * 3 + 1], splatSet.positions[splatId * 3 + 2], splatSet.f_dc[splatId * 3 + 0],
           splatSet.f_dc[splatId * 3 + 1], splatSet.f_dc[splatId * 3 + 2]);
      for(int i = 0; i < 45; ++i)
      {
        LOGD("f_rest[%d]: %f\n", i, splatSet.f_rest[splatId * 45 + i]);
      }
      LOGD("opacity: %f  scale: %f %f %f  rotation: %f %f %f %f\n", splatSet.opacity[splatId],
           splatSet.scale[splatId * 3 + 0], splatSet.scale[splatId * 3 + 1], splatSet.scale[splatId * 3 + 2],
           splatSet.rotation[splatId * 4 + 0], splatSet.rotation[splatId * 4 + 1], splatSet.rotation[splatId * 4 + 2],
           splatSet.rotation[splatId * 4 + 3]);
    }
  }
  auto       startTime  = std::chrono::high_resolution_clock::now();
  const auto splatCount = (uint32_t)splatSet.positions.size() / 3;

  VkCommandBuffer cmd = m_app->createTempCmdBuffer();

  // host buffers flags
  VkBufferUsageFlagBits2   hostBufferUsageFlags = VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT;
  VmaMemoryUsage           hostMemoryUsageFlags = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
  VmaAllocationCreateFlags hostAllocCreateFlags = VMA_ALLOCATION_CREATE_MAPPED_BIT
                                                  // for parallel access
                                                  | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
  // device buffers flags
  VkBufferUsageFlagBits2 deviceBufferUsageFlags =
      VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT
      | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_VERTEX_BUFFER_BIT;
  VmaMemoryUsage deviceMemoryUsageFlags = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

  // set of buffer to be freed after command execution
  std::vector<nvvk::Buffer> buffersToDestroy;

  // Centers and Scales (scales and rotations are only for raytrace, raster
  // uses pre-computed covariances, see covariance section hereafter)
  {
    const uint32_t bufferSize1Comp = splatCount * 1 * sizeof(float);
    const uint32_t bufferSize3Comp = splatCount * 3 * sizeof(float);
    const uint32_t bufferSize4Comp = splatCount * 4 * sizeof(float);

    // allocate host and device buffers
    nvvk::Buffer hostBufferCenters;
    m_alloc->createBuffer(hostBufferCenters, bufferSize3Comp, hostBufferUsageFlags, hostMemoryUsageFlags, hostAllocCreateFlags);
    NVVK_DBG_NAME(hostBufferCenters.buffer);
    nvvk::Buffer hostBufferScales;
    m_alloc->createBuffer(hostBufferScales, bufferSize3Comp, hostBufferUsageFlags, hostMemoryUsageFlags, hostAllocCreateFlags);
    NVVK_DBG_NAME(hostBufferScales.buffer);
    nvvk::Buffer hostBufferRotations;
    m_alloc->createBuffer(hostBufferRotations, bufferSize4Comp, hostBufferUsageFlags, hostMemoryUsageFlags, hostAllocCreateFlags);
    NVVK_DBG_NAME(hostBufferRotations.buffer);

    m_alloc->createBuffer(centersBuffer, bufferSize3Comp, deviceBufferUsageFlags, deviceMemoryUsageFlags);
    NVVK_DBG_NAME(centersBuffer.buffer);
    m_alloc->createBuffer(scalesBuffer, bufferSize3Comp, deviceBufferUsageFlags, deviceMemoryUsageFlags);
    NVVK_DBG_NAME(scalesBuffer.buffer);
    m_alloc->createBuffer(rotationsBuffer, bufferSize4Comp, deviceBufferUsageFlags, deviceMemoryUsageFlags);
    NVVK_DBG_NAME(rotationsBuffer.buffer);

    // fill host buffer
    memcpy(hostBufferCenters.mapping, splatSet.positions.data(), bufferSize3Comp);
    memcpy(hostBufferScales.mapping, splatSet.scale.data(), bufferSize3Comp);
    memcpy(hostBufferRotations.mapping, splatSet.rotation.data(), bufferSize4Comp);

    // copy from host buffer to device buffer
    // barrier at the end of this method.
    VkBufferCopy bc3Comp{.srcOffset = 0, .dstOffset = 0, .size = bufferSize3Comp};
    vkCmdCopyBuffer(cmd, hostBufferCenters.buffer, centersBuffer.buffer, 1, &bc3Comp);
    vkCmdCopyBuffer(cmd, hostBufferScales.buffer, scalesBuffer.buffer, 1, &bc3Comp);
    VkBufferCopy bc4Comp{.srcOffset = 0, .dstOffset = 0, .size = bufferSize4Comp};
    vkCmdCopyBuffer(cmd, hostBufferRotations.buffer, rotationsBuffer.buffer, 1, &bc4Comp);

    // free host buffer after command execution
    buffersToDestroy.push_back(hostBufferCenters);
    buffersToDestroy.push_back(hostBufferScales);
    buffersToDestroy.push_back(hostBufferRotations);

    // memory statistics
    memoryStats.srcCenters  = bufferSize3Comp;
    memoryStats.odevCenters = bufferSize3Comp;  // no compression or quantization
    memoryStats.devCenters  = bufferSize3Comp;  // same size as source
  }

  {
    const uint32_t count = splatSet.has_time_data ? splatCount : 1;
    const uint32_t bufferSizeMotion = count * 3 * sizeof(float);
    const uint32_t bufferSizeTime   = count * 2 * sizeof(float);

    nvvk::Buffer hostBufferMotion;
    m_alloc->createBuffer(hostBufferMotion, bufferSizeMotion, hostBufferUsageFlags, hostMemoryUsageFlags, hostAllocCreateFlags);
    NVVK_DBG_NAME(hostBufferMotion.buffer);
    
    nvvk::Buffer hostBufferTime;
    m_alloc->createBuffer(hostBufferTime, bufferSizeTime, hostBufferUsageFlags, hostMemoryUsageFlags, hostAllocCreateFlags);
    NVVK_DBG_NAME(hostBufferTime.buffer);

    m_alloc->createBuffer(motionBuffer, bufferSizeMotion, deviceBufferUsageFlags, deviceMemoryUsageFlags);
    NVVK_DBG_NAME(motionBuffer.buffer);
    
    m_alloc->createBuffer(timeBuffer, bufferSizeTime, deviceBufferUsageFlags, deviceMemoryUsageFlags);
    NVVK_DBG_NAME(timeBuffer.buffer);

    if (splatSet.has_time_data) {
        memcpy(hostBufferMotion.mapping, splatSet.motion.data(), bufferSizeMotion);
        float* timeMapped = (float*)hostBufferTime.mapping;
        START_PAR_LOOP(splatCount, i)
        {
          // JS Order: x = scale, y = center
          timeMapped[i * 2 + 0] = splatSet.time_scale[i];
          timeMapped[i * 2 + 1] = splatSet.time[i];
        }
        END_PAR_LOOP()
    } else {
        memset(hostBufferMotion.mapping, 0, bufferSizeMotion);
        
        float* timeMapped = (float*)hostBufferTime.mapping;
        for(uint32_t i = 0; i < count; ++i) {
            timeMapped[i * 2 + 0] = 1.0e20f; // Scale
            timeMapped[i * 2 + 1] = 0.0f;     // Center
        }
    }

    VkBufferCopy bcMotion{.srcOffset = 0, .dstOffset = 0, .size = bufferSizeMotion};
    vkCmdCopyBuffer(cmd, hostBufferMotion.buffer, motionBuffer.buffer, 1, &bcMotion);
    
    VkBufferCopy bcTime{.srcOffset = 0, .dstOffset = 0, .size = bufferSizeTime};
    vkCmdCopyBuffer(cmd, hostBufferTime.buffer, timeBuffer.buffer, 1, &bcTime);

    buffersToDestroy.push_back(hostBufferMotion);
    buffersToDestroy.push_back(hostBufferTime);
  }

  // Precomputed variance (exp(2*scale)) — saves GPU exp() calls in stochastic GS
  {
    const uint32_t bufferSize = splatCount * 3 * sizeof(float);

    nvvk::Buffer hostBuffer;
    m_alloc->createBuffer(hostBuffer, bufferSize, hostBufferUsageFlags, hostMemoryUsageFlags, hostAllocCreateFlags);
    NVVK_DBG_NAME(hostBuffer.buffer);

    m_alloc->createBuffer(varianceBuffer, bufferSize, deviceBufferUsageFlags, deviceMemoryUsageFlags);
    NVVK_DBG_NAME(varianceBuffer.buffer);

    float* mapped = static_cast<float*>(hostBuffer.mapping);
    START_PAR_LOOP(splatCount, i)
    {
      const auto stride3 = i * 3;
      mapped[stride3 + 0] = std::exp(2.0f * splatSet.scale[stride3 + 0]);
      mapped[stride3 + 1] = std::exp(2.0f * splatSet.scale[stride3 + 1]);
      mapped[stride3 + 2] = std::exp(2.0f * splatSet.scale[stride3 + 2]);
    }
    END_PAR_LOOP()

    VkBufferCopy bc{.srcOffset = 0, .dstOffset = 0, .size = bufferSize};
    vkCmdCopyBuffer(cmd, hostBuffer.buffer, varianceBuffer.buffer, 1, &bc);

    buffersToDestroy.push_back(hostBuffer);
  }

  // TRON PBR material attributes
  {
    const bool    hasMaterial        = splatSet.has_material_data;
    const uint32_t count             = hasMaterial ? splatCount : 1;
    const uint32_t bufferSizeBasecolor = count * 3 * sizeof(float);
    const uint32_t bufferSizeRoughness = count * 1 * sizeof(float);
    const uint32_t bufferSizeMetallic  = count * 1 * sizeof(float);

    nvvk::Buffer hostBufferBasecolor;
    m_alloc->createBuffer(hostBufferBasecolor, bufferSizeBasecolor, hostBufferUsageFlags, hostMemoryUsageFlags, hostAllocCreateFlags);
    NVVK_DBG_NAME(hostBufferBasecolor.buffer);

    nvvk::Buffer hostBufferRoughness;
    m_alloc->createBuffer(hostBufferRoughness, bufferSizeRoughness, hostBufferUsageFlags, hostMemoryUsageFlags, hostAllocCreateFlags);
    NVVK_DBG_NAME(hostBufferRoughness.buffer);

    nvvk::Buffer hostBufferMetallic;
    m_alloc->createBuffer(hostBufferMetallic, bufferSizeMetallic, hostBufferUsageFlags, hostMemoryUsageFlags, hostAllocCreateFlags);
    NVVK_DBG_NAME(hostBufferMetallic.buffer);

    m_alloc->createBuffer(basecolorBuffer, bufferSizeBasecolor, deviceBufferUsageFlags, deviceMemoryUsageFlags);
    NVVK_DBG_NAME(basecolorBuffer.buffer);

    m_alloc->createBuffer(roughnessBuffer, bufferSizeRoughness, deviceBufferUsageFlags, deviceMemoryUsageFlags);
    NVVK_DBG_NAME(roughnessBuffer.buffer);

    m_alloc->createBuffer(metallicBuffer, bufferSizeMetallic, deviceBufferUsageFlags, deviceMemoryUsageFlags);
    NVVK_DBG_NAME(metallicBuffer.buffer);

    if(hasMaterial)
    {
      memcpy(hostBufferBasecolor.mapping, splatSet.basecolor.data(), bufferSizeBasecolor);
      memcpy(hostBufferRoughness.mapping, splatSet.roughness.data(), bufferSizeRoughness);
      memcpy(hostBufferMetallic.mapping, splatSet.metallic.data(), bufferSizeMetallic);
    }
    else
    {
      float* basecolorMapped = (float*)hostBufferBasecolor.mapping;
      float* roughnessMapped = (float*)hostBufferRoughness.mapping;
      float* metallicMapped  = (float*)hostBufferMetallic.mapping;
      for(uint32_t i = 0; i < count; ++i)
      {
        basecolorMapped[i * 3 + 0] = 1.0f;
        basecolorMapped[i * 3 + 1] = 1.0f;
        basecolorMapped[i * 3 + 2] = 1.0f;
        roughnessMapped[i]          = 0.5f;
        metallicMapped[i]           = 0.0f;
      }
    }

    VkBufferCopy bcBasecolor{.srcOffset = 0, .dstOffset = 0, .size = bufferSizeBasecolor};
    vkCmdCopyBuffer(cmd, hostBufferBasecolor.buffer, basecolorBuffer.buffer, 1, &bcBasecolor);

    VkBufferCopy bcRoughness{.srcOffset = 0, .dstOffset = 0, .size = bufferSizeRoughness};
    vkCmdCopyBuffer(cmd, hostBufferRoughness.buffer, roughnessBuffer.buffer, 1, &bcRoughness);

    VkBufferCopy bcMetallic{.srcOffset = 0, .dstOffset = 0, .size = bufferSizeMetallic};
    vkCmdCopyBuffer(cmd, hostBufferMetallic.buffer, metallicBuffer.buffer, 1, &bcMetallic);

    buffersToDestroy.push_back(hostBufferBasecolor);
    buffersToDestroy.push_back(hostBufferRoughness);
    buffersToDestroy.push_back(hostBufferMetallic);
  }

  // covariances (for raster only)
  {
    const uint32_t bufferSize = splatCount * 2 * 3 * sizeof(float);

    // allocate host and device buffers
    nvvk::Buffer hostBuffer;
    m_alloc->createBuffer(hostBuffer, bufferSize, hostBufferUsageFlags, hostMemoryUsageFlags, hostAllocCreateFlags);
    NVVK_DBG_NAME(hostBuffer.buffer);

    m_alloc->createBuffer(covariancesBuffer, bufferSize, deviceBufferUsageFlags, deviceMemoryUsageFlags);
    NVVK_DBG_NAME(covariancesBuffer.buffer);

    // map and fill host buffer
    float* hostBufferMapped = (float*)(hostBuffer.mapping);

    //for(uint32_t splatIdx = 0; splatIdx < splatCount; ++splatIdx)
    START_PAR_LOOP(splatCount, splatIdx)
    {
      const auto stride3 = splatIdx * 3;
      const auto stride4 = splatIdx * 4;
      const auto stride6 = splatIdx * 6;
      glm::vec3  scale{std::exp(splatSet.scale[stride3 + 0]), std::exp(splatSet.scale[stride3 + 1]),
                      std::exp(splatSet.scale[stride3 + 2])};

      glm::quat rotation{splatSet.rotation[stride4 + 0], splatSet.rotation[stride4 + 1], splatSet.rotation[stride4 + 2],
                         splatSet.rotation[stride4 + 3]};
      rotation = glm::normalize(rotation);

      // computes the covariance
      const glm::mat3 scaleMatrix           = glm::mat3(glm::scale(scale));
      const glm::mat3 rotationMatrix        = glm::mat3_cast(rotation);  // where rotation is a quaternion
      const glm::mat3 covarianceMatrix      = rotationMatrix * scaleMatrix;
      glm::mat3       transformedCovariance = covarianceMatrix * glm::transpose(covarianceMatrix);

      hostBufferMapped[stride6 + 0] = glm::value_ptr(transformedCovariance)[0];
      hostBufferMapped[stride6 + 1] = glm::value_ptr(transformedCovariance)[3];
      hostBufferMapped[stride6 + 2] = glm::value_ptr(transformedCovariance)[6];

      hostBufferMapped[stride6 + 3] = glm::value_ptr(transformedCovariance)[4];
      hostBufferMapped[stride6 + 4] = glm::value_ptr(transformedCovariance)[7];
      hostBufferMapped[stride6 + 5] = glm::value_ptr(transformedCovariance)[8];
    }
    END_PAR_LOOP();

    // copy from host buffer to device buffer
    // barrier at the end of this method.
    VkBufferCopy bc{.srcOffset = 0, .dstOffset = 0, .size = bufferSize};
    vkCmdCopyBuffer(cmd, hostBuffer.buffer, covariancesBuffer.buffer, 1, &bc);

    // free host buffer after command execution
    buffersToDestroy.push_back(hostBuffer);

    // memory statistics
    memoryStats.srcCov  = (splatCount * (4 + 3)) * sizeof(float);
    memoryStats.odevCov = bufferSize;  // no compression
    memoryStats.devCov  = bufferSize;  // covariance takes less space than rotation + scale
  }

  // Colors. SH degree 0 is not view dependent, so we directly transform to base color
  // this will make some economy of processing in the shader at each frame
  {
    const uint32_t bufferSize = splatCount * 4 * sizeof(float);

    // allocate host and device buffers
    nvvk::Buffer hostBuffer;
    m_alloc->createBuffer(hostBuffer, bufferSize, hostBufferUsageFlags, hostMemoryUsageFlags, hostAllocCreateFlags);
    NVVK_DBG_NAME(hostBuffer.buffer);

    m_alloc->createBuffer(colorsBuffer, bufferSize, deviceBufferUsageFlags, deviceMemoryUsageFlags);
    NVVK_DBG_NAME(colorsBuffer.buffer);

    // fill host buffer
    float* hostBufferMapped = (float*)(hostBuffer.mapping);

    //for(uint32_t splatIdx = 0; splatIdx < splatCount; ++splatIdx)
    START_PAR_LOOP(splatCount, splatIdx)
    {
      const auto  stride3           = splatIdx * 3;
      const auto  stride4           = splatIdx * 4;
      const float SH_C0             = 0.28209479177387814f;
      hostBufferMapped[stride4 + 0] = glm::clamp(0.5f + SH_C0 * splatSet.f_dc[stride3 + 0], 0.0f, 1.0f);
      hostBufferMapped[stride4 + 1] = glm::clamp(0.5f + SH_C0 * splatSet.f_dc[stride3 + 1], 0.0f, 1.0f);
      hostBufferMapped[stride4 + 2] = glm::clamp(0.5f + SH_C0 * splatSet.f_dc[stride3 + 2], 0.0f, 1.0f);
      hostBufferMapped[stride4 + 3] = glm::clamp(1.0f / (1.0f + std::exp(-splatSet.opacity[splatIdx])), 0.0f, 1.0f);
    }
    END_PAR_LOOP()

    // copy from host buffer to device buffer
    // barrier at the end of this method.
    VkBufferCopy bc{.srcOffset = 0, .dstOffset = 0, .size = bufferSize};
    vkCmdCopyBuffer(cmd, hostBuffer.buffer, colorsBuffer.buffer, 1, &bc);

    // free host buffer after command execution
    buffersToDestroy.push_back(hostBuffer);

    // memory statistics
    memoryStats.srcSh0  = bufferSize;
    memoryStats.odevSh0 = bufferSize;
    memoryStats.devSh0  = bufferSize;
  }

  // Spherical harmonics of degree 1 to 3
  if(!splatSet.f_rest.empty())
  {
    const uint32_t totalSphericalHarmonicsComponentCount    = (uint32_t)splatSet.f_rest.size() / splatCount;
    const uint32_t sphericalHarmonicsCoefficientsPerChannel = totalSphericalHarmonicsComponentCount / 3;
    // find the maximum SH degree stored in the file
    int sphericalHarmonicsDegree = 0;
    int splatStride              = 0;
    if(sphericalHarmonicsCoefficientsPerChannel >= 3)
    {
      sphericalHarmonicsDegree = 1;
      splatStride += 3 * 3;
    }
    if(sphericalHarmonicsCoefficientsPerChannel >= 8)
    {
      sphericalHarmonicsDegree = 2;
      splatStride += 5 * 3;
    }
    if(sphericalHarmonicsCoefficientsPerChannel == 15)
    {
      sphericalHarmonicsDegree = 3;
      splatStride += 7 * 3;
    }

    // same for the time beeing, would be less if we do not upload all src degrees
    int targetSplatStride = splatStride;

    // allocate host and device buffers
    const uint32_t bufferSize = splatCount * splatStride * formatSize(m_format);

    nvvk::Buffer hostBuffer;
    m_alloc->createBuffer(hostBuffer, bufferSize, hostBufferUsageFlags, hostMemoryUsageFlags, hostAllocCreateFlags);
    NVVK_DBG_NAME(hostBuffer.buffer);

    m_alloc->createBuffer(sphericalHarmonicsBuffer, bufferSize, deviceBufferUsageFlags, deviceMemoryUsageFlags);
    NVVK_DBG_NAME(sphericalHarmonicsBuffer.buffer);

    // fill host buffer
    float* hostBufferMapped = (float*)(hostBuffer.mapping);

    auto startShTime = std::chrono::high_resolution_clock::now();

    // for(uint32_t splatIdx = 0; splatIdx < splatCount; ++splatIdx)
    START_PAR_LOOP(splatCount, splatIdx)
    {
      const auto srcBase   = splatStride * splatIdx;
      const auto destBase  = targetSplatStride * splatIdx;
      int        dstOffset = 0;
      // degree 1, three coefs per component
      for(auto i = 0; i < 3; i++)
      {
        for(auto rgb = 0; rgb < 3; rgb++)
        {
          const auto srcIndex = srcBase + (sphericalHarmonicsCoefficientsPerChannel * rgb + i);
          const auto dstIndex = destBase + dstOffset++;  // inc after add

          storeSh(m_format, splatSet.f_rest.data(), srcIndex, hostBufferMapped, dstIndex);
        }
      }
      // degree 2, five coefs per component
      for(auto i = 0; i < 5; i++)
      {
        for(auto rgb = 0; rgb < 3; rgb++)
        {
          const auto srcIndex = srcBase + (sphericalHarmonicsCoefficientsPerChannel * rgb + 3 + i);
          const auto dstIndex = destBase + dstOffset++;  // inc after add

          storeSh(m_format, splatSet.f_rest.data(), srcIndex, hostBufferMapped, dstIndex);
        }
      }
      // degree 3, seven coefs per component
      for(auto i = 0; i < 7; i++)
      {
        for(auto rgb = 0; rgb < 3; rgb++)
        {
          const auto srcIndex = srcBase + (sphericalHarmonicsCoefficientsPerChannel * rgb + 3 + 5 + i);
          const auto dstIndex = destBase + dstOffset++;  // inc after add

          storeSh(m_format, splatSet.f_rest.data(), srcIndex, hostBufferMapped, dstIndex);
        }
      }
    }
    END_PAR_LOOP()

    auto      endShTime   = std::chrono::high_resolution_clock::now();
    long long buildShTime = std::chrono::duration_cast<std::chrono::milliseconds>(endShTime - startShTime).count();
    LOGI("Sh data updated in %lldms\n", buildShTime);

    // copy from host buffer to device buffer
    // barrier at the end of this method.
    VkBufferCopy bc{.srcOffset = 0, .dstOffset = 0, .size = bufferSize};
    vkCmdCopyBuffer(cmd, hostBuffer.buffer, sphericalHarmonicsBuffer.buffer, 1, &bc);

    // free host buffer after command execution
    buffersToDestroy.push_back(hostBuffer);

    // memory statistics
    memoryStats.srcShOther  = (uint32_t)splatSet.f_rest.size() * sizeof(float);
    memoryStats.odevShOther = bufferSize;  // no compression or quantization
    memoryStats.devShOther  = bufferSize;
  }

  // sync with end of copy to device
  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT,
                       0, 1, &barrier, 0, NULL, 0, NULL);

  m_app->submitAndWaitTempCmdBuffer(cmd);

  // free temp buffers
  for(auto& buffer : buffersToDestroy)
  {
    m_alloc->destroyBuffer(buffer);
  }

  // update statistics totals
  memoryStats.srcShAll  = memoryStats.srcSh0 + memoryStats.srcShOther;
  memoryStats.odevShAll = memoryStats.odevSh0 + memoryStats.odevShOther;
  memoryStats.devShAll  = memoryStats.devSh0 + memoryStats.devShOther;

  memoryStats.srcAll  = memoryStats.srcCenters + memoryStats.srcCov + memoryStats.srcSh0 + memoryStats.srcShOther;
  memoryStats.odevAll = memoryStats.odevCenters + memoryStats.odevCov + memoryStats.odevSh0 + memoryStats.odevShOther;
  memoryStats.devAll  = memoryStats.devCenters + memoryStats.devCov + memoryStats.devSh0 + memoryStats.devShOther;

  auto      endTime   = std::chrono::high_resolution_clock::now();
  long long buildTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
  LOGI("Data buffers updated in %lldms\n", buildTime);
}

void SplatSetVk::deinitDataBuffers()
{
  m_alloc->destroyBuffer(centersBuffer);
  m_alloc->destroyBuffer(scalesBuffer);
  m_alloc->destroyBuffer(rotationsBuffer);
  m_alloc->destroyBuffer(colorsBuffer);
  m_alloc->destroyBuffer(covariancesBuffer);
  m_alloc->destroyBuffer(sphericalHarmonicsBuffer);
  
  m_alloc->destroyBuffer(motionBuffer);
  m_alloc->destroyBuffer(timeBuffer);

  m_alloc->destroyBuffer(basecolorBuffer);
  m_alloc->destroyBuffer(roughnessBuffer);
  m_alloc->destroyBuffer(metallicBuffer);
  m_alloc->destroyBuffer(varianceBuffer);
}

///////////////////
// using texture maps to store splatset in VRAM

void SplatSetVk::initDataTextures(SplatSet& splatSet)
{
  auto startTime = std::chrono::high_resolution_clock::now();

  const auto splatCount = (uint32_t)splatSet.positions.size() / 3;

  // centers (3 components but texture map is only allowed with 4 components)
  // TODO: May pack as done for covariances not to waste alpha chanel ? but must
  // compare performance (1 lookup vs 2 lookups due to packing)
  {
    glm::ivec2         centersMapSize = computeDataTextureSize(3, 3, splatCount);
    std::vector<float> centers(centersMapSize.x * centersMapSize.y * 4);  // includes some padding and unused w channel

    glm::ivec2         scalesMapSize = computeDataTextureSize(3, 3, splatCount);
    std::vector<float> scales(scalesMapSize.x * scalesMapSize.y * 4);  // includes some padding and unused w channel

    glm::ivec2         rotationsMapSize = computeDataTextureSize(4, 4, splatCount);
    std::vector<float> rotations(splatSet.rotation);
    rotations.resize(rotationsMapSize.x * rotationsMapSize.y * 4);  // includes some padding

    //for(uint32_t i = 0; i < splatCount; ++i)
    START_PAR_LOOP(splatCount, splatIdx)
    {
      // we skip the alpha channel that is left undefined and not used in the shader
      for(uint32_t cmp = 0; cmp < 3; ++cmp)
      {
        centers[splatIdx * 4 + cmp] = splatSet.positions[splatIdx * 3 + cmp];
        scales[splatIdx * 4 + cmp]  = splatSet.scale[splatIdx * 3 + cmp];
      }
    }
    END_PAR_LOOP()

    // place the result in the dedicated texture map
    initTexture(centersMapSize.x, centersMapSize.y, (uint32_t)centers.size() * sizeof(float), (void*)centers.data(),
                VK_FORMAT_R32G32B32A32_SFLOAT, *m_sampler, centersMap);

    initTexture(scalesMapSize.x, scalesMapSize.y, (uint32_t)scales.size() * sizeof(float), (void*)scales.data(),
                VK_FORMAT_R32G32B32A32_SFLOAT, *m_sampler, scalesMap);

    initTexture(rotationsMapSize.x, rotationsMapSize.y, (uint32_t)rotations.size() * sizeof(float),
                (void*)rotations.data(), VK_FORMAT_R32G32B32A32_SFLOAT, *m_sampler, rotationsMap);

    // memory statistics
    memoryStats.srcCenters  = splatCount * 3 * sizeof(float);
    memoryStats.odevCenters = splatCount * 3 * sizeof(float);  // no compression or quantization yet
    memoryStats.devCenters  = centersMapSize.x * centersMapSize.y * 4 * sizeof(float);
  }
  
  {
    const bool hasTime = splatSet.has_time_data;
    glm::ivec2 mapSize = hasTime ? computeDataTextureSize(3, 3, splatCount) : glm::ivec2(1, 1);
    std::vector<float> motion(mapSize.x * mapSize.y * 4, 0.0f);
    
    glm::ivec2 timeMapSize = hasTime ? computeDataTextureSize(2, 2, splatCount) : glm::ivec2(1, 1);
    std::vector<float> timeData(timeMapSize.x * timeMapSize.y * 2, 0.0f);
    
    if (hasTime) {
        START_PAR_LOOP(splatCount, i)
        {
          motion[i * 4 + 0] = splatSet.motion[i * 3 + 0];
          motion[i * 4 + 1] = splatSet.motion[i * 3 + 1];
          motion[i * 4 + 2] = splatSet.motion[i * 3 + 2];
          
          // JS Order: x = scale, y = center
          timeData[i * 2 + 0] = splatSet.time_scale[i];
          timeData[i * 2 + 1] = splatSet.time[i];
        }
        END_PAR_LOOP()
    } else {
        timeData[0] = 1.0e20f; // Scale
        timeData[1] = 0.0f;     // Center
    }
    
    initTexture(mapSize.x, mapSize.y, (uint32_t)motion.size() * sizeof(float), (void*)motion.data(),
                VK_FORMAT_R32G32B32A32_SFLOAT, *m_sampler, motionMap);
                
    initTexture(timeMapSize.x, timeMapSize.y, (uint32_t)timeData.size() * sizeof(float), (void*)timeData.data(),
                VK_FORMAT_R32G32_SFLOAT, *m_sampler, timeMap);
  }

  // TRON PBR material attributes
  {
    const bool hasMaterial = splatSet.has_material_data;

    glm::ivec2 basecolorMapSize = hasMaterial ? computeDataTextureSize(3, 3, splatCount) : glm::ivec2(1, 1);
    std::vector<float> basecolor(basecolorMapSize.x * basecolorMapSize.y * 4, 0.0f);

    glm::ivec2 roughnessMapSize = hasMaterial ? computeDataTextureSize(1, 1, splatCount) : glm::ivec2(1, 1);
    std::vector<float> roughness(roughnessMapSize.x * roughnessMapSize.y * 1, 0.0f);

    glm::ivec2 metallicMapSize = hasMaterial ? computeDataTextureSize(1, 1, splatCount) : glm::ivec2(1, 1);
    std::vector<float> metallicData(metallicMapSize.x * metallicMapSize.y * 1, 0.0f);

    if(hasMaterial)
    {
      START_PAR_LOOP(splatCount, i)
      {
        basecolor[i * 4 + 0] = splatSet.basecolor[i * 3 + 0];
        basecolor[i * 4 + 1] = splatSet.basecolor[i * 3 + 1];
        basecolor[i * 4 + 2] = splatSet.basecolor[i * 3 + 2];

        roughness[i]     = splatSet.roughness[i];
        metallicData[i]  = splatSet.metallic[i];
      }
      END_PAR_LOOP()
    }
    else
    {
      basecolor[0] = 1.0f;
      basecolor[1] = 1.0f;
      basecolor[2] = 1.0f;
      roughness[0] = 0.5f;
      metallicData[0] = 0.0f;
    }

    initTexture(basecolorMapSize.x, basecolorMapSize.y, (uint32_t)basecolor.size() * sizeof(float), (void*)basecolor.data(),
                VK_FORMAT_R32G32B32A32_SFLOAT, *m_sampler, basecolorMap);

    initTexture(roughnessMapSize.x, roughnessMapSize.y, (uint32_t)roughness.size() * sizeof(float), (void*)roughness.data(),
                VK_FORMAT_R32_SFLOAT, *m_sampler, roughnessMap);

    initTexture(metallicMapSize.x, metallicMapSize.y, (uint32_t)metallicData.size() * sizeof(float), (void*)metallicData.data(),
                VK_FORMAT_R32_SFLOAT, *m_sampler, metallicMap);
  }

  // covariances
  {
    glm::ivec2         mapSize = computeDataTextureSize(4, 6, splatCount);
    std::vector<float> covariances(mapSize.x * mapSize.y * 4, 0.0f);
    //for(uint32_t splatIdx = 0; splatIdx < splatCount; ++splatIdx)
    START_PAR_LOOP(splatCount, splatIdx)
    {
      const auto stride3 = splatIdx * 3;
      const auto stride4 = splatIdx * 4;
      const auto stride6 = splatIdx * 6;
      glm::vec3  scale{std::exp(splatSet.scale[stride3 + 0]), std::exp(splatSet.scale[stride3 + 1]),
                      std::exp(splatSet.scale[stride3 + 2])};

      glm::quat rotation{splatSet.rotation[stride4 + 0], splatSet.rotation[stride4 + 1], splatSet.rotation[stride4 + 2],
                         splatSet.rotation[stride4 + 3]};
      rotation = glm::normalize(rotation);

      // computes the covariance
      const glm::mat3 scaleMatrix           = glm::mat3(glm::scale(scale));
      const glm::mat3 rotationMatrix        = glm::mat3_cast(rotation);  // where rotation is a quaternion
      const glm::mat3 covarianceMatrix      = rotationMatrix * scaleMatrix;
      glm::mat3       transformedCovariance = covarianceMatrix * glm::transpose(covarianceMatrix);

      covariances[stride6 + 0] = glm::value_ptr(transformedCovariance)[0];
      covariances[stride6 + 1] = glm::value_ptr(transformedCovariance)[3];
      covariances[stride6 + 2] = glm::value_ptr(transformedCovariance)[6];

      covariances[stride6 + 3] = glm::value_ptr(transformedCovariance)[4];
      covariances[stride6 + 4] = glm::value_ptr(transformedCovariance)[7];
      covariances[stride6 + 5] = glm::value_ptr(transformedCovariance)[8];
    }
    END_PAR_LOOP()

    // place the result in the dedicated texture map
    initTexture(mapSize.x, mapSize.y, (uint32_t)covariances.size() * sizeof(float), (void*)covariances.data(),
                VK_FORMAT_R32G32B32A32_SFLOAT, *m_sampler, covariancesMap);
    // memory statistics
    memoryStats.srcCov  = (splatCount * (4 + 3)) * sizeof(float);
    memoryStats.odevCov = splatCount * 6 * sizeof(float);  // covariance takes less space than rotation + scale
    memoryStats.devCov  = mapSize.x * mapSize.y * 4 * sizeof(float);
  }
  // SH degree 0 is not view dependent, so we directly transform to base color
  // this will make some economy of processing in the shader at each frame
  {
    glm::ivec2           mapSize = computeDataTextureSize(4, 4, splatCount);
    std::vector<uint8_t> colors(mapSize.x * mapSize.y * 4);  // includes some padding
    //for(uint32_t splatIdx = 0; splatIdx < splatCount; ++splatIdx)
    START_PAR_LOOP(splatCount, splatIdx)
    {
      const auto  stride3 = splatIdx * 3;
      const auto  stride4 = splatIdx * 4;
      const float SH_C0   = 0.28209479177387814f;
      colors[stride4 + 0] = (uint8_t)glm::clamp(std::floor((0.5f + SH_C0 * splatSet.f_dc[stride3 + 0]) * 255), 0.0f, 255.0f);
      colors[stride4 + 1] = (uint8_t)glm::clamp(std::floor((0.5f + SH_C0 * splatSet.f_dc[stride3 + 1]) * 255), 0.0f, 255.0f);
      colors[stride4 + 2] = (uint8_t)glm::clamp(std::floor((0.5f + SH_C0 * splatSet.f_dc[stride3 + 2]) * 255), 0.0f, 255.0f);
      colors[stride4 + 3] =
          (uint8_t)glm::clamp(std::floor((1.0f / (1.0f + std::exp(-splatSet.opacity[splatIdx]))) * 255), 0.0f, 255.0f);
    }
    END_PAR_LOOP()
    // place the result in the dedicated texture map
    initTexture(mapSize.x, mapSize.y, (uint32_t)colors.size(), (void*)colors.data(), VK_FORMAT_R8G8B8A8_UNORM, *m_sampler, colorsMap);
    // memory statistics
    memoryStats.srcSh0  = splatCount * 4 * sizeof(float);  // original sh0 and opacity are floats
    memoryStats.odevSh0 = splatCount * 4 * sizeof(uint8_t);
    memoryStats.devSh0  = mapSize.x * mapSize.y * 4 * sizeof(uint8_t);
  }
  // Prepare the spherical harmonics of degree 1 to 3
  if(!splatSet.f_rest.empty())
  {
    const uint32_t sphericalHarmonicsElementsPerTexel       = 4;
    const uint32_t totalSphericalHarmonicsComponentCount    = (uint32_t)splatSet.f_rest.size() / splatCount;
    const uint32_t sphericalHarmonicsCoefficientsPerChannel = totalSphericalHarmonicsComponentCount / 3;
    // find the maximum SH degree stored in the file
    int sphericalHarmonicsDegree = 0;
    if(sphericalHarmonicsCoefficientsPerChannel >= 3)
      sphericalHarmonicsDegree = 1;
    if(sphericalHarmonicsCoefficientsPerChannel >= 8)
      sphericalHarmonicsDegree = 2;
    if(sphericalHarmonicsCoefficientsPerChannel >= 15)
      sphericalHarmonicsDegree = 3;

    // add some padding at each splat if needed for easy texture lookups
    int sphericalHarmonicsComponentCount = 0;
    if(sphericalHarmonicsDegree == 1)
      sphericalHarmonicsComponentCount = 9;
    if(sphericalHarmonicsDegree == 2)
      sphericalHarmonicsComponentCount = 24;
    if(sphericalHarmonicsDegree == 3)
      sphericalHarmonicsComponentCount = 45;

    int paddedSphericalHarmonicsComponentCount = sphericalHarmonicsComponentCount;
    while(paddedSphericalHarmonicsComponentCount % 4 != 0)
      paddedSphericalHarmonicsComponentCount++;

    glm::ivec2 mapSize =
        computeDataTextureSize(sphericalHarmonicsElementsPerTexel, paddedSphericalHarmonicsComponentCount, splatCount);

    const uint32_t bufferSize = mapSize.x * mapSize.y * sphericalHarmonicsElementsPerTexel * formatSize(m_format);

    std::vector<uint8_t> paddedSHArray(bufferSize, 0);

    void* data = (void*)paddedSHArray.data();

    //for(uint32_t splatIdx = 0; splatIdx < splatCount; ++splatIdx)
    START_PAR_LOOP(splatCount, splatIdx)
    {
      const auto srcBase   = totalSphericalHarmonicsComponentCount * splatIdx;
      const auto destBase  = paddedSphericalHarmonicsComponentCount * splatIdx;
      int        dstOffset = 0;
      // degree 1, three coefs per component
      for(auto i = 0; i < 3; i++)
      {
        for(auto rgb = 0; rgb < 3; rgb++)
        {
          const auto srcIndex = srcBase + (sphericalHarmonicsCoefficientsPerChannel * rgb + i);
          const auto dstIndex = destBase + dstOffset++;  // inc after add

          storeSh(m_format, splatSet.f_rest.data(), srcIndex, data, dstIndex);
        }
      }

      // degree 2, five coefs per component
      for(auto i = 0; i < 5; i++)
      {
        for(auto rgb = 0; rgb < 3; rgb++)
        {
          const auto srcIndex = srcBase + (sphericalHarmonicsCoefficientsPerChannel * rgb + 3 + i);
          const auto dstIndex = destBase + dstOffset++;  // inc after add

          storeSh(m_format, splatSet.f_rest.data(), srcIndex, data, dstIndex);
        }
      }
      // degree 3, seven coefs per component
      for(auto i = 0; i < 7; i++)
      {
        for(auto rgb = 0; rgb < 3; rgb++)
        {
          const auto srcIndex = srcBase + (sphericalHarmonicsCoefficientsPerChannel * rgb + 3 + 5 + i);
          const auto dstIndex = destBase + dstOffset++;  // inc after add

          storeSh(m_format, splatSet.f_rest.data(), srcIndex, data, dstIndex);
        }
      }
    }
    END_PAR_LOOP()

    // place the result in the dedicated texture map
    if(m_format == FORMAT_FLOAT32)
    {
      initTexture(mapSize.x, mapSize.y, bufferSize, data, VK_FORMAT_R32G32B32A32_SFLOAT, *m_sampler, sphericalHarmonicsMap);
    }
    else if(m_format == FORMAT_FLOAT16)
    {
      initTexture(mapSize.x, mapSize.y, bufferSize, data, VK_FORMAT_R16G16B16A16_SFLOAT, *m_sampler, sphericalHarmonicsMap);
    }
    else if(m_format == FORMAT_UINT8)
    {
      initTexture(mapSize.x, mapSize.y, bufferSize, data, VK_FORMAT_R8G8B8A8_UNORM, *m_sampler, sphericalHarmonicsMap);
    }

    // memory statistics
    memoryStats.srcShOther  = (uint32_t)splatSet.f_rest.size() * sizeof(float);
    memoryStats.odevShOther = (uint32_t)splatSet.f_rest.size() * formatSize(m_format);
    memoryStats.devShOther  = bufferSize;
  }

  // update statistics totals
  memoryStats.srcShAll  = memoryStats.srcSh0 + memoryStats.srcShOther;
  memoryStats.odevShAll = memoryStats.odevSh0 + memoryStats.odevShOther;
  memoryStats.devShAll  = memoryStats.devSh0 + memoryStats.devShOther;

  memoryStats.srcAll  = memoryStats.srcCenters + memoryStats.srcCov + memoryStats.srcSh0 + memoryStats.srcShOther;
  memoryStats.odevAll = memoryStats.odevCenters + memoryStats.odevCov + memoryStats.odevSh0 + memoryStats.odevShOther;
  memoryStats.devAll  = memoryStats.devCenters + memoryStats.devCov + memoryStats.devSh0 + memoryStats.devShOther;

  auto      endTime   = std::chrono::high_resolution_clock::now();
  long long buildTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
  LOGI("Data textures updated in %lldms\n", buildTime);
}

void SplatSetVk::deinitDataTextures()
{
  deinitTexture(centersMap);
  deinitTexture(scalesMap);
  deinitTexture(rotationsMap);
  deinitTexture(covariancesMap);

  deinitTexture(colorsMap);
  deinitTexture(sphericalHarmonicsMap);
  
  deinitTexture(motionMap);
  deinitTexture(timeMap);

  deinitTexture(basecolorMap);
  deinitTexture(roughnessMap);
  deinitTexture(metallicMap);
}

void SplatSetVk::initTexture(uint32_t width, uint32_t height, uint32_t bufsize, void* data, VkFormat format, const VkSampler& sampler, nvvk::Image& texture)
{
  const VkImageLayout imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

  VkImageCreateInfo createInfo = DEFAULT_VkImageCreateInfo;
  createInfo.mipLevels         = 1;
  createInfo.extent            = {width, height, 1};
  createInfo.usage             = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  createInfo.format            = format;

  VkCommandBuffer cmd = m_app->createTempCmdBuffer();
  NVVK_CHECK(m_alloc->createImage(texture, createInfo, DEFAULT_VkImageViewCreateInfo));
  NVVK_DBG_NAME(texture.image);
  NVVK_DBG_NAME(texture.descriptor.imageView);

  NVVK_CHECK(m_uploader->appendImage(texture, std::span<uint8_t>((uint8_t*)data, bufsize), imageLayout));
  m_uploader->cmdUploadAppended(cmd);

  texture.descriptor.sampler = sampler;

  m_app->submitAndWaitTempCmdBuffer(cmd);
}

void SplatSetVk::deinitTexture(nvvk::Image& texture)
{
  m_alloc->destroyImage(texture);
}

/////////////// section related to RTX


// Comment from Paper code
//
// phi = golden ratio = (1 + sqrt(5)) / 2
// r = radius of the inscribed circle = 1 = (phi^2 * s) / ( 2 * sqrt(3))
// s = edge length = ( 2 * sqrt(3) ) / phi^2
// V = (5/12) * ( 3 + sqrt(5) ) * s^3 = 8.0
constexpr float goldenRatio   = 1.618033988749895f;  // cast to float discards a bit of precision
constexpr float icosaEdge     = 1.323169076499215f;  // cast to float discards a bit of precision
constexpr float icosaVrtScale = 0.5 * icosaEdge;

float kernelScale(float density, float modulatedMinResponse, float kernelDegree, bool adaptiveClamping)
{
  const float responseModulation = adaptiveClamping ? density : 1.0f;
  const float minResponse        = fminf(modulatedMinResponse / responseModulation, 0.97f);

  // linear kernel
  if(kernelDegree == 0)
  {
    return ((1.0f - minResponse) / 3.0f) / -0.329630334487f;
  }

  /// generalized gaussian of degree b : scaling a = -4.5/3^b
  /// e^{a*|x|^b}
  const float b = kernelDegree;
  const float a = -4.5f / powf(3.0f, static_cast<float>(b));
  /// find distance r (>0) st e^{a*r^b} = minResponse
  /// TODO : reshuffle the math to call powf only once
  return powf(logf(minResponse) / a, 1.0f / b);
}

glm::mat4 SplatSetVk::rtxComputeTransformMatrix(SplatSet& splatSet, uint64_t splatIdx)
{
  const auto stride3 = splatIdx * 3;
  const auto stride4 = splatIdx * 4;

  // compute the transformation matrix
  glm::vec3 scale{std::exp(splatSet.scale[stride3 + 0]), std::exp(splatSet.scale[stride3 + 1]),
                  std::exp(splatSet.scale[stride3 + 2])};

  glm::quat rotation{splatSet.rotation[stride4 + 0], splatSet.rotation[stride4 + 1], splatSet.rotation[stride4 + 2],
                     splatSet.rotation[stride4 + 3]};
  rotation = glm::normalize(rotation);

  glm::vec3 position{splatSet.positions[stride3 + 0], splatSet.positions[stride3 + 1], splatSet.positions[stride3 + 2]};

  const float density = 1.0f / (1.0f + std::exp(-splatSet.opacity[splatIdx]));

  const float kerScale = kernelScale(density, m_rtxKernelMinResponse, float(m_rtxKernelDegree), m_rtxKernelAdaptiveClamping);

  const glm::vec3 totalScale = scale * icosaVrtScale * kerScale;

  const glm::mat4 translateMatrix = glm::translate(position);
  const glm::mat4 scaleMatrix     = glm::scale(totalScale);
  const glm::mat4 rotationMatrix  = glm::mat4_cast(rotation);  // where rotation is a quaternion

  return transform * translateMatrix * rotationMatrix * scaleMatrix;
}

// transformed unit regular icosahedron
void SplatSetVk::rtxCreateSplatIcosahedron(std::vector<glm::vec3>& vertices,
                                           std::vector<uint32_t>&  indices,
                                           std::vector<SplatAabb>& aabbs,
                                           glm::mat4               transform)  // = glm::mat4(1.0))
{
  // Golden ratio for icosahedron
  const float t = (1.0f + sqrt(5.0f)) / 2.0f;

  // Define the 12 vertices of an icosahedron
  static const std::vector<glm::vec4> s_vertices = {{-1, t, 0, 1},  {1, t, 0, 1},   {0, 1, -t, 1}, {-t, 0, -1, 1},
                                                    {-t, 0, 1, 1},  {0, 1, t, 1},   {t, 0, 1, 1},  {0, -1, t, 1},
                                                    {-1, -t, 0, 1}, {0, -1, -t, 1}, {t, 0, -1, 1}, {1, -t, 0, 1}};

  // Define the 20 triangular faces by vertex indices
  static const std::vector<uint32_t> s_indices = {0,  1, 2, 0,  2, 3,  0,  3, 4,  0, 4,  5, 0,  5,  1, 6, 1, 5, 6,  5,
                                                  7,  6, 7, 11, 6, 11, 10, 6, 10, 1, 8,  4, 3,  8,  3, 9, 8, 9, 11, 8,
                                                  11, 7, 8, 7,  4, 9,  3,  2, 9,  2, 10, 9, 10, 11, 5, 4, 7, 1, 10, 2};

  SplatAabb aabb{.minimum = glm::vec3(std::numeric_limits<float>::max()), .maximum = glm::vec3(std::numeric_limits<float>::min())};

  const auto vertexOffset = vertices.size();
  const auto indexOffset  = indices.size();

  vertices.resize(vertices.size() + s_vertices.size());
  indices.resize(indices.size() + s_indices.size());

  for(auto i = 0; i < s_vertices.size(); ++i)
  {
    const auto pos             = glm::vec3(transform * s_vertices[i]);
    vertices[vertexOffset + i] = pos;
    aabb.maximum               = glm::max(pos, aabb.maximum);
    aabb.minimum               = glm::min(pos, aabb.minimum);
  }
  aabbs.push_back(aabb);
  for(auto i = 0; i < s_indices.size(); ++i)
  {
    indices[indexOffset + i] = uint32_t(vertexOffset) + s_indices[i];
  }
}

void SplatSetVk::rtxInitSplatModel(SplatSet& splatSet, bool useInstances, bool useAABBs, bool useSpheres, bool compressBlas, int kernelDegree, float kernelMinResponse, bool kernelAdaptiveClamping)
{
  // stored for later use by rtxComputeTransformMatrix and rtxInitAccelerationStructures
  m_rtxUseAABBs               = useAABBs;
  m_rtxUseSpheres             = useSpheres;
  m_rtxUseInstances           = useInstances;
  m_rtxCompressBlas           = compressBlas;
  m_rtxKernelDegree           = kernelDegree;
  m_rtxKernelMinResponse      = kernelMinResponse;
  m_rtxKernelAdaptiveClamping = kernelAdaptiveClamping;

  std::vector<glm::vec3> vertices;
  std::vector<uint32_t>  indices;
  std::vector<SplatAabb> aabbs;

  // Sphere data for Blackwell native primitives
  std::vector<glm::vec3> sphereCenters;
  std::vector<float>     sphereRadii;

  if(useInstances)
  {
    rtxCreateSplatIcosahedron(vertices, indices, aabbs);

    // For sphere mode with instances, create a unit sphere at origin
    // The TLAS instance transform will scale/position it per-splat
    if(useSpheres)
    {
      sphereCenters.push_back(glm::vec3(0.0f));
      // Radius 3.0 = 3σ coverage for Gaussian density
      sphereRadii.push_back(3.0f);
    }
  }
  else
  {
    // TODO: change method to not use push_back but pre-allocate, output will allow for Parallel loop
    const uint64_t splatCount = splatSet.size();
    for(auto splatIdx = 0; splatIdx < splatCount; ++splatIdx)
    {
      const glm::mat4 transform = rtxComputeTransformMatrix(splatSet, splatIdx);
      rtxCreateSplatIcosahedron(vertices, indices, aabbs, transform);
    }
  }

  // Store vertex and index count and aabbs
  m_splatModel.nbVertices = static_cast<uint32_t>(vertices.size());
  m_splatModel.nbIndices  = static_cast<uint32_t>(indices.size());
  m_splatModel.nbAABB     = static_cast<uint32_t>(aabbs.size());
  m_splatModel.nbSpheres  = static_cast<uint32_t>(sphereCenters.size());

  // Create the buffers on Device and copy vertices and indices
  VkCommandBuffer    cmd             = m_app->createTempCmdBuffer();
  VkBufferUsageFlags flag            = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  VkBufferUsageFlags rayTracingFlags =  // used also for building acceleration structures
      flag | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

  // indices
  NVVK_CHECK(m_alloc->createBuffer(m_splatModel.vertexBuffer, vertices.size() * sizeof(glm::vec3),
                                   VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | rayTracingFlags));
  NVVK_CHECK(m_uploader->appendBuffer(m_splatModel.vertexBuffer, 0, std::span(vertices)));
  NVVK_DBG_NAME(m_splatModel.vertexBuffer.buffer);

  // vertices
  NVVK_CHECK(m_alloc->createBuffer(m_splatModel.indexBuffer, indices.size() * sizeof(uint32_t),
                                   VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | rayTracingFlags));
  NVVK_CHECK(m_uploader->appendBuffer(m_splatModel.indexBuffer, 0, std::span(indices)));
  NVVK_DBG_NAME(m_splatModel.indexBuffer.buffer);

  // aabbs
  NVVK_CHECK(m_alloc->createBuffer(m_splatModel.aabbBuffer, aabbs.size() * sizeof(SplatAabb),
                                   VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | rayTracingFlags));
  NVVK_CHECK(m_uploader->appendBuffer(m_splatModel.aabbBuffer, 0, std::span(aabbs)));
  NVVK_DBG_NAME(m_splatModel.aabbBuffer.buffer);

  // Sphere buffers for Blackwell native primitives
  if(useSpheres && !sphereCenters.empty())
  {
    NVVK_CHECK(m_alloc->createBuffer(m_splatModel.sphereCenterBuffer, sphereCenters.size() * sizeof(glm::vec3),
                                     rayTracingFlags));
    NVVK_CHECK(m_uploader->appendBuffer(m_splatModel.sphereCenterBuffer, 0, std::span(sphereCenters)));
    NVVK_DBG_NAME(m_splatModel.sphereCenterBuffer.buffer);

    NVVK_CHECK(m_alloc->createBuffer(m_splatModel.sphereRadiusBuffer, sphereRadii.size() * sizeof(float),
                                     rayTracingFlags));
    NVVK_CHECK(m_uploader->appendBuffer(m_splatModel.sphereRadiusBuffer, 0, std::span(sphereRadii)));
    NVVK_DBG_NAME(m_splatModel.sphereRadiusBuffer.buffer);

    LOGI("Sphere primitives created: %u spheres (Blackwell native mode)\n", m_splatModel.nbSpheres);
  }

  m_uploader->cmdUploadAppended(cmd);
  m_app->submitAndWaitTempCmdBuffer(cmd);
  m_uploader->releaseStaging();
}

nvvk::AccelerationStructureGeometryInfo SplatSetVk::rtxCreateSplatModelAccelerationStructureGeometryInfo()
{
  // Use Blackwell native sphere primitives (VK_NV_ray_tracing_linear_swept_spheres)
  if(m_rtxUseSpheres && m_splatModel.nbSpheres > 0)
  {
    // Sphere geometry data - hardware ray-sphere intersection
    VkAccelerationStructureGeometrySpheresDataNV spheres{};
    spheres.sType                  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_SPHERES_DATA_NV;
    spheres.vertexFormat           = VK_FORMAT_R32G32B32_SFLOAT;  // vec3 sphere center
    spheres.vertexData.deviceAddress = m_splatModel.sphereCenterBuffer.address;
    spheres.vertexStride           = sizeof(glm::vec3);
    spheres.radiusFormat           = VK_FORMAT_R32_SFLOAT;  // float radius
    spheres.radiusData.deviceAddress = m_splatModel.sphereRadiusBuffer.address;
    spheres.radiusStride           = sizeof(float);
    spheres.indexType              = VK_INDEX_TYPE_NONE_KHR;  // No indexing, direct access
    spheres.indexData.deviceAddress = 0;
    spheres.indexStride            = 0;

    // Geometry wrapping the sphere data via pNext chain
    // Note: VK_GEOMETRY_TYPE_AABBS_KHR is used as base, spheres data in pNext
    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.pNext        = &spheres;  // Sphere data attached via pNext
    geometry.geometryType = VK_GEOMETRY_TYPE_SPHERES_NV;
    geometry.flags        = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.firstVertex     = 0;
    rangeInfo.primitiveCount  = m_splatModel.nbSpheres;
    rangeInfo.primitiveOffset = 0;
    rangeInfo.transformOffset = 0;

    LOGI("Using Blackwell native sphere primitives: %u spheres\n", m_splatModel.nbSpheres);
    return nvvk::AccelerationStructureGeometryInfo{.geometry = geometry, .rangeInfo = rangeInfo};
  }
  // use icosa mesh
  else if(!m_rtxUseAABBs)
  {
    // Describe buffer as array of glm::vec3 wint uint32 indices.
    VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
    triangles.sType                    = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    triangles.vertexFormat             = VK_FORMAT_R32G32B32_SFLOAT;  // vec3 vertex position data.
    triangles.vertexData.deviceAddress = m_splatModel.vertexBuffer.address;
    triangles.vertexStride             = sizeof(glm::vec3);
    triangles.indexType                = VK_INDEX_TYPE_UINT32;
    triangles.indexData.deviceAddress  = m_splatModel.indexBuffer.address;
    triangles.transformData            = {};  // identity
    triangles.maxVertex                = m_splatModel.nbVertices - 1;

    // Identify the data as containing opaque triangles.
    VkAccelerationStructureGeometryKHR geometry{};  // do not forget default init {} so pNext is NULL
    geometry.sType              = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType       = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geometry.flags              = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;
    geometry.geometry.triangles = triangles;

    // The entire array will be used to build the BLAS.
    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.firstVertex     = 0;
    rangeInfo.primitiveCount  = m_splatModel.nbIndices / 3;
    rangeInfo.primitiveOffset = 0;
    rangeInfo.transformOffset = 0;

    return nvvk::AccelerationStructureGeometryInfo{.geometry = geometry, .rangeInfo = rangeInfo};
  }
  // Use AABBs
  else
  {
    VkAccelerationStructureGeometryAabbsDataKHR aabbs{};
    aabbs.sType              = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
    aabbs.data.deviceAddress = m_splatModel.aabbBuffer.address;
    aabbs.stride             = sizeof(SplatAabb);

    // Identify the data as containing opaque AABBs.
    VkAccelerationStructureGeometryKHR geometry{};  // do not forget default init {} so pNext is NULL
    geometry.sType          = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType   = VK_GEOMETRY_TYPE_AABBS_KHR;
    geometry.flags          = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;
    geometry.geometry.aabbs = aabbs;

    // The entire array will be used to build the BLAS.
    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.firstVertex     = 0;
    rangeInfo.primitiveCount  = m_splatModel.nbAABB;
    rangeInfo.primitiveOffset = 0;
    rangeInfo.transformOffset = 0;

    return nvvk::AccelerationStructureGeometryInfo{.geometry = geometry, .rangeInfo = rangeInfo};
  }
}

void SplatSetVk::rtxInitAccelerationStructures(SplatSet& splatSet)
{
  SCOPED_TIMER(std::string(__FUNCTION__) + "\n");

  // GS BLAS - Storing the icosahedron or AABB in a geometry
  {
    std::vector<nvvk::AccelerationStructureGeometryInfo> asGeoInfo{rtxCreateSplatModelAccelerationStructureGeometryInfo()};

    if(m_rtxCompressBlas)
      rtAccelerationStructures.blasSubmitBuildAndWait(asGeoInfo, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                                                                     | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR
                                                                     | VK_BUILD_ACCELERATION_STRUCTURE_LOW_MEMORY_BIT_KHR);
    else
      rtAccelerationStructures.blasSubmitBuildAndWait(asGeoInfo, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);

    // Statistics
    LOGI("%s%s\n", nvutils::ScopedTimer::indent().c_str(), rtAccelerationStructures.blasBuildStatistics.toString().c_str());
  }

  // GS TLAS
  {
    const auto instCount = m_rtxUseInstances ? splatSet.size() : 1;

    std::vector<VkAccelerationStructureInstanceKHR> tlasInstances(
        instCount, {
                       // We do not use gl_InstanceCustomIndexEXT
                       .instanceCustomIndex = 0,
                       //  Only be hit if rayMask & instance.mask != 0
                       .mask = 0xFF,
                       // We will use the any hit hit group for all objects)
                       .instanceShaderBindingTableRecordOffset = 0,
                       // 0 = backface culling on
                       // VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR,
                       // VK_GEOMETRY_INSTANCE_TRIANGLE_FRONT_COUNTERCLOCKWISE_BIT_KHR
                       .flags = VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR,
                       // use the only BLAS
                       .accelerationStructureReference = rtAccelerationStructures.blasSet[0].address,
                   });

    // estimate the memory usage and early return if max than authorized limit
    VkDeviceSize sizeBytes = std::span<VkAccelerationStructureInstanceKHR const>(tlasInstances).size_bytes();

    // Check hardware limits
    if(m_rtxUseInstances && instCount > m_accelStructProps.maxInstanceCount)
    {
      LOGW("Splat count (%zu) exceeds maxInstanceCount (%llu). Disabling RTX.\n", instCount,
           (unsigned long long)m_accelStructProps.maxInstanceCount);
      rtxValid = false;
      rtxDeinitAccelerationStructures();
      return;
    }

    if(!m_rtxUseInstances)
    {
      uint64_t primitiveCount = m_rtxUseAABBs ? m_splatModel.nbAABB : (m_splatModel.nbIndices / 3);
      if(primitiveCount > m_accelStructProps.maxPrimitiveCount)
      {
        LOGW("Primitive count (%" PRIu64 ") exceeds maxPrimitiveCount (%" PRIu64 "). Disabling RTX.\n", primitiveCount,
              m_accelStructProps.maxPrimitiveCount);
        rtxValid = false;
        rtxDeinitAccelerationStructures();
        return;
      }
    }

    // TODO: Should query the device
    if(sizeBytes >= m_deviceInfo->properties11.maxMemoryAllocationSize)
    {
      LOGW("Model too large to generate RTX acceleration structure, Raytracing will be deactivated\n");
      rtxValid = false;
      rtxDeinitAccelerationStructures();
      return;
    }

    if(m_rtxUseInstances)
    {  // one instance per splat
      // for(uint32_t splatIdx = 0; splatIdx < instCount; ++splatIdx)
      START_PAR_LOOP(instCount, splatIdx)
      {
        const glm::mat4 transform = rtxComputeTransformMatrix(splatSet, splatIdx);

        // set the transformation matrix
        tlasInstances[splatIdx].transform = nvvk::toTransformMatrixKHR(transform);
      }
      END_PAR_LOOP()
    }
    else
    {  // first instance contains all splats
      tlasInstances[0].transform = nvvk::toTransformMatrixKHR(glm::mat4(1.0f));
    }

    // then build the static TLAS
    rtAccelerationStructures.tlasSubmitBuildAndWait(tlasInstances, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
  }

  // Update memory statistics
  tlasSizeBytes = rtAccelerationStructures.tlasBuildData.sizeInfo.accelerationStructureSize;
  blasSizeBytes = 0;
  for(auto& bd : rtAccelerationStructures.blasBuildData)
  {
    blasSizeBytes += bd.sizeInfo.accelerationStructureSize;
  }

  //
  rtxValid = true;
}

////////////////////////////////////////////////////////////////////////////////
// PTLAS (VK_NV_partitioned_acceleration_structure) support
// Enables GPU-driven partial TLAS updates for sparse FreeTimeGS animations
////////////////////////////////////////////////////////////////////////////////

void SplatSetVk::rtxInitPtlasPartitions(SplatSet& splatSet, float cellSize)
{
  const uint32_t splatCount = static_cast<uint32_t>(splatSet.size());
  if(splatCount == 0)
    return;

  auto startTime = std::chrono::high_resolution_clock::now();

  // Store cell size
  ptlas.cellSize = cellSize;

  // Compute scene AABB
  ptlas.sceneMin = glm::vec3(std::numeric_limits<float>::max());
  ptlas.sceneMax = glm::vec3(std::numeric_limits<float>::lowest());

  for(uint32_t i = 0; i < splatCount; ++i)
  {
    glm::vec3 pos(splatSet.positions[i * 3 + 0], splatSet.positions[i * 3 + 1], splatSet.positions[i * 3 + 2]);
    ptlas.sceneMin = glm::min(ptlas.sceneMin, pos);
    ptlas.sceneMax = glm::max(ptlas.sceneMax, pos);
  }

  // Add small padding to avoid edge cases
  ptlas.sceneMin -= glm::vec3(0.001f);
  ptlas.sceneMax += glm::vec3(0.001f);

  // Compute grid dimensions
  glm::vec3 sceneSize = ptlas.sceneMax - ptlas.sceneMin;
  ptlas.gridDims = glm::max(glm::ivec3(1), glm::ivec3(glm::ceil(sceneSize / cellSize)));
  ptlas.numPartitions = ptlas.gridDims.x * ptlas.gridDims.y * ptlas.gridDims.z;

  // Allocate per-splat partition IDs
  ptlas.splatPartitionIds.resize(splatCount);

  // Allocate partition splat lists
  ptlas.partitionSplatLists.resize(ptlas.numPartitions);
  for(auto& list : ptlas.partitionSplatLists)
    list.clear();

  // Assign splats to partitions
  for(uint32_t i = 0; i < splatCount; ++i)
  {
    glm::vec3 pos(splatSet.positions[i * 3 + 0], splatSet.positions[i * 3 + 1], splatSet.positions[i * 3 + 2]);
    uint32_t partitionIdx = ptlas.getPartitionIndex(pos);
    ptlas.splatPartitionIds[i] = partitionIdx;
    ptlas.partitionSplatLists[partitionIdx].push_back(i);
  }

  // Initialize dirty tracking
  ptlas.dirtyPartitions.resize(ptlas.numPartitions, false);

  auto endTime = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

  // Log partition statistics
  uint32_t maxSplatsPerPartition = 0;
  uint32_t emptyPartitions = 0;
  for(const auto& list : ptlas.partitionSplatLists)
  {
    maxSplatsPerPartition = std::max(maxSplatsPerPartition, static_cast<uint32_t>(list.size()));
    if(list.empty())
      emptyPartitions++;
  }

  LOGI("PTLAS partitioning: %u splats -> %u partitions (%dx%dx%d grid, cell=%.2f)\n", splatCount, ptlas.numPartitions,
       ptlas.gridDims.x, ptlas.gridDims.y, ptlas.gridDims.z, cellSize);
  LOGI("  Max splats/partition: %u, Empty partitions: %u/%u\n", maxSplatsPerPartition, emptyPartitions, ptlas.numPartitions);
  LOGI("  Partitioning completed in %lld ms\n", static_cast<long long>(duration.count()));
}

void SplatSetVk::rtxBuildPtlas(SplatSet& splatSet)
{
  // PTLAS requires the extension to be available
  if(!ptlas.extensionSupported)
  {
    LOGW("PTLAS: Extension VK_NV_partitioned_acceleration_structure not supported\n");
    return;
  }

  if(ptlas.numPartitions == 0)
  {
    LOGW("PTLAS: No partitions initialized. Call rtxInitPtlasPartitions first.\n");
    return;
  }

  // For now, log what would happen and note this is a stub for when we have Blackwell hardware
  uint32_t dirtyCount = ptlas.countDirty();

  if(dirtyCount == 0)
  {
    LOGD("PTLAS: No dirty partitions, skipping update\n");
    return;
  }

  LOGI("PTLAS: Would update %u/%u dirty partitions (%.1f%% of TLAS)\n", dirtyCount, ptlas.numPartitions,
       100.0f * dirtyCount / ptlas.numPartitions);

  // TODO: Implement actual PTLAS building when running on Blackwell hardware
  // The implementation would:
  // 1. Upload VkPartitionedAccelerationStructureWriteInstanceDataNV for dirty instances
  // 2. Build VkBuildPartitionedAccelerationStructureIndirectCommandNV with WRITE_INSTANCE op
  // 3. Call vkCmdBuildPartitionedAccelerationStructuresNV
  //
  // For now, we fall back to regular TLAS rebuild which the caller handles

  ptlas.clearDirty();
}

void SplatSetVk::rtxDeinitPtlas()
{
  // Clean up PTLAS device resources
  if(ptlas.ptlas != VK_NULL_HANDLE)
  {
    vkDestroyAccelerationStructureKHR(m_app->getDevice(), ptlas.ptlas, nullptr);
    ptlas.ptlas = VK_NULL_HANDLE;
  }

  if(ptlas.ptlasBuffer.buffer != VK_NULL_HANDLE)
    m_alloc->destroyBuffer(ptlas.ptlasBuffer);
  if(ptlas.ptlasInstancesBuffer.buffer != VK_NULL_HANDLE)
    m_alloc->destroyBuffer(ptlas.ptlasInstancesBuffer);
  if(ptlas.ptlasBuildOpsBuffer.buffer != VK_NULL_HANDLE)
    m_alloc->destroyBuffer(ptlas.ptlasBuildOpsBuffer);
  if(ptlas.ptlasOpCountBuffer.buffer != VK_NULL_HANDLE)
    m_alloc->destroyBuffer(ptlas.ptlasOpCountBuffer);
  if(ptlas.ptlasScratchBuffer.buffer != VK_NULL_HANDLE)
    m_alloc->destroyBuffer(ptlas.ptlasScratchBuffer);

  // Reset all state
  ptlas = PtlasData{};
}

}  // namespace vk_viewer
