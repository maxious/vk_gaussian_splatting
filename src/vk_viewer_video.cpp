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

#include "vk_viewer.h"
#include <nvutils/logger.hpp>

namespace vk_viewer {

void VkViewer::updateDepthRendering(VkCommandBuffer cmd)
{
#ifdef WITH_TCP_DEPTH
  if(m_tcpDepthSingleImageRequested)
  {
    if(!m_tcpDepthSingleImageDone && m_tcpServerManager)
    {
      m_tcpServerManager->update();

      DepthFrame frame;
      if(m_depthBuffer.getFrame(0, frame) && m_depthManager)
      {
        m_depthManager->uploadDepthFrame(frame, cmd);
        ++m_depthFrameCounter;
        ++m_depthFrameCount;

        if(m_descriptorSet != VK_NULL_HANDLE)
        {
          const auto& depthTexture = m_depthManager->getCurrentTexture();
          if(depthTexture.image.descriptor.imageView)
          {
            VkDescriptorImageInfo depthImageInfo{};
            depthImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            depthImageInfo.imageView = depthTexture.image.descriptor.imageView;
            depthImageInfo.sampler = m_sampler;

            VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            write.dstSet = m_descriptorSet;
            write.dstBinding = BINDING_VDZ_DEPTH_TEXTURE;
            write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write.descriptorCount = 1;
            write.pImageInfo = &depthImageInfo;
            vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);
          }
        }

        m_tcpDepthSingleImageDone = true;
        LOGI("Single image depth received: %dx%d\n", frame.width, frame.height);
      }
    }
    return;
  }

  static auto lastFpsTime = std::chrono::steady_clock::now();
  auto now = std::chrono::steady_clock::now();
  float elapsed_sec = std::chrono::duration<float>(now - lastFpsTime).count();
  if (elapsed_sec >= 1.0f) {
      m_depthFps = m_depthFrameCount / elapsed_sec;
      m_depthFrameCount = 0;
      lastFpsTime = now;
  }
#endif

#ifdef WITH_TCP_DEPTH
  if(m_tcpDepthEnabled && m_tcpServerManager)
  {
    if(m_playbackPaused)
    {
      return;
    }

    m_tcpServerManager->update();

    const uint64_t currentTimeMs = static_cast<uint64_t>(prmFrame.currentTime * 1000.0f);
    DepthFrame frame;
    if(m_depthBuffer.getFrame(currentTimeMs, frame) && m_depthManager)
    {
      m_depthManager->uploadDepthFrame(frame, cmd);
      ++m_depthFrameCounter;
#ifdef WITH_TCP_DEPTH
      ++m_depthFrameCount;
#endif

      if(m_descriptorSet != VK_NULL_HANDLE)
      {
        const auto& depthTexture = m_depthManager->getCurrentTexture();
        if(depthTexture.image.descriptor.imageView)
        {
          VkDescriptorImageInfo depthImageInfo{};
          depthImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
          depthImageInfo.imageView = depthTexture.image.descriptor.imageView;
          depthImageInfo.sampler = m_sampler;

          VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
          write.dstSet = m_descriptorSet;
          write.dstBinding = BINDING_VDZ_DEPTH_TEXTURE;
          write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
          write.descriptorCount = 1;
          write.pImageInfo = &depthImageInfo;
          vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);
        }
      }
    }
    return;
  }
#endif

#ifdef WITH_VIDEO_DECODER
  if (m_videoDepthManager && m_videoDepthManager->isPlaying())
  {
    if(m_playbackPaused) 
        return;

    double t = m_videoDepthManager->getCurrentTime();
    PlaybackFrame pf;
    if(!m_videoDepthManager->getFrameAtTime(t, pf))
        return;

    // Handle RGB Frame (HW or SW)
    if (pf.width > 0 && pf.height > 0)
    {
        bool updateDescriptor = false;
        bool isHWFrame = (pf.rgbImage != VK_NULL_HANDLE);

        if (isHWFrame) 
        {
            // Hardware Decoded Frame (Zero-Copy)
            // Just update the descriptor if the image handle changed
            if (m_videoTexture.image.image != pf.rgbImage)
            {
                // We don't own this image, so we don't destroy it.
                // But we need to ensure m_videoTexture doesn't think it owns a previous SW image.
                if (m_videoTexture.image.image != VK_NULL_HANDLE && m_videoTexture.image.allocation != nullptr) {
                    m_alloc.destroyImage(m_videoTexture.image);
                }
                
                if (m_videoTexture.view != VK_NULL_HANDLE) { 
                    vkDestroyImageView(m_device, m_videoTexture.view, nullptr); 
                    m_videoTexture.view = VK_NULL_HANDLE; 
                }

                // Update internal tracking
                m_videoTexture.image.image = pf.rgbImage;
                m_videoTexture.image.allocation = nullptr; // Imported/External
                m_videoTexture.width = pf.width;
                m_videoTexture.height = pf.height;

                // Create View
                VkImageViewCreateInfo viewInfo = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
                viewInfo.image = pf.rgbImage;
                viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
                // Format comes from FFmpeg (likely VK_FORMAT_G8_B8R8_2PLANE_420_UNORM or similar for NV12)
                // If it's undefined, default to R8G8B8A8_UNORM (unlikely to work for HW, but fallback)
                viewInfo.format = (pf.rgbFormat != VK_FORMAT_UNDEFINED) ? pf.rgbFormat : VK_FORMAT_R8G8B8A8_UNORM;
                
                // Mappings for YCbCr if needed. For now assume identity or handled by sampler conversion
                viewInfo.components = {
                    VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                    VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY
                };

                viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                
                vkCreateImageView(m_device, &viewInfo, nullptr, &m_videoTexture.view);
                updateDescriptor = true;
                
                // Transition layout if needed
                VkImageMemoryBarrier barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                barrier.oldLayout = pf.rgbLayout; 
                barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                barrier.srcAccessMask = 0; // Handled by semaphore ideally, or assume decode done
                barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                barrier.image = pf.rgbImage;
                barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                
                // If semaphore provided, we should wait on it? 
                // Currently single queue, so execution barrier might suffice if on same queue.
                if (pf.rgbLayout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                {
                    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 
                                         0, 0, nullptr, 0, nullptr, 1, &barrier);
                }
            }
        }
        else if (!pf.rgbRGBA.empty())
        {
            // Software Decoded Frame (Upload)
            if(m_videoTexture.width != pf.width || m_videoTexture.height != pf.height || m_videoTexture.image.image == VK_NULL_HANDLE || m_videoTexture.image.allocation == nullptr)
            {
                vkDeviceWaitIdle(m_device);
                if(m_videoTexture.view != VK_NULL_HANDLE) { vkDestroyImageView(m_device, m_videoTexture.view, nullptr); m_videoTexture.view = VK_NULL_HANDLE; }
                if(m_videoTexture.image.image != VK_NULL_HANDLE) { 
                    // If we are switching from HW to SW, we might have an image handle but no allocation.
                    // If allocation is null, we assume it's external and don't destroy it.
                    // But if it IS internal, we must destroy it.
                    if (m_videoTexture.image.allocation != nullptr) {
                        m_alloc.destroyImage(m_videoTexture.image); 
                    }
                    m_videoTexture.image = {}; 
                }

                VkImageCreateInfo info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
                info.imageType = VK_IMAGE_TYPE_2D;
                info.format = VK_FORMAT_R8G8B8A8_UNORM;
                info.extent = {pf.width, pf.height, 1};
                info.mipLevels = 1;
                info.arrayLayers = 1;
                info.samples = VK_SAMPLE_COUNT_1_BIT;
                info.tiling = VK_IMAGE_TILING_OPTIMAL;
                info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
                info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                
                m_alloc.createImage(m_videoTexture.image, info);

                VkImageViewCreateInfo viewInfo = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
                viewInfo.image = m_videoTexture.image.image;
                viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
                viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
                viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                vkCreateImageView(m_device, &viewInfo, nullptr, &m_videoTexture.view);
                
                m_videoTexture.width = pf.width;
                m_videoTexture.height = pf.height;
                updateDescriptor = true;
            }

            VkImageMemoryBarrier barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.image = m_videoTexture.image.image;
            barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

            VkDeviceSize imageSize = pf.rgbRGBA.size();
            m_uploader.appendImage(m_videoTexture.image, imageSize, pf.rgbRGBA.data());
            m_uploader.cmdUploadAppended(cmd);

            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
        }

        if(updateDescriptor && m_descriptorSet != VK_NULL_HANDLE)
        {
            VkDescriptorImageInfo imageInfo{};
            imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo.imageView = m_videoTexture.view;
            imageInfo.sampler = m_sampler; // Note: YCbCr might need immutable sampler with conversion

            VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            write.dstSet = m_descriptorSet;
            write.dstBinding = BINDING_VDZ_VIDEO_TEXTURE;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write.descriptorCount = 1;
            write.pImageInfo = &imageInfo;
            vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);
        }
    }

    if(m_depthManager && !pf.depthMeters.empty() && m_lastVdzFrameIndex != static_cast<size_t>(pf.frameIndex))
    {
        DepthFrame uploadFrame;
        uploadFrame.timestampMs = static_cast<uint32_t>(pf.timestampSec * 1000.0);
        uploadFrame.width = pf.width;
        uploadFrame.height = pf.height;
        uploadFrame.data = pf.depthMeters;
        uploadFrame.scale = 1.0f;
        uploadFrame.bias = 0.0f;
        uploadFrame.zMax = pf.zMax;

        m_depthManager->uploadDepthFrame(uploadFrame, cmd);
        m_lastVdzFrameIndex = pf.frameIndex;
        m_depthFrameCounter++;
#ifdef WITH_TCP_DEPTH
        m_depthFrameCount++;
#endif
        
        if(pf.zMax > pf.zMin && pf.zMax > 0.0f)
        {
            prmFrame.vdzZMin = pf.zMin;
            prmFrame.vdzZMax = pf.zMax;
        }
        else
        {
            static bool warnedOnce = false;
            if(!warnedOnce)
            {
                LOGW("Depth metadata missing z_min/z_max, using defaults [0, 1]\n");
                warnedOnce = true;
            }
            prmFrame.vdzZMin = 0.0f;
            prmFrame.vdzZMax = 1.0f;
        }
        
        if(m_descriptorSet != VK_NULL_HANDLE)
        {
            const auto& depthTexture = m_depthManager->getCurrentTexture();
            if(depthTexture.image.descriptor.imageView)
            {
                VkDescriptorImageInfo depthImageInfo{};
                depthImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                depthImageInfo.imageView = depthTexture.image.descriptor.imageView;
                depthImageInfo.sampler = m_sampler;

                VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                write.dstSet = m_descriptorSet;
                write.dstBinding = BINDING_VDZ_DEPTH_TEXTURE;
                write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                write.descriptorCount = 1;
                write.pImageInfo = &depthImageInfo;
                vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);
            }
        }
    }
  }

  // HLS playback mode
  if (m_hlsPlayer && m_hlsPlaybackMode)
  {
    if (m_playbackPaused)
      return;

    // Get current time from playback
    auto now = std::chrono::steady_clock::now();
    double elapsedMs = std::chrono::duration<double, std::milli>(now - m_playbackStartTime).count();
    double currentTimeSec = elapsedMs / 1000.0 + m_playbackTimeOffset / 1000.0;

    // Seek if needed
    if (m_lastVdzFrameIndex == SIZE_MAX)
    {
      m_hlsPlayer->seekToTime(currentTimeSec);
    }

    // Get next frame
    HlsDecodedFrame frame;
    if (!m_hlsPlayer->getNextFrame(frame))
    {
      // End of stream - loop or stop
      if (currentTimeSec >= m_hlsMetadata.duration)
      {
        m_playbackStartTime = now;
        m_playbackTimeOffset = 0.0;
        m_hlsPlayer->seekToTime(0.0);
      }
      return;
    }

    if (frame.width > 0 && frame.height > 0 && !frame.rgbData.empty())
    {
      bool updateDescriptor = false;
      if (m_videoTexture.width != frame.width || m_videoTexture.height != frame.height || m_videoTexture.image.image == VK_NULL_HANDLE)
      {
        vkDeviceWaitIdle(m_device);
        if (m_videoTexture.view != VK_NULL_HANDLE)
        {
          vkDestroyImageView(m_device, m_videoTexture.view, nullptr);
          m_videoTexture.view = VK_NULL_HANDLE;
        }
        if (m_videoTexture.image.image != VK_NULL_HANDLE)
        {
          m_alloc.destroyImage(m_videoTexture.image);
          m_videoTexture.image = {};
        }

        VkImageCreateInfo info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        info.imageType = VK_IMAGE_TYPE_2D;
        info.format = VK_FORMAT_R8G8B8A8_UNORM;
        info.extent = {static_cast<uint32_t>(frame.width), static_cast<uint32_t>(frame.height), 1};
        info.mipLevels = 1;
        info.arrayLayers = 1;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        m_alloc.createImage(m_videoTexture.image, info);

        VkImageViewCreateInfo viewInfo = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        viewInfo.image = m_videoTexture.image.image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
        viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCreateImageView(m_device, &viewInfo, nullptr, &m_videoTexture.view);

        m_videoTexture.width = frame.width;
        m_videoTexture.height = frame.height;
        updateDescriptor = true;
      }

      VkImageMemoryBarrier barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
      barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      barrier.srcAccessMask = 0;
      barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      barrier.image = m_videoTexture.image.image;
      barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

      VkDeviceSize imageSize = frame.rgbData.size();
      m_uploader.appendImage(m_videoTexture.image, imageSize, frame.rgbData.data());
      m_uploader.cmdUploadAppended(cmd);

      barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

      if (updateDescriptor && m_descriptorSet != VK_NULL_HANDLE)
      {
        VkDescriptorImageInfo imageInfo{};
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo.imageView = m_videoTexture.view;
        imageInfo.sampler = m_sampler;

        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = m_descriptorSet;
        write.dstBinding = BINDING_VDZ_VIDEO_TEXTURE;
        write.dstArrayElement = 0;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.descriptorCount = 1;
        write.pImageInfo = &imageInfo;
        vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);
      }
    }

    // Upload depth data if available and frame changed
    if (m_depthManager && !frame.depthData.empty())
    {
      size_t frameHash = std::hash<std::string>()(
        std::string(reinterpret_cast<char*>(frame.depthData.data()), frame.depthData.size() * sizeof(float)));

      if (m_lastVdzFrameIndex != frameHash)
      {
        DepthFrame uploadFrame;
        uploadFrame.timestampMs = static_cast<uint32_t>(frame.timestamp * 1000.0);
        uploadFrame.width = frame.width;
        uploadFrame.height = frame.height;
        uploadFrame.data = frame.depthData;
        uploadFrame.scale = m_hlsMetadata.scale;
        uploadFrame.bias = m_hlsMetadata.zMin;
        uploadFrame.zMax = m_hlsMetadata.zMax;

        m_depthManager->uploadDepthFrame(uploadFrame, cmd);
        m_lastVdzFrameIndex = frameHash;
        m_depthFrameCounter++;
#ifdef WITH_TCP_DEPTH
        m_depthFrameCount++;
#endif

        prmFrame.vdzZMin = m_hlsMetadata.zMin;
        prmFrame.vdzZMax = m_hlsMetadata.zMax;

        if (m_descriptorSet != VK_NULL_HANDLE)
        {
          const auto& depthTexture = m_depthManager->getCurrentTexture();
          if (depthTexture.image.descriptor.imageView)
          {
            VkDescriptorImageInfo depthImageInfo{};
            depthImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            depthImageInfo.imageView = depthTexture.image.descriptor.imageView;
            depthImageInfo.sampler = m_sampler;

            VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            write.dstSet = m_descriptorSet;
            write.dstBinding = BINDING_VDZ_DEPTH_TEXTURE;
            write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write.descriptorCount = 1;
            write.pImageInfo = &depthImageInfo;
            vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);
           }
        }
      }
    }
  }
#endif
  {
    // Handle streaming case
    if(!m_depthClient)
    {
      return;
    }

    uint64_t timestampMs = static_cast<uint64_t>(prmFrame.currentTime * 1000.0f);
    m_depthClient->update(prmFrame.currentTime * 1000.0f, m_depthClient->getFps());

    if(m_playbackPaused)
    {
      m_depthStats = m_depthClient->getStats();
      return;
    }

    DepthFrame frame;
    if(m_depthClient->getFrame(timestampMs, frame)) {
        if(m_depthManager) {
             m_depthManager->uploadDepthFrame(frame, cmd);
#ifdef WITH_TCP_DEPTH
             m_depthFrameCount++;
#endif
             
             if(m_descriptorSet != VK_NULL_HANDLE) {
               const auto& depthTexture = m_depthManager->getCurrentTexture();
               if(depthTexture.image.descriptor.imageView) {
                 VkDescriptorImageInfo depthImageInfo{};
                 depthImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                 depthImageInfo.imageView = depthTexture.image.descriptor.imageView;
                 depthImageInfo.sampler = m_sampler;
                 VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                 write.dstSet = m_descriptorSet;
                 write.dstBinding = BINDING_VDZ_DEPTH_TEXTURE;
                 write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                 write.descriptorCount = 1;
                 write.pImageInfo = &depthImageInfo;
                 vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);
               }
             }
        }
    }
    m_depthStats = m_depthClient->getStats();
  }
}

} // namespace vk_viewer
