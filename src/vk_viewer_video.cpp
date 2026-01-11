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
#ifdef WITH_VIDEO_DECODER
  if (m_videoDepthManager && m_videoDepthManager->isPlaying())
  {
    static int updateLogCounter = 0;
    if(updateLogCounter++ % 120 == 0) {
      LOGD("updateDepthRendering: videoDepthManager active, paused=%d\n", m_playbackPaused ? 1 : 0);
    }
    
    if(m_playbackPaused) return;

    VideoDecoder* videoDecoder = m_videoDepthManager->getVideoDecoder();
    DepthVideoLoader* depthLoader = m_videoDepthManager->getDepthLoader();

    if(videoDecoder && depthLoader)
    {
        DecodedFrame videoFrame;
        bool gotFrame = videoDecoder->getNextFrame(videoFrame);
        if(updateLogCounter % 120 == 1) {
          LOGD("updateDepthRendering: getNextFrame=%d, frame size=%dx%d\n", 
               gotFrame ? 1 : 0, videoFrame.width, videoFrame.height);
        }
        if(gotFrame)
        {
            if (videoFrame.width > 0 && videoFrame.height > 0)
            {
                bool updateDescriptor = false;
                if(m_videoTexture.width != videoFrame.width || m_videoTexture.height != videoFrame.height || m_videoTexture.image.image == VK_NULL_HANDLE)
                {
                    vkDeviceWaitIdle(m_device);
                    if(m_videoTexture.view != VK_NULL_HANDLE) { vkDestroyImageView(m_device, m_videoTexture.view, nullptr); m_videoTexture.view = VK_NULL_HANDLE; }
                    if(m_videoTexture.image.image != VK_NULL_HANDLE) { m_alloc.destroyImage(m_videoTexture.image); m_videoTexture.image = {}; }

                    VkImageCreateInfo info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
                    info.imageType = VK_IMAGE_TYPE_2D;
                    info.format = VK_FORMAT_R8G8B8A8_UNORM;
                    info.extent = {static_cast<uint32_t>(videoFrame.width), static_cast<uint32_t>(videoFrame.height), 1};
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
                    
                    m_videoTexture.width = videoFrame.width;
                    m_videoTexture.height = videoFrame.height;
                    updateDescriptor = true;
                }

                // Barrier: Undefined/ShaderRead -> TransferDst
                VkImageMemoryBarrier barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                barrier.srcAccessMask = 0;
                barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                barrier.image = m_videoTexture.image.image;
                barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

                VkDeviceSize imageSize = videoFrame.data.size();
                m_uploader.appendImage(m_videoTexture.image, imageSize, videoFrame.data.data());
                m_uploader.cmdUploadAppended(cmd);

                // Barrier: TransferDst -> ShaderRead
                barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

                if(updateDescriptor && m_descriptorSet != VK_NULL_HANDLE)
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

            uint32_t videoTimestampMs = static_cast<uint32_t>(videoFrame.timestamp * 1000.0);
            DepthVideoFrame depthFrame;
            if(depthLoader->getFrameByTimestamp(videoTimestampMs, depthFrame))
            {
                if(m_depthManager && m_lastVdzFrameIndex != static_cast<size_t>(depthFrame.timestampMs))
                {
                    DepthFrame uploadFrame;
                    uploadFrame.timestampMs = depthFrame.timestampMs;
                    uploadFrame.width = depthFrame.width;
                    uploadFrame.height = depthFrame.height;
                    uploadFrame.data = std::move(depthFrame.data);
                    uploadFrame.scale = 1.0f;
                    uploadFrame.bias = 0.0f;
                    uploadFrame.zMax = depthFrame.zMax;

                    m_depthManager->uploadDepthFrame(uploadFrame, cmd);
                    m_lastVdzFrameIndex = depthFrame.timestampMs;
                    m_depthFrameCounter++;

                    // Log frame info periodically
                    static int frameLogCounter = 0;
                    if (frameLogCounter++ % 30 == 0) {
                        glm::vec3 eye, center, up;
                        cameraManip->getLookat(eye, center, up);
                        float distToCenter = glm::length(center - eye);
                        LOGI("=== FRAME %u ===\n", depthFrame.timestampMs);
                        LOGI("Camera: eye=(%.2f, %.2f, %.2f) dist=%.2f\n", eye.x, eye.y, eye.z, distToCenter);
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
        }
    }
  }
  else
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
