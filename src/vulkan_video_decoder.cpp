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

#include "vulkan_video_decoder.h"
#include <nvutils/logger.hpp>
#include <cstring>
#include <algorithm>



// FFmpeg includes for bitstream parsing
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
}

namespace vk_viewer {

// Start code for H.265 NAL units: 0x00 0x00 0x01
static const uint8_t START_CODE[] = {0x00, 0x00, 0x01};
static const size_t START_CODE_SIZE = sizeof(START_CODE);

// Maximum bitstream buffer size (16 MB should be enough for most frames)
static constexpr VkDeviceSize MAX_BITSTREAM_SIZE = 16 * 1024 * 1024;

// Number of output frames to allocate (for pipelining)
static constexpr uint32_t NUM_OUTPUT_FRAMES = 4;

std::vector<const char*> getVulkanVideoExtensions()
{
    return {
        VK_KHR_VIDEO_QUEUE_EXTENSION_NAME,
        VK_KHR_VIDEO_DECODE_QUEUE_EXTENSION_NAME,
        VK_KHR_VIDEO_DECODE_H265_EXTENSION_NAME,
        VK_KHR_VIDEO_DECODE_H264_EXTENSION_NAME,  // Optional, for future use
    };
}

bool checkVulkanVideoExtensionSupport(VkPhysicalDevice physicalDevice)
{
    uint32_t extensionCount = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, nullptr);
    
    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, availableExtensions.data());
    
    // Check for required extensions
    std::vector<const char*> requiredExtensions = {
        VK_KHR_VIDEO_QUEUE_EXTENSION_NAME,
        VK_KHR_VIDEO_DECODE_QUEUE_EXTENSION_NAME,
        VK_KHR_VIDEO_DECODE_H265_EXTENSION_NAME,
    };
    
    for (const char* required : requiredExtensions) {
        bool found = false;
        for (const auto& available : availableExtensions) {
            if (strcmp(required, available.extensionName) == 0) {
                found = true;
                break;
            }
        }
        if (!found) {
            LOGD("Vulkan Video extension not available: %s\n", required);
            return false;
        }
    }
    
    return true;
}

VulkanVideoDecoder::VulkanVideoDecoder() = default;

VulkanVideoDecoder::~VulkanVideoDecoder()
{
    close();
    destroyResources();
}

bool VulkanVideoDecoder::queryCapabilities(VkPhysicalDevice physicalDevice, VulkanVideoCapabilities& caps)
{
    caps = {};
    
    // Check extension support first
    if (!checkVulkanVideoExtensionSupport(physicalDevice)) {
        LOGI("Vulkan Video extensions not supported on this device\n");
        return false;
    }
    
    // Query queue families for video decode support
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
    
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());
    
    for (uint32_t i = 0; i < queueFamilyCount; ++i) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_VIDEO_DECODE_BIT_KHR) {
            caps.videoQueueFamilyIndex = i;
            break;
        }
    }
    
    if (caps.videoQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED) {
        LOGI("No video decode queue family found\n");
        return false;
    }
    
    // Query H.265 decode capabilities
    VkVideoDecodeH265ProfileInfoKHR h265Profile = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_PROFILE_INFO_KHR,
        .stdProfileIdc = STD_VIDEO_H265_PROFILE_IDC_MAIN,
    };
    
    VkVideoProfileInfoKHR videoProfile = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_PROFILE_INFO_KHR,
        .pNext = &h265Profile,
        .videoCodecOperation = VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR,
        .chromaSubsampling = VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR,
        .lumaBitDepth = VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR,
        .chromaBitDepth = VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR,
    };
    
    VkVideoDecodeCapabilitiesKHR decodeCapabilities = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_CAPABILITIES_KHR,
    };
    
    VkVideoCapabilitiesKHR videoCapabilities = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_CAPABILITIES_KHR,
        .pNext = &decodeCapabilities,
    };
    
    VkResult result = vkGetPhysicalDeviceVideoCapabilitiesKHR(physicalDevice, &videoProfile, &videoCapabilities);
    if (result != VK_SUCCESS) {
        LOGI("Failed to query H.265 decode capabilities: %d\n", result);
        return false;
    }
    
    caps.supported = true;
    caps.h265DecodeSupported = true;
    
    // DPB configuration
    caps.dpbAndOutputCoincide = (decodeCapabilities.flags & VK_VIDEO_DECODE_CAPABILITY_DPB_AND_OUTPUT_COINCIDE_BIT_KHR) != 0;
    caps.dpbAndOutputDistinct = (decodeCapabilities.flags & VK_VIDEO_DECODE_CAPABILITY_DPB_AND_OUTPUT_DISTINCT_BIT_KHR) != 0;
    caps.separateReferenceImages = (videoCapabilities.flags & VK_VIDEO_CAPABILITY_SEPARATE_REFERENCE_IMAGES_BIT_KHR) != 0;
    
    caps.maxDpbSlots = videoCapabilities.maxDpbSlots;
    caps.maxActiveReferencePictures = videoCapabilities.maxActiveReferencePictures;
    caps.minCodedExtent = videoCapabilities.minCodedExtent;
    caps.maxCodedExtent = videoCapabilities.maxCodedExtent;
    
    LOGI("Vulkan Video H.265 decode supported:\n");
    LOGI("  Video queue family: %u\n", caps.videoQueueFamilyIndex);
    LOGI("  Max DPB slots: %u\n", caps.maxDpbSlots);
    LOGI("  Max active references: %u\n", caps.maxActiveReferencePictures);
    LOGI("  DPB/Output coincide: %s\n", caps.dpbAndOutputCoincide ? "yes" : "no");
    LOGI("  DPB/Output distinct: %s\n", caps.dpbAndOutputDistinct ? "yes" : "no");
    LOGI("  Separate ref images: %s\n", caps.separateReferenceImages ? "yes" : "no");
    LOGI("  Max resolution: %ux%u\n", caps.maxCodedExtent.width, caps.maxCodedExtent.height);
    
    return true;
}

bool VulkanVideoDecoder::initialize(VkDevice device, VkPhysicalDevice physicalDevice,
                                     uint32_t videoQueueFamilyIndex, VkQueue videoQueue)
{
    if (m_initialized) {
        LOGW("VulkanVideoDecoder already initialized\n");
        return true;
    }
    
    m_device = device;
    m_physicalDevice = physicalDevice;
    m_videoQueueFamilyIndex = videoQueueFamilyIndex;
    m_videoQueue = videoQueue;
    
    // Query capabilities for this device
    VulkanVideoCapabilities caps;
    if (!queryCapabilities(physicalDevice, caps)) {
        LOGE("Device does not support Vulkan Video decode\n");
        return false;
    }
    
    m_maxDpbSlots = std::min(caps.maxDpbSlots, 16u);
    
    // Store capabilities for later use
    m_decodeCapabilities = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_CAPABILITIES_KHR,
    };
    m_videoCapabilities = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_CAPABILITIES_KHR,
        .pNext = &m_decodeCapabilities,
    };
    
    // Create command resources
    if (!createCommandResources()) {
        LOGE("Failed to create command resources\n");
        return false;
    }
    
    m_initialized = true;
    LOGI("VulkanVideoDecoder initialized successfully\n");
    return true;
}

bool VulkanVideoDecoder::open(const std::filesystem::path& filepath)
{
    if (!m_initialized) {
        LOGE("VulkanVideoDecoder not initialized\n");
        return false;
    }
    
    if (m_fileOpen) {
        close();
    }
    
    // Open video file with FFmpeg for parsing
    std::string pathStr = filepath.string();
    int ret = avformat_open_input(&m_formatContext, pathStr.c_str(), nullptr, nullptr);
    if (ret < 0) {
        char errBuf[256];
        av_strerror(ret, errBuf, sizeof(errBuf));
        LOGE("Failed to open video file: %s (%s)\n", pathStr.c_str(), errBuf);
        return false;
    }
    
    ret = avformat_find_stream_info(m_formatContext, nullptr);
    if (ret < 0) {
        LOGE("Failed to find stream info\n");
        avformat_close_input(&m_formatContext);
        return false;
    }
    
    // Find H.265 video stream
    m_videoStreamIndex = -1;
    for (unsigned int i = 0; i < m_formatContext->nb_streams; ++i) {
        AVStream* stream = m_formatContext->streams[i];
        if (stream->codecpar->codec_type == AVMEDIA_TYPE_VIDEO &&
            stream->codecpar->codec_id == AV_CODEC_ID_HEVC) {
            m_videoStreamIndex = static_cast<int>(i);
            break;
        }
    }
    
    if (m_videoStreamIndex < 0) {
        LOGE("No H.265/HEVC video stream found in file\n");
        avformat_close_input(&m_formatContext);
        return false;
    }
    
    AVStream* videoStream = m_formatContext->streams[m_videoStreamIndex];
    AVCodecParameters* codecParams = videoStream->codecpar;
    
    m_width = codecParams->width;
    m_height = codecParams->height;
    
    if (videoStream->avg_frame_rate.den > 0) {
        m_frameRate = av_q2d(videoStream->avg_frame_rate);
    } else if (videoStream->r_frame_rate.den > 0) {
        m_frameRate = av_q2d(videoStream->r_frame_rate);
    } else {
        m_frameRate = 30.0;
    }
    
    if (m_formatContext->duration > 0) {
        m_duration = static_cast<double>(m_formatContext->duration) / AV_TIME_BASE;
    }
    
    m_frameCount = videoStream->nb_frames;
    if (m_frameCount == 0 && m_duration > 0 && m_frameRate > 0) {
        m_frameCount = static_cast<int64_t>(m_duration * m_frameRate);
    }
    
    // Allocate packet
    m_packet = av_packet_alloc();
    if (!m_packet) {
        LOGE("Failed to allocate AVPacket\n");
        avformat_close_input(&m_formatContext);
        return false;
    }
    
    // Set up H.265 video profile
    m_h265Profile = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_PROFILE_INFO_KHR,
        .stdProfileIdc = STD_VIDEO_H265_PROFILE_IDC_MAIN,
    };
    
    m_videoProfile = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_PROFILE_INFO_KHR,
        .pNext = &m_h265Profile,
        .videoCodecOperation = VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR,
        .chromaSubsampling = VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR,
        .lumaBitDepth = VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR,
        .chromaBitDepth = VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR,
    };
    
    m_videoProfileList = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_PROFILE_LIST_INFO_KHR,
        .profileCount = 1,
        .pProfiles = &m_videoProfile,
    };
    
    // Query video capabilities with profile
    VkResult result = vkGetPhysicalDeviceVideoCapabilitiesKHR(m_physicalDevice, &m_videoProfile, &m_videoCapabilities);
    if (result != VK_SUCCESS) {
        LOGE("Failed to query video capabilities for file: %d\n", result);
        av_packet_free(&m_packet);
        avformat_close_input(&m_formatContext);
        return false;
    }
    
    // Determine output format
    m_outputFormat = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;  // NV12 format (common for H.265)
    
    // Create video session
    if (!createVideoSession()) {
        LOGE("Failed to create video session\n");
        av_packet_free(&m_packet);
        avformat_close_input(&m_formatContext);
        return false;
    }
    
    // Allocate session memory
    if (!allocateSessionMemory()) {
        LOGE("Failed to allocate session memory\n");
        vkDestroyVideoSessionKHR(m_device, m_videoSession, nullptr);
        m_videoSession = VK_NULL_HANDLE;
        av_packet_free(&m_packet);
        avformat_close_input(&m_formatContext);
        return false;
    }
    
    // Create bitstream buffer
    if (!createBitstreamBuffer(MAX_BITSTREAM_SIZE)) {
        LOGE("Failed to create bitstream buffer\n");
        destroyResources();
        return false;
    }
    
    // Create DPB resources
    if (!createDpbResources()) {
        LOGE("Failed to create DPB resources\n");
        destroyResources();
        return false;
    }
    
    // Create output resources
    if (!createOutputResources()) {
        LOGE("Failed to create output resources\n");
        destroyResources();
        return false;
    }
    
    // Create YCbCr conversion for sampling decoded frames
    if (!createYcbcrConversion()) {
        LOGE("Failed to create YCbCr conversion\n");
        destroyResources();
        return false;
    }
    
    // Extract parameter sets from extradata
    if (codecParams->extradata && codecParams->extradata_size > 0) {
        extractParameterSets(codecParams->extradata, codecParams->extradata_size);
    }
    
    // Create video session parameters (after extracting VPS/SPS/PPS)
    if (!createVideoSessionParameters()) {
        LOGE("Failed to create video session parameters\n");
        destroyResources();
        return false;
    }
    
    m_fileOpen = true;
    m_currentFrameIndex = 0;
    
    LOGI("Opened H.265 video: %ux%u @ %.2f fps, duration: %.2f s, frames: %lld\n",
         m_width, m_height, m_frameRate, m_duration, static_cast<long long>(m_frameCount));
    
    return true;
}

void VulkanVideoDecoder::close()
{
    if (!m_fileOpen) {
        return;
    }
    
    // Wait for any pending decode operations
    if (m_decodeFence != VK_NULL_HANDLE) {
        vkWaitForFences(m_device, 1, &m_decodeFence, VK_TRUE, UINT64_MAX);
    }
    
    // Cleanup FFmpeg
    if (m_packet) {
        av_packet_free(&m_packet);
        m_packet = nullptr;
    }
    
    if (m_formatContext) {
        avformat_close_input(&m_formatContext);
        m_formatContext = nullptr;
    }
    
    m_videoStreamIndex = -1;
    m_parameterSetsReceived = false;
    m_vpsData.clear();
    m_spsData.clear();
    m_ppsData.clear();
    
    m_fileOpen = false;
    m_currentFrameIndex = 0;
}

bool VulkanVideoDecoder::decodeNextFrame(VulkanDecodedFrame& frame)
{
    if (!m_fileOpen) {
        return false;
    }
    
    // Read next packet
    while (true) {
        int ret = av_read_frame(m_formatContext, m_packet);
        if (ret < 0) {
            if (ret == AVERROR_EOF) {
                LOGD("End of video stream\n");
            } else {
                char errBuf[256];
                av_strerror(ret, errBuf, sizeof(errBuf));
                LOGE("Error reading frame: %s\n", errBuf);
            }
            return false;
        }
        
        if (m_packet->stream_index == m_videoStreamIndex) {
            break;
        }
        
        av_packet_unref(m_packet);
    }
    
    // Parse and decode the frame
    bool success = parseAndDecodeFrame(frame);
    av_packet_unref(m_packet);
    
    return success;
}

bool VulkanVideoDecoder::parseAndDecodeFrame(VulkanDecodedFrame& frame)
{
    // Check if we have an available output frame slot
    if (m_availableOutputFrames.empty()) {
        LOGE("No available output frame slots\n");
        return false;
    }
    
    uint32_t outputSlotIndex = m_availableOutputFrames.front();
    m_availableOutputFrames.pop_front();
    
    // Copy bitstream data to GPU buffer
    size_t bitstreamSize = m_packet->size;
    if (bitstreamSize > m_bitstreamBufferSize) {
        LOGE("Bitstream too large: %zu > %llu\n", bitstreamSize, 
             static_cast<unsigned long long>(m_bitstreamBufferSize));
        m_availableOutputFrames.push_back(outputSlotIndex);
        return false;
    }
    
    // Add start code if needed (H.265 Annex B format)
    std::vector<uint8_t> bitstreamData;
    if (m_packet->data[0] != 0 || m_packet->data[1] != 0 || 
        (m_packet->data[2] != 0 && m_packet->data[2] != 1)) {
        // NAL unit length prefix format, convert to Annex B
        const uint8_t* data = m_packet->data;
        size_t remaining = m_packet->size;
        
        while (remaining >= 4) {
            uint32_t nalSize = (data[0] << 24) | (data[1] << 16) | (data[2] << 8) | data[3];
            data += 4;
            remaining -= 4;
            
            if (nalSize > remaining) {
                break;
            }
            
            bitstreamData.insert(bitstreamData.end(), START_CODE, START_CODE + START_CODE_SIZE);
            bitstreamData.insert(bitstreamData.end(), data, data + nalSize);
            data += nalSize;
            remaining -= nalSize;
        }
    } else {
        // Already in Annex B format
        bitstreamData.assign(m_packet->data, m_packet->data + m_packet->size);
    }
    
    // Copy to GPU buffer
    memcpy(m_bitstreamMapped, bitstreamData.data(), bitstreamData.size());
    
    // TODO: Parse slice headers to determine reference frames
    // For this PoC, assume no B-frames and simple I/P structure
    std::vector<int32_t> refSlots;
    
    // Submit decode command
    if (!submitDecodeCommand(bitstreamData.data(), bitstreamData.size(), outputSlotIndex, refSlots)) {
        LOGE("Failed to submit decode command\n");
        m_availableOutputFrames.push_back(outputSlotIndex);
        return false;
    }
    
    // Fill output frame info
    DpbSlot& outputSlot = m_outputFrames[outputSlotIndex];
    frame.image = outputSlot.image;
    frame.imageView = outputSlot.imageView;
    frame.format = m_outputFormat;
    frame.currentLayout = VK_IMAGE_LAYOUT_VIDEO_DECODE_DST_KHR;
    frame.width = m_width;
    frame.height = m_height;
    frame.frameIndex = m_currentFrameIndex++;
    
    if (m_packet->pts != AV_NOPTS_VALUE) {
        AVStream* stream = m_formatContext->streams[m_videoStreamIndex];
        frame.pts = m_packet->pts;
        frame.timestampSec = static_cast<double>(m_packet->pts) * av_q2d(stream->time_base);
    } else {
        frame.pts = frame.frameIndex;
        frame.timestampSec = static_cast<double>(frame.frameIndex) / m_frameRate;
    }
    
    frame.ycbcrConversion = m_ycbcrConversion;
    frame.ycbcrSampler = m_ycbcrSampler;
    
    return true;
}

bool VulkanVideoDecoder::submitDecodeCommand(const uint8_t* bitstreamData, size_t bitstreamSize,
                                              uint32_t outputSlotIndex, const std::vector<int32_t>& refSlots)
{
    // Wait for previous decode to complete
    vkWaitForFences(m_device, 1, &m_decodeFence, VK_TRUE, UINT64_MAX);
    vkResetFences(m_device, 1, &m_decodeFence);
    
    // Reset and begin command buffer
    vkResetCommandBuffer(m_cmdBuffer, 0);
    
    VkCommandBufferBeginInfo beginInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(m_cmdBuffer, &beginInfo);
    
    // Begin video coding scope
    DpbSlot& outputSlot = m_outputFrames[outputSlotIndex];
    
    VkVideoReferenceSlotInfoKHR outputSlotInfo = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR,
        .slotIndex = -1,  // -1 for decode output
        .pPictureResource = nullptr,
    };
    
    VkVideoPictureResourceInfoKHR outputPictureResource = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR,
        .codedOffset = {0, 0},
        .codedExtent = {m_width, m_height},
        .baseArrayLayer = 0,
        .imageViewBinding = outputSlot.imageView,
    };
    
    VkVideoBeginCodingInfoKHR beginCodingInfo = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_BEGIN_CODING_INFO_KHR,
        .videoSession = m_videoSession,
        .videoSessionParameters = m_videoSessionParams,
        .referenceSlotCount = 0,
        .pReferenceSlots = nullptr,
    };
    
    vkCmdBeginVideoCodingKHR(m_cmdBuffer, &beginCodingInfo);
    
    // Transition output image to decode destination layout
    VkImageMemoryBarrier2 preDecodeBarrier = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
        .srcAccessMask = VK_ACCESS_2_NONE,
        .dstStageMask = VK_PIPELINE_STAGE_2_VIDEO_DECODE_BIT_KHR,
        .dstAccessMask = VK_ACCESS_2_VIDEO_DECODE_WRITE_BIT_KHR,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_VIDEO_DECODE_DST_KHR,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = outputSlot.image,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };
    
    VkDependencyInfo preDependency = {
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &preDecodeBarrier,
    };
    vkCmdPipelineBarrier2(m_cmdBuffer, &preDependency);
    
    // Find slice segment offsets in the bitstream (look for start codes)
    std::vector<uint32_t> sliceOffsets;
    for (size_t i = 0; i + 2 < bitstreamSize; ++i) {
        // Check for start code (0x000001 or 0x00000001)
        if (bitstreamData[i] == 0 && bitstreamData[i + 1] == 0) {
            if (bitstreamData[i + 2] == 1) {
                // 3-byte start code
                size_t nalStart = i + 3;
                if (nalStart < bitstreamSize) {
                    uint8_t nalType = (bitstreamData[nalStart] >> 1) & 0x3F;
                    // Slice NAL types: 0-9 (VCL NAL units)
                    if (nalType <= 9 || (nalType >= 16 && nalType <= 21)) {
                        sliceOffsets.push_back(static_cast<uint32_t>(i));
                    }
                }
            } else if (i + 3 < bitstreamSize && bitstreamData[i + 2] == 0 && bitstreamData[i + 3] == 1) {
                // 4-byte start code
                size_t nalStart = i + 4;
                if (nalStart < bitstreamSize) {
                    uint8_t nalType = (bitstreamData[nalStart] >> 1) & 0x3F;
                    if (nalType <= 9 || (nalType >= 16 && nalType <= 21)) {
                        sliceOffsets.push_back(static_cast<uint32_t>(i));
                    }
                }
                i++; // Skip extra byte
            }
        }
    }
    
    // Ensure at least one slice offset
    if (sliceOffsets.empty()) {
        sliceOffsets.push_back(0);
    }
    
    // Create H.265 picture info with minimal required fields
    StdVideoDecodeH265PictureInfo stdPictureInfo = {};
    stdPictureInfo.flags.IrapPicFlag = (m_currentFrameIndex == 0) ? 1 : 0;
    stdPictureInfo.flags.IdrPicFlag = (m_currentFrameIndex == 0) ? 1 : 0;
    stdPictureInfo.sps_video_parameter_set_id = 0;
    stdPictureInfo.pps_seq_parameter_set_id = 0;
    stdPictureInfo.pps_pic_parameter_set_id = 0;
    stdPictureInfo.NumDeltaPocsOfRefRpsIdx = 0;
    stdPictureInfo.PicOrderCntVal = static_cast<int32_t>(m_currentFrameIndex);
    stdPictureInfo.NumBitsForSTRefPicSetInSlice = 0;
    
    VkVideoDecodeH265PictureInfoKHR h265PictureInfo = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_PICTURE_INFO_KHR,
        .pStdPictureInfo = &stdPictureInfo,
        .sliceSegmentCount = static_cast<uint32_t>(sliceOffsets.size()),
        .pSliceSegmentOffsets = sliceOffsets.data(),
    };
    
    VkVideoDecodeInfoKHR decodeInfo = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_INFO_KHR,
        .pNext = &h265PictureInfo,
        .srcBuffer = m_bitstreamBuffer,
        .srcBufferOffset = 0,
        .srcBufferRange = bitstreamSize,
        .dstPictureResource = outputPictureResource,
        .pSetupReferenceSlot = nullptr,  // No setup reference for this PoC
        .referenceSlotCount = 0,
        .pReferenceSlots = nullptr,
    };
    
    vkCmdDecodeVideoKHR(m_cmdBuffer, &decodeInfo);
    
    // End video coding scope
    VkVideoEndCodingInfoKHR endCodingInfo = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_END_CODING_INFO_KHR,
    };
    vkCmdEndVideoCodingKHR(m_cmdBuffer, &endCodingInfo);
    
    vkEndCommandBuffer(m_cmdBuffer);
    
    // Submit
    VkSubmitInfo submitInfo = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &m_cmdBuffer,
    };
    
    VkResult result = vkQueueSubmit(m_videoQueue, 1, &submitInfo, m_decodeFence);
    if (result != VK_SUCCESS) {
        LOGE("Failed to submit decode command: %d\n", result);
        return false;
    }
    
    // Wait for decode to complete (synchronous for PoC)
    vkWaitForFences(m_device, 1, &m_decodeFence, VK_TRUE, UINT64_MAX);
    
    return true;
}

bool VulkanVideoDecoder::seekToTime(double timestampSec)
{
    if (!m_fileOpen) {
        return false;
    }
    
    int64_t timestamp = static_cast<int64_t>(timestampSec * AV_TIME_BASE);
    int ret = av_seek_frame(m_formatContext, -1, timestamp, AVSEEK_FLAG_BACKWARD);
    if (ret < 0) {
        LOGE("Failed to seek to time %.2f\n", timestampSec);
        return false;
    }
    
    return true;
}

void VulkanVideoDecoder::getDimensions(uint32_t& width, uint32_t& height) const
{
    width = m_width;
    height = m_height;
}

void VulkanVideoDecoder::transitionFrameLayout(VkCommandBuffer cmdBuffer, VulkanDecodedFrame& frame,
                                                VkImageLayout dstLayout, uint32_t dstQueueFamilyIndex)
{
    if (frame.currentLayout == dstLayout) {
        return;
    }
    
    // If dstQueueFamilyIndex is not specified, use IGNORED (no ownership transfer)
    // If specified and different from video queue, this is the "acquire" side of a queue family transfer
    bool isQueueTransfer = (dstQueueFamilyIndex != VK_QUEUE_FAMILY_IGNORED && 
                            dstQueueFamilyIndex != m_videoQueueFamilyIndex);
    
    VkImageMemoryBarrier2 barrier = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_VIDEO_DECODE_BIT_KHR,
        .srcAccessMask = VK_ACCESS_2_VIDEO_DECODE_WRITE_BIT_KHR,
        .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        .oldLayout = frame.currentLayout,
        .newLayout = dstLayout,
        .srcQueueFamilyIndex = isQueueTransfer ? m_videoQueueFamilyIndex : VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = isQueueTransfer ? dstQueueFamilyIndex : VK_QUEUE_FAMILY_IGNORED,
        .image = frame.image,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };
    
    VkDependencyInfo dependency = {
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &barrier,
    };
    
    vkCmdPipelineBarrier2(cmdBuffer, &dependency);
    frame.currentLayout = dstLayout;
    
    if (isQueueTransfer) {
        LOGD("Queue family ownership transferred: video(%u) -> graphics(%u)\n", 
             m_videoQueueFamilyIndex, dstQueueFamilyIndex);
    }
}

void VulkanVideoDecoder::releaseFrame(VulkanDecodedFrame& frame)
{
    // Find the slot index and return it to available pool
    for (uint32_t i = 0; i < m_outputFrames.size(); ++i) {
        if (m_outputFrames[i].image == frame.image) {
            m_availableOutputFrames.push_back(i);
            break;
        }
    }
}

// Private methods

bool VulkanVideoDecoder::createVideoSession()
{
    VkVideoSessionCreateInfoKHR createInfo = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_CREATE_INFO_KHR,
        .queueFamilyIndex = m_videoQueueFamilyIndex,
        .pVideoProfile = &m_videoProfile,
        .pictureFormat = m_outputFormat,
        .maxCodedExtent = {m_width, m_height},
        .referencePictureFormat = m_outputFormat,
        .maxDpbSlots = m_maxDpbSlots,
        .maxActiveReferencePictures = m_maxDpbSlots - 1,
        .pStdHeaderVersion = &m_videoCapabilities.stdHeaderVersion,
    };
    
    VkResult result = vkCreateVideoSessionKHR(m_device, &createInfo, nullptr, &m_videoSession);
    if (result != VK_SUCCESS) {
        LOGE("Failed to create video session: %d\n", result);
        return false;
    }
    
    LOGI("Created video session for %ux%u\n", m_width, m_height);
    return true;
}

bool VulkanVideoDecoder::createVideoSessionParameters()
{
    // Build parameter sets add info if we have parsed VPS/SPS/PPS
    VkVideoDecodeH265SessionParametersAddInfoKHR addInfo = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_SESSION_PARAMETERS_ADD_INFO_KHR,
        .stdVPSCount = 0,
        .pStdVPSs = nullptr,
        .stdSPSCount = 0,
        .pStdSPSs = nullptr,
        .stdPPSCount = 0,
        .pStdPPSs = nullptr,
    };
    
    // Placeholder structures for VPS/SPS/PPS
    // In a full implementation, these would be populated by parsing the extradata
    StdVideoH265VideoParameterSet vps = {};
    StdVideoH265SequenceParameterSet sps = {};
    StdVideoH265PictureParameterSet pps = {};
    
    // Initialize with minimal defaults for decoding
    StdVideoH265ProfileTierLevel profileTierLevel = {
        .flags = {},
        .general_profile_idc = STD_VIDEO_H265_PROFILE_IDC_MAIN,
        .general_level_idc = STD_VIDEO_H265_LEVEL_IDC_5_1,
    };
    
    StdVideoH265DecPicBufMgr decPicBufMgr = {};
    decPicBufMgr.max_latency_increase_plus1[0] = 1;
    decPicBufMgr.max_dec_pic_buffering_minus1[0] = static_cast<uint8_t>(m_maxDpbSlots - 1);
    decPicBufMgr.max_num_reorder_pics[0] = 0;
    
    if (m_parameterSetsReceived) {
        // We have extracted parameter sets - create minimal structures
        // Full implementation would parse the NAL units properly
        vps.vps_video_parameter_set_id = 0;
        vps.vps_max_sub_layers_minus1 = 0;
        vps.pDecPicBufMgr = &decPicBufMgr;
        vps.pProfileTierLevel = &profileTierLevel;
        
        sps.sps_video_parameter_set_id = 0;
        sps.sps_seq_parameter_set_id = 0;
        sps.sps_max_sub_layers_minus1 = 0;
        sps.chroma_format_idc = STD_VIDEO_H265_CHROMA_FORMAT_IDC_420;
        sps.bit_depth_luma_minus8 = 0;
        sps.bit_depth_chroma_minus8 = 0;
        sps.pic_width_in_luma_samples = m_width;
        sps.pic_height_in_luma_samples = m_height;
        sps.pDecPicBufMgr = &decPicBufMgr;
        sps.pProfileTierLevel = &profileTierLevel;
        
        pps.pps_pic_parameter_set_id = 0;
        pps.pps_seq_parameter_set_id = 0;
        
        addInfo.stdVPSCount = 1;
        addInfo.pStdVPSs = &vps;
        addInfo.stdSPSCount = 1;
        addInfo.pStdSPSs = &sps;
        addInfo.stdPPSCount = 1;
        addInfo.pStdPPSs = &pps;
        
        LOGI("Creating video session parameters with parsed VPS/SPS/PPS\n");
    } else {
        LOGW("No parameter sets available - video session parameters may be incomplete\n");
    }
    
    VkVideoDecodeH265SessionParametersCreateInfoKHR h265Params = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_SESSION_PARAMETERS_CREATE_INFO_KHR,
        .maxStdVPSCount = 1,
        .maxStdSPSCount = 1,
        .maxStdPPSCount = 1,
        .pParametersAddInfo = m_parameterSetsReceived ? &addInfo : nullptr,
    };
    
    VkVideoSessionParametersCreateInfoKHR createInfo = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_PARAMETERS_CREATE_INFO_KHR,
        .pNext = &h265Params,
        .videoSession = m_videoSession,
    };
    
    VkResult result = vkCreateVideoSessionParametersKHR(m_device, &createInfo, nullptr, &m_videoSessionParams);
    if (result != VK_SUCCESS) {
        LOGE("Failed to create video session parameters: %d\n", result);
        return false;
    }
    
    LOGI("Created video session parameters successfully\n");
    return true;
}

bool VulkanVideoDecoder::allocateSessionMemory()
{
    // Query memory requirements
    uint32_t memReqCount = 0;
    vkGetVideoSessionMemoryRequirementsKHR(m_device, m_videoSession, &memReqCount, nullptr);
    
    if (memReqCount == 0) {
        return true;  // No memory required
    }
    
    std::vector<VkVideoSessionMemoryRequirementsKHR> memReqs(memReqCount);
    for (auto& req : memReqs) {
        req.sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_MEMORY_REQUIREMENTS_KHR;
    }
    vkGetVideoSessionMemoryRequirementsKHR(m_device, m_videoSession, &memReqCount, memReqs.data());
    
    // Allocate and bind memory
    std::vector<VkBindVideoSessionMemoryInfoKHR> bindInfos(memReqCount);
    m_sessionMemory.resize(memReqCount);
    
    for (uint32_t i = 0; i < memReqCount; ++i) {
        VkMemoryAllocateInfo allocInfo = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = memReqs[i].memoryRequirements.size,
            .memoryTypeIndex = static_cast<uint32_t>(
                findMemoryType(memReqs[i].memoryRequirements.memoryTypeBits,
                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)),
        };
        
        VkResult result = vkAllocateMemory(m_device, &allocInfo, nullptr, &m_sessionMemory[i]);
        if (result != VK_SUCCESS) {
            LOGE("Failed to allocate session memory: %d\n", result);
            return false;
        }
        
        bindInfos[i] = {
            .sType = VK_STRUCTURE_TYPE_BIND_VIDEO_SESSION_MEMORY_INFO_KHR,
            .memoryBindIndex = memReqs[i].memoryBindIndex,
            .memory = m_sessionMemory[i],
            .memoryOffset = 0,
            .memorySize = memReqs[i].memoryRequirements.size,
        };
    }
    
    VkResult result = vkBindVideoSessionMemoryKHR(m_device, m_videoSession,
                                                   static_cast<uint32_t>(bindInfos.size()),
                                                   bindInfos.data());
    if (result != VK_SUCCESS) {
        LOGE("Failed to bind session memory: %d\n", result);
        return false;
    }
    
    return true;
}

bool VulkanVideoDecoder::createDpbResources()
{
    m_dpbSlots.resize(m_maxDpbSlots);
    
    for (uint32_t i = 0; i < m_maxDpbSlots; ++i) {
        DpbSlot& slot = m_dpbSlots[i];
        slot.slotIndex = static_cast<int32_t>(i);
        
        // Create DPB image
        VkImageCreateInfo imageInfo = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = &m_videoProfileList,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = m_outputFormat,
            .extent = {m_width, m_height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR | VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };
        
        VkResult result = vkCreateImage(m_device, &imageInfo, nullptr, &slot.image);
        if (result != VK_SUCCESS) {
            LOGE("Failed to create DPB image %u: %d\n", i, result);
            return false;
        }
        
        // Allocate memory
        VkMemoryRequirements memReqs;
        vkGetImageMemoryRequirements(m_device, slot.image, &memReqs);
        
        VkMemoryAllocateInfo allocInfo = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = memReqs.size,
            .memoryTypeIndex = static_cast<uint32_t>(
                findMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)),
        };
        
        result = vkAllocateMemory(m_device, &allocInfo, nullptr, &slot.memory);
        if (result != VK_SUCCESS) {
            LOGE("Failed to allocate DPB memory %u: %d\n", i, result);
            return false;
        }
        
        vkBindImageMemory(m_device, slot.image, slot.memory, 0);
        
        // Create image view
        VkImageViewCreateInfo viewInfo = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = slot.image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = m_outputFormat,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };
        
        result = vkCreateImageView(m_device, &viewInfo, nullptr, &slot.imageView);
        if (result != VK_SUCCESS) {
            LOGE("Failed to create DPB image view %u: %d\n", i, result);
            return false;
        }
    }
    
    LOGI("Created %u DPB slots\n", m_maxDpbSlots);
    return true;
}

bool VulkanVideoDecoder::createOutputResources()
{
    m_outputFrames.resize(NUM_OUTPUT_FRAMES);
    
    for (uint32_t i = 0; i < NUM_OUTPUT_FRAMES; ++i) {
        DpbSlot& slot = m_outputFrames[i];
        slot.slotIndex = static_cast<int32_t>(i);
        
        // Create output image
        VkImageCreateInfo imageInfo = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = &m_videoProfileList,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = m_outputFormat,
            .extent = {m_width, m_height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR | VK_IMAGE_USAGE_SAMPLED_BIT |
                     VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };
        
        VkResult result = vkCreateImage(m_device, &imageInfo, nullptr, &slot.image);
        if (result != VK_SUCCESS) {
            LOGE("Failed to create output image %u: %d\n", i, result);
            return false;
        }
        
        // Allocate memory
        VkMemoryRequirements memReqs;
        vkGetImageMemoryRequirements(m_device, slot.image, &memReqs);
        
        VkMemoryAllocateInfo allocInfo = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = memReqs.size,
            .memoryTypeIndex = static_cast<uint32_t>(
                findMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)),
        };
        
        result = vkAllocateMemory(m_device, &allocInfo, nullptr, &slot.memory);
        if (result != VK_SUCCESS) {
            LOGE("Failed to allocate output memory %u: %d\n", i, result);
            return false;
        }
        
        vkBindImageMemory(m_device, slot.image, slot.memory, 0);
        
        // Create image view with YCbCr conversion info
        VkImageViewCreateInfo viewInfo = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = &m_ycbcrConversionInfo,
            .image = slot.image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = m_outputFormat,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };
        
        result = vkCreateImageView(m_device, &viewInfo, nullptr, &slot.imageView);
        if (result != VK_SUCCESS) {
            LOGE("Failed to create output image view %u: %d\n", i, result);
            return false;
        }
        
        m_availableOutputFrames.push_back(i);
    }
    
    LOGI("Created %u output frames\n", NUM_OUTPUT_FRAMES);
    return true;
}

bool VulkanVideoDecoder::createBitstreamBuffer(VkDeviceSize size)
{
    m_bitstreamBufferSize = size;
    
    VkBufferCreateInfo bufferInfo = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = &m_videoProfileList,
        .size = size,
        .usage = VK_BUFFER_USAGE_VIDEO_DECODE_SRC_BIT_KHR,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    
    VkResult result = vkCreateBuffer(m_device, &bufferInfo, nullptr, &m_bitstreamBuffer);
    if (result != VK_SUCCESS) {
        LOGE("Failed to create bitstream buffer: %d\n", result);
        return false;
    }
    
    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(m_device, m_bitstreamBuffer, &memReqs);
    
    VkMemoryAllocateInfo allocInfo = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = memReqs.size,
        .memoryTypeIndex = static_cast<uint32_t>(
            findMemoryType(memReqs.memoryTypeBits,
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)),
    };
    
    result = vkAllocateMemory(m_device, &allocInfo, nullptr, &m_bitstreamMemory);
    if (result != VK_SUCCESS) {
        LOGE("Failed to allocate bitstream memory: %d\n", result);
        return false;
    }
    
    vkBindBufferMemory(m_device, m_bitstreamBuffer, m_bitstreamMemory, 0);
    vkMapMemory(m_device, m_bitstreamMemory, 0, size, 0, &m_bitstreamMapped);
    
    return true;
}

bool VulkanVideoDecoder::createCommandResources()
{
    VkCommandPoolCreateInfo poolInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = m_videoQueueFamilyIndex,
    };
    
    VkResult result = vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_cmdPool);
    if (result != VK_SUCCESS) {
        LOGE("Failed to create command pool: %d\n", result);
        return false;
    }
    
    VkCommandBufferAllocateInfo allocInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = m_cmdPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    
    result = vkAllocateCommandBuffers(m_device, &allocInfo, &m_cmdBuffer);
    if (result != VK_SUCCESS) {
        LOGE("Failed to allocate command buffer: %d\n", result);
        return false;
    }
    
    VkFenceCreateInfo fenceInfo = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT,
    };
    
    result = vkCreateFence(m_device, &fenceInfo, nullptr, &m_decodeFence);
    if (result != VK_SUCCESS) {
        LOGE("Failed to create fence: %d\n", result);
        return false;
    }
    
    return true;
}

bool VulkanVideoDecoder::createYcbcrConversion()
{
    VkSamplerYcbcrConversionCreateInfo conversionInfo = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO,
        .format = m_outputFormat,
        .ycbcrModel = VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709,
        .ycbcrRange = VK_SAMPLER_YCBCR_RANGE_ITU_NARROW,
        .components = {
            VK_COMPONENT_SWIZZLE_IDENTITY,
            VK_COMPONENT_SWIZZLE_IDENTITY,
            VK_COMPONENT_SWIZZLE_IDENTITY,
            VK_COMPONENT_SWIZZLE_IDENTITY,
        },
        .xChromaOffset = VK_CHROMA_LOCATION_MIDPOINT,
        .yChromaOffset = VK_CHROMA_LOCATION_MIDPOINT,
        .chromaFilter = VK_FILTER_LINEAR,
        .forceExplicitReconstruction = VK_FALSE,
    };
    
    VkResult result = vkCreateSamplerYcbcrConversion(m_device, &conversionInfo, nullptr, &m_ycbcrConversion);
    if (result != VK_SUCCESS) {
        LOGE("Failed to create YCbCr conversion: %d\n", result);
        return false;
    }
    
    m_ycbcrConversionInfo = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO,
        .conversion = m_ycbcrConversion,
    };
    
    VkSamplerCreateInfo samplerInfo = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext = &m_ycbcrConversionInfo,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK,
        .unnormalizedCoordinates = VK_FALSE,
    };
    
    result = vkCreateSampler(m_device, &samplerInfo, nullptr, &m_ycbcrSampler);
    if (result != VK_SUCCESS) {
        LOGE("Failed to create YCbCr sampler: %d\n", result);
        return false;
    }
    
    return true;
}

bool VulkanVideoDecoder::extractParameterSets(const uint8_t* data, size_t size)
{
    // Parse HEVC decoder configuration record (hvcC) or NAL units
    // This is a simplified implementation - full implementation would parse properly
    
    if (size < 23) {
        return false;
    }
    
    // Check if this is hvcC format (configuration record)
    if (data[0] == 1) {
        // hvcC format - skip header and extract NAL arrays
        size_t offset = 22;
        uint8_t numArrays = data[21];
        
        for (uint8_t i = 0; i < numArrays && offset < size; ++i) {
            if (offset + 3 > size) break;
            
            uint8_t nalType = data[offset] & 0x3F;
            uint16_t numNalus = (data[offset + 1] << 8) | data[offset + 2];
            offset += 3;
            
            for (uint16_t j = 0; j < numNalus && offset + 2 <= size; ++j) {
                uint16_t nalSize = (data[offset] << 8) | data[offset + 1];
                offset += 2;
                
                if (offset + nalSize > size) break;
                
                // Store based on NAL type
                switch (nalType) {
                    case 32:  // VPS
                        m_vpsData.assign(data + offset, data + offset + nalSize);
                        break;
                    case 33:  // SPS
                        m_spsData.assign(data + offset, data + offset + nalSize);
                        break;
                    case 34:  // PPS
                        m_ppsData.assign(data + offset, data + offset + nalSize);
                        break;
                }
                
                offset += nalSize;
            }
        }
        
        m_parameterSetsReceived = !m_vpsData.empty() && !m_spsData.empty() && !m_ppsData.empty();
    }
    
    return m_parameterSetsReceived;
}

void VulkanVideoDecoder::updateVideoSessionParameters()
{
    // Would update session parameters with new VPS/SPS/PPS
    // Omitted for PoC
}

void VulkanVideoDecoder::destroyResources()
{
    if (m_device == VK_NULL_HANDLE) {
        return;
    }
    
    vkDeviceWaitIdle(m_device);
    
    // Destroy YCbCr conversion
    if (m_ycbcrSampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_device, m_ycbcrSampler, nullptr);
        m_ycbcrSampler = VK_NULL_HANDLE;
    }
    if (m_ycbcrConversion != VK_NULL_HANDLE) {
        vkDestroySamplerYcbcrConversion(m_device, m_ycbcrConversion, nullptr);
        m_ycbcrConversion = VK_NULL_HANDLE;
    }
    
    // Destroy output frames
    for (auto& slot : m_outputFrames) {
        if (slot.imageView != VK_NULL_HANDLE) {
            vkDestroyImageView(m_device, slot.imageView, nullptr);
        }
        if (slot.image != VK_NULL_HANDLE) {
            vkDestroyImage(m_device, slot.image, nullptr);
        }
        if (slot.memory != VK_NULL_HANDLE) {
            vkFreeMemory(m_device, slot.memory, nullptr);
        }
    }
    m_outputFrames.clear();
    m_availableOutputFrames.clear();
    
    // Destroy DPB slots
    for (auto& slot : m_dpbSlots) {
        if (slot.imageView != VK_NULL_HANDLE) {
            vkDestroyImageView(m_device, slot.imageView, nullptr);
        }
        if (slot.image != VK_NULL_HANDLE) {
            vkDestroyImage(m_device, slot.image, nullptr);
        }
        if (slot.memory != VK_NULL_HANDLE) {
            vkFreeMemory(m_device, slot.memory, nullptr);
        }
    }
    m_dpbSlots.clear();
    
    // Destroy bitstream buffer
    if (m_bitstreamMapped) {
        vkUnmapMemory(m_device, m_bitstreamMemory);
        m_bitstreamMapped = nullptr;
    }
    if (m_bitstreamBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(m_device, m_bitstreamBuffer, nullptr);
        m_bitstreamBuffer = VK_NULL_HANDLE;
    }
    if (m_bitstreamMemory != VK_NULL_HANDLE) {
        vkFreeMemory(m_device, m_bitstreamMemory, nullptr);
        m_bitstreamMemory = VK_NULL_HANDLE;
    }
    
    // Destroy video session parameters
    if (m_videoSessionParams != VK_NULL_HANDLE) {
        vkDestroyVideoSessionParametersKHR(m_device, m_videoSessionParams, nullptr);
        m_videoSessionParams = VK_NULL_HANDLE;
    }
    
    // Destroy video session
    if (m_videoSession != VK_NULL_HANDLE) {
        vkDestroyVideoSessionKHR(m_device, m_videoSession, nullptr);
        m_videoSession = VK_NULL_HANDLE;
    }
    
    // Free session memory
    for (auto& mem : m_sessionMemory) {
        if (mem != VK_NULL_HANDLE) {
            vkFreeMemory(m_device, mem, nullptr);
        }
    }
    m_sessionMemory.clear();
    
    // Destroy command resources
    if (m_decodeFence != VK_NULL_HANDLE) {
        vkDestroyFence(m_device, m_decodeFence, nullptr);
        m_decodeFence = VK_NULL_HANDLE;
    }
    if (m_cmdPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(m_device, m_cmdPool, nullptr);
        m_cmdPool = VK_NULL_HANDLE;
    }
    
    m_initialized = false;
}

int32_t VulkanVideoDecoder::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memProperties);
    
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
        if ((typeFilter & (1 << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return static_cast<int32_t>(i);
        }
    }
    
    return -1;
}

} // namespace vk_viewer
