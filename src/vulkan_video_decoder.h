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

#pragma once

/**
 * @file vulkan_video_decoder.h
 * @brief Vulkan Video hardware-accelerated video decoder
 *
 * This is a proof-of-concept implementation of Vulkan Video decoding for
 * H.265/HEVC depth video streams. It provides GPU-accelerated decoding with
 * zero-copy access to decoded frames as VkImage objects.
 *
 * Key features:
 * - Hardware H.265 decoding via VK_KHR_video_decode_queue + VK_KHR_video_decode_h265
 * - Decoded frames available directly as VkImage (no CPU->GPU upload)
 * - DPB (Decoded Picture Buffer) management for reference frames
 * - Support for both coincide and distinct DPB modes
 *
 * References:
 * - https://www.khronos.org/blog/an-introduction-to-vulkan-video
 * - https://github.com/KhronosGroup/Vulkan-Video-Samples
 * - https://lynne.ee/vulkan-video-decoding.html
 */

// Use volk for dynamic Vulkan function loading (project convention)
#include <volk.h>
#include <filesystem>
#include <memory>
#include <vector>
#include <mutex>
#include <atomic>
#include <functional>
#include <deque>
#include <optional>

// Forward declarations for FFmpeg (still needed for bitstream parsing)
struct AVFormatContext;
struct AVCodecContext;
struct AVPacket;
struct AVCodecParserContext;

namespace vk_viewer {

/**
 * @brief Vulkan Video capability information
 */
struct VulkanVideoCapabilities {
    bool supported = false;
    bool h264DecodeSupported = false;
    bool h265DecodeSupported = false;
    uint32_t videoQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    
    // DPB configuration
    bool dpbAndOutputCoincide = false;   // Intel-style: output usable as DPB
    bool dpbAndOutputDistinct = false;   // NVIDIA/AMD-style: separate DPB images
    bool separateReferenceImages = true; // Multiple images vs layered image
    
    uint32_t maxDpbSlots = 0;
    uint32_t maxActiveReferencePictures = 0;
    VkExtent2D minCodedExtent = {};
    VkExtent2D maxCodedExtent = {};
};

/**
 * @brief Decoded video frame with Vulkan image handle
 */
struct VulkanDecodedFrame {
    VkImage image = VK_NULL_HANDLE;
    VkImageView imageView = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageLayout currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    uint32_t width = 0;
    uint32_t height = 0;
    double timestampSec = 0.0;
    int64_t pts = 0;
    uint32_t frameIndex = 0;
    
    // For YCbCr sampling
    VkSamplerYcbcrConversion ycbcrConversion = VK_NULL_HANDLE;
    VkSampler ycbcrSampler = VK_NULL_HANDLE;
};

/**
 * @brief DPB slot for reference frame management
 */
struct DpbSlot {
    VkImage image = VK_NULL_HANDLE;
    VkImageView imageView = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    int32_t slotIndex = -1;
    int64_t pts = 0;
    bool inUse = false;
};

/**
 * @brief Vulkan Video hardware-accelerated decoder
 *
 * Provides GPU-accelerated H.265 decoding with decoded frames available
 * directly as VkImage objects for use in graphics/compute pipelines.
 */
class VulkanVideoDecoder {
public:
    VulkanVideoDecoder();
    ~VulkanVideoDecoder();

    /**
     * @brief Check if Vulkan Video is supported on this device
     * @param physicalDevice The physical device to query
     * @param caps Output capabilities structure
     * @return true if video decoding is supported
     */
    static bool queryCapabilities(VkPhysicalDevice physicalDevice, VulkanVideoCapabilities& caps);

    /**
     * @brief Initialize the decoder
     * @param device Vulkan device
     * @param physicalDevice Physical device
     * @param videoQueueFamilyIndex Queue family index for video operations
     * @param videoQueue Video queue handle
     * @return true if initialization succeeded
     */
    bool initialize(VkDevice device, VkPhysicalDevice physicalDevice,
                    uint32_t videoQueueFamilyIndex, VkQueue videoQueue);

    /**
     * @brief Open a video file for decoding
     * @param filepath Path to video file (H.265/HEVC)
     * @return true if file opened successfully
     */
    bool open(const std::filesystem::path& filepath);

    /**
     * @brief Close video file and release resources
     */
    void close();

    /**
     * @brief Check if decoder is initialized
     */
    bool isInitialized() const { return m_initialized; }

    /**
     * @brief Check if a file is open
     */
    bool isOpen() const { return m_fileOpen; }

    /**
     * @brief Decode next frame
     * @param frame Output decoded frame
     * @return true if frame decoded, false if EOF or error
     */
    bool decodeNextFrame(VulkanDecodedFrame& frame);

    /**
     * @brief Seek to timestamp
     * @param timestampSec Timestamp in seconds
     * @return true if seek succeeded
     */
    bool seekToTime(double timestampSec);

    /**
     * @brief Get video duration in seconds
     */
    double getDuration() const { return m_duration; }

    /**
     * @brief Get video frame rate
     */
    double getFrameRate() const { return m_frameRate; }

    /**
     * @brief Get video dimensions
     */
    void getDimensions(uint32_t& width, uint32_t& height) const;

    /**
     * @brief Get frame count
     */
    int64_t getFrameCount() const { return m_frameCount; }

    /**
     * @brief Transition decoded frame to shader-readable layout
     * @param cmdBuffer Command buffer for layout transition
     * @param frame Frame to transition
     * @param dstLayout Target layout (typically VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
     * @param dstQueueFamilyIndex Target queue family (for ownership transfer)
     */
    void transitionFrameLayout(VkCommandBuffer cmdBuffer, VulkanDecodedFrame& frame,
                                VkImageLayout dstLayout,
                                uint32_t dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED);

    /**
     * @brief Get video queue family index (for queue family ownership transfer)
     */
    uint32_t getVideoQueueFamilyIndex() const { return m_videoQueueFamilyIndex; }

    /**
     * @brief Release a decoded frame back to the decoder
     * @param frame Frame to release
     */
    void releaseFrame(VulkanDecodedFrame& frame);

    /**
     * @brief Get the YCbCr sampler for decoded frames
     */
    VkSampler getYcbcrSampler() const { return m_ycbcrSampler; }

    /**
     * @brief Get the sampler YCbCr conversion info for descriptor set creation
     */
    const VkSamplerYcbcrConversionInfo* getYcbcrConversionInfo() const { return &m_ycbcrConversionInfo; }

private:
    // Vulkan handles
    VkDevice m_device = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VkQueue m_videoQueue = VK_NULL_HANDLE;
    uint32_t m_videoQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

    // Video session
    VkVideoSessionKHR m_videoSession = VK_NULL_HANDLE;
    VkVideoSessionParametersKHR m_videoSessionParams = VK_NULL_HANDLE;
    std::vector<VkDeviceMemory> m_sessionMemory;

    // Decode resources
    VkCommandPool m_cmdPool = VK_NULL_HANDLE;
    VkCommandBuffer m_cmdBuffer = VK_NULL_HANDLE;
    VkFence m_decodeFence = VK_NULL_HANDLE;

    // Bitstream buffer
    VkBuffer m_bitstreamBuffer = VK_NULL_HANDLE;
    VkDeviceMemory m_bitstreamMemory = VK_NULL_HANDLE;
    void* m_bitstreamMapped = nullptr;
    VkDeviceSize m_bitstreamBufferSize = 0;

    // DPB (Decoded Picture Buffer)
    std::vector<DpbSlot> m_dpbSlots;
    uint32_t m_maxDpbSlots = 0;

    // Output frames
    std::vector<DpbSlot> m_outputFrames;
    std::deque<uint32_t> m_availableOutputFrames;

    // YCbCr conversion
    VkSamplerYcbcrConversion m_ycbcrConversion = VK_NULL_HANDLE;
    VkSampler m_ycbcrSampler = VK_NULL_HANDLE;
    VkSamplerYcbcrConversionInfo m_ycbcrConversionInfo = {};

    // Video profile
    VkVideoProfileInfoKHR m_videoProfile = {};
    VkVideoProfileListInfoKHR m_videoProfileList = {};
    VkVideoDecodeH265ProfileInfoKHR m_h265Profile = {};
    VkVideoDecodeCapabilitiesKHR m_decodeCapabilities = {};
    VkVideoCapabilitiesKHR m_videoCapabilities = {};

    // FFmpeg for parsing (still required for NAL unit extraction)
    AVFormatContext* m_formatContext = nullptr;
    AVCodecParserContext* m_parserContext = nullptr;
    AVPacket* m_packet = nullptr;
    int m_videoStreamIndex = -1;

    // Video info
    uint32_t m_width = 0;
    uint32_t m_height = 0;
    double m_frameRate = 0.0;
    double m_duration = 0.0;
    int64_t m_frameCount = 0;
    VkFormat m_outputFormat = VK_FORMAT_UNDEFINED;

    // State
    std::atomic<bool> m_initialized{false};
    std::atomic<bool> m_fileOpen{false};
    uint32_t m_currentFrameIndex = 0;

    // H.265 parameter sets (parsed from bitstream)
    std::vector<uint8_t> m_vpsData;
    std::vector<uint8_t> m_spsData;
    std::vector<uint8_t> m_ppsData;
    bool m_parameterSetsReceived = false;

    // Private methods
    bool createVideoSession();
    bool createVideoSessionParameters();
    bool allocateSessionMemory();
    bool createDpbResources();
    bool createOutputResources();
    bool createBitstreamBuffer(VkDeviceSize size);
    bool createCommandResources();
    bool createYcbcrConversion();

    bool parseAndDecodeFrame(VulkanDecodedFrame& frame);
    bool submitDecodeCommand(const uint8_t* bitstreamData, size_t bitstreamSize,
                             uint32_t outputSlotIndex, const std::vector<int32_t>& refSlots);

    void destroyResources();
    int32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    
    // H.265 parsing helpers
    bool extractParameterSets(const uint8_t* data, size_t size);
    void updateVideoSessionParameters();
};

/**
 * @brief Get required device extensions for Vulkan Video
 * @return Vector of extension names
 */
std::vector<const char*> getVulkanVideoExtensions();

/**
 * @brief Check if Vulkan Video extensions are available
 * @param physicalDevice Physical device to check
 * @return true if all required extensions are available
 */
bool checkVulkanVideoExtensionSupport(VkPhysicalDevice physicalDevice);

} // namespace vk_viewer
