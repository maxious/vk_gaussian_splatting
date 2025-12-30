#pragma once
#include <vulkan/vulkan.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/staging.hpp>
#include <nvvk/resources.hpp>
#include <memory>
#include <vector>
#include "depth_parser.h"

class DepthTextureManager {
public:
    struct DepthTexture {
        nvvk::Image image;
        uint32_t width;
        uint32_t height;
        uint64_t lastUsedMs;
    };

    bool initialize(VkDevice device, VkPhysicalDevice physicalDevice,
                 VkQueue graphicsQueue, nvvk::ResourceAllocator* allocator);

    void uploadDepthFrame(const DepthFrame& frame, VkCommandBuffer cmd);
    void cleanup();

    const DepthTexture& getCurrentTexture() const;

private:
    DepthTexture* findOrCreateTexture(uint32_t width, uint32_t height);
    void createTexture(uint32_t width, uint32_t height, DepthTexture& outTexture);

    vk::Device                 m_device;
    vk::PhysicalDevice           m_physicalDevice;
    vk::Queue                  m_graphicsQueue;
    nvvk::ResourceAllocator*   m_allocator{nullptr};
    std::unique_ptr<nvvk::StagingUploader> m_stagingAllocator;

    std::vector<DepthTexture> m_textures;
    size_t m_currentTextureIndex{0};
    
    DepthTexture m_emptyTexture{};
};
