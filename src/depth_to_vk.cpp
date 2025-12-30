#include <volk.h>
#include "depth_to_vk.h"
#include <nvutils/logger.hpp>
#include <nvvk/barriers.hpp>

bool DepthTextureManager::initialize(VkDevice device, VkPhysicalDevice physicalDevice,
                                VkQueue graphicsQueue, nvvk::ResourceAllocator* allocator) {
    m_device = device;
    m_physicalDevice = physicalDevice;
    m_graphicsQueue = graphicsQueue;
    m_allocator = allocator;

    m_stagingAllocator = std::make_unique<nvvk::StagingUploader>();
    m_stagingAllocator->init(allocator);

    return true;
}

void DepthTextureManager::cleanup() {
    for (auto& tex : m_textures) {
        if (tex.image.descriptor.imageView) vkDestroyImageView(m_device, tex.image.descriptor.imageView, nullptr);
        m_allocator->destroyImage(tex.image);
    }
    m_textures.clear();
    
    if (m_stagingAllocator) {
        m_stagingAllocator->deinit();
    }
}

const DepthTextureManager::DepthTexture& DepthTextureManager::getCurrentTexture() const {
    if (m_textures.empty()) return m_emptyTexture;
    return m_textures[m_currentTextureIndex];
}

DepthTextureManager::DepthTexture* DepthTextureManager::findOrCreateTexture(uint32_t width, uint32_t height) {
    for (size_t i = 0; i < m_textures.size(); ++i) {
        if (m_textures[i].width == width && m_textures[i].height == height) {
            m_currentTextureIndex = i;
            return &m_textures[i];
        }
    }

    DepthTexture newTex;
    createTexture(width, height, newTex);
    
    if (!newTex.image.image) return nullptr;

    m_textures.push_back(newTex);
    m_currentTextureIndex = m_textures.size() - 1;
    return &m_textures.back();
}

void DepthTextureManager::createTexture(uint32_t width, uint32_t height, DepthTexture& outTexture) {
    vk::ImageCreateInfo info{};
    info.imageType = vk::ImageType::e2D;
    info.format = vk::Format::eR32Sfloat;
    info.extent = vk::Extent3D{width, height, 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = vk::SampleCountFlagBits::e1;
    info.tiling = vk::ImageTiling::eOptimal;
    info.usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
    info.sharingMode = vk::SharingMode::eExclusive;
    info.initialLayout = vk::ImageLayout::eUndefined;

    m_allocator->createImage(outTexture.image, (const VkImageCreateInfo&)info);
    outTexture.width = width;
    outTexture.height = height;

    vk::ImageViewCreateInfo viewInfo{};
    viewInfo.image = outTexture.image.image;
    viewInfo.viewType = vk::ImageViewType::e2D;
    viewInfo.format = vk::Format::eR32Sfloat;
    viewInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    VkImageViewCreateInfo vkViewInfo = viewInfo;
    vkCreateImageView(m_device, &vkViewInfo, nullptr, &outTexture.image.descriptor.imageView);
    outTexture.image.descriptor.imageLayout = static_cast<VkImageLayout>(vk::ImageLayout::eShaderReadOnlyOptimal);
}

void DepthTextureManager::uploadDepthFrame(const DepthFrame& frame, VkCommandBuffer cmd) {
    DepthTexture* texture = findOrCreateTexture(frame.width, frame.height);
    if (!texture) {
        LOGE("Failed to find/create texture for depth frame %dx%d\n", frame.width, frame.height);
        return;
    }

    size_t bufferSize = frame.data.size() * sizeof(float);

    m_stagingAllocator->appendImage(texture->image,
                                  bufferSize,
                                  frame.data.data(),
                                  static_cast<VkImageLayout>(vk::ImageLayout::eShaderReadOnlyOptimal));

    m_stagingAllocator->cmdUploadAppended(cmd);

    texture->lastUsedMs = frame.timestampMs;
}
