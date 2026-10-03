#include "dlss_nr.h"

#ifdef WITH_DLSS_NR

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include <nvutils/logger.hpp>

#include "kernels.h"
#include "nr_graph.h"
#include "nr_model.h"
#include "numeric.h"
#include "reference.h"
#include "vk_context.h"

namespace {

// Must match the `Params` block in shaders/dlss_nr/*.comp (std140; every member is 4-byte aligned and the two
// vec2s land on 8-byte boundaries, so the C++ layout is the same).
struct Params {
  uint32_t fullWidth = 0, fullHeight = 0, validWidth = 0, validHeight = 0;
  uint32_t historyValid = 0, seed = 0;
  float    blendScale = 1.0F;
  float    autoMask = 1.0F, localTone = 1.0F, localStructure = 1.0F, skinStructure = -1.0F, style = 0.0F;
  float    motionScale[2] = {-0.5F, -0.5F};
  float    motionBias[2] = {0.0F, 0.0F};
  // DLSS5NR_DEBUG: 1 publishes the sampled proxy code, 2 publishes the network residual, 3 forces a constant
  // field, 4 keeps the scene but zeroes the noise lane. See shaders/dlss_nr/.
  uint32_t debugStage = 0, padDbg = 0;
  // 1 is the network's own output, 0 the rendered frame.
  float    intensity = 1.0F;
  uint32_t pad2 = 0;
};
static_assert(sizeof(Params) == 80, "Params must match the shader's std140 block");
// std140 rules: scalars align to 4, vec2 to 8, and the struct size rounds up to its largest alignment (8).
// Verified at compile time so a silent offset drift cannot reach the shaders.
static_assert(offsetof(Params, fullWidth) == 0, "fullWidth offset");
static_assert(offsetof(Params, validWidth) == 8, "validWidth offset");
static_assert(offsetof(Params, historyValid) == 16, "historyValid offset");
static_assert(offsetof(Params, seed) == 20, "seed offset");
static_assert(offsetof(Params, blendScale) == 24, "blendScale offset");
static_assert(offsetof(Params, style) == 44, "style offset");
static_assert(offsetof(Params, motionScale) == 48, "motionScale must land on an 8-byte boundary");
static_assert(offsetof(Params, motionBias) == 56, "motionBias must land on an 8-byte boundary");
static_assert(offsetof(Params, debugStage) == 64, "debugStage offset");
static_assert(offsetof(Params, intensity) == 72, "intensity offset");

}  // namespace

struct DlssNrPass::Impl {
  struct Image {
    VkImage        image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView    view = VK_NULL_HANDLE;
    VkFormat       format = VK_FORMAT_UNDEFINED;
    VkImageLayout  layout = VK_IMAGE_LAYOUT_UNDEFINED;
  };
  // A view onto one of the viewer's images. Re-made when the viewer's image changes (a resize, a new G-buffer).
  struct External {
    VkImage     image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat    format = VK_FORMAT_UNDEFINED;
  };

  std::unique_ptr<vk::Context> context;
  std::unique_ptr<nr::Model>   model;
  std::unique_ptr<nr::Kernels> kernels;
  std::unique_ptr<nr::Graph>   graph;
  nr::Geometry                 geometry{};
  nr::Activation*              features = nullptr;
  float                        blendScale = 1.0F;

  VkDevice device = VK_NULL_HANDLE;
  uint32_t width = 0, height = 0;
  std::string shaderDir;
  std::string error;

  Image       history[2];
  Image       output;
  External    color;
  External    motion;
  VkSampler   linearSampler = VK_NULL_HANDLE;
  VkSampler   nearestSampler = VK_NULL_HANDLE;
  vk::Buffer  params;

  VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
  VkPipelineLayout      pipelineLayout = VK_NULL_HANDLE;
  VkPipeline            preprocessPipeline = VK_NULL_HANDLE;
  VkPipeline            compositePipeline = VK_NULL_HANDLE;
  VkDescriptorPool      pool = VK_NULL_HANDLE;
  // [history parity][0 preprocess, 1 composite]. They must be separate sets: binding 3 is the Features buffer for
  // the preprocess (writeonly) and the Head buffer for the composite (readonly).
  VkDescriptorSet       sets[2][2]{};
  VkCommandPool         commandPool = VK_NULL_HANDLE;
  VkCommandBuffer       commands[2]{};   // [history parity], pre-recorded

  uint32_t frames = 0;

  // ---------------------------------------------------------------------------------------
  bool fail(const std::string& message) {
    error = message;
    LOGE("DLSS-NR: %s\n", message.c_str());
    return false;
  }

  bool check(VkResult result, const char* what) {
    if (result == VK_SUCCESS) return true;
    return fail(std::string(what) + " failed (VkResult " + std::to_string(int(result)) + ")");
  }

  // ---------------------------------------------------------------------------------------
  Image createImage(uint32_t inWidth, uint32_t inHeight, VkFormat format, VkImageUsageFlags usage) {
    Image image;
    image.format = format;
    VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = format;
    info.extent = {inWidth, inHeight, 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = usage;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VK_CHECK(vkCreateImage(device, &info, nullptr, &image.image));

    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(device, image.image, &req);
    VkPhysicalDeviceMemoryProperties props{};
    vkGetPhysicalDeviceMemoryProperties(context->physical(), &props);
    uint32_t type = UINT32_MAX;
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i)
    {
      if ((req.memoryTypeBits & (1U << i)) && (props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
      {
        type = i;
        break;
      }
    }
    if(type == UINT32_MAX)
      throw std::runtime_error("DLSS-NR: no device-local memory for an image");
    VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = type;
    VK_CHECK(vkAllocateMemory(device, &alloc, nullptr, &image.memory));
    VK_CHECK(vkBindImageMemory(device, image.image, image.memory, 0));

    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image = image.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VK_CHECK(vkCreateImageView(device, &viewInfo, nullptr, &image.view));
    return image;
  }

  void destroyImage(Image& image) {
    if(image.view)
      vkDestroyImageView(device, image.view, nullptr);
    if(image.image)
      vkDestroyImage(device, image.image, nullptr);
    if(image.memory)
      vkFreeMemory(device, image.memory, nullptr);
    image = Image{};
  }

  void transition(VkCommandBuffer cmd, Image& image, VkImageLayout to, VkPipelineStageFlags srcStage,
                  VkAccessFlags srcAccess, VkPipelineStageFlags dstStage, VkAccessFlags dstAccess) {
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.srcAccessMask = srcAccess;
    barrier.dstAccessMask = dstAccess;
    barrier.oldLayout = image.layout;
    barrier.newLayout = to;
    barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image.image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    image.layout = to;
  }

  void barrier(VkCommandBuffer cmd, VkImage image, VkImageLayout from, VkImageLayout to, VkPipelineStageFlags srcStage,
               VkAccessFlags srcAccess, VkPipelineStageFlags dstStage, VkAccessFlags dstAccess) {
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.srcAccessMask = srcAccess;
    barrier.dstAccessMask = dstAccess;
    barrier.oldLayout = from;
    barrier.newLayout = to;
    barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
  }

  // Re-makes the view when the viewer's image or its format changed. Returns true when it did.
  bool bindExternal(External& external, const GpuImage& image) {
    if(image.image == external.image && image.format == external.format)
      return false;
    if(external.view)
      vkDestroyImageView(device, external.view, nullptr);
    external = External{};
    external.image = image.image;
    external.format = image.format;
    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image = image.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = image.format;
    viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VK_CHECK(vkCreateImageView(device, &viewInfo, nullptr, &external.view));
    return true;
  }

  VkPipeline computePipeline(const std::string& spv) {
    VkShaderModule module = context->loadShaderModule(spv);
    VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, module,
                  "main", nullptr};
    info.layout = pipelineLayout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline));
    return pipeline;
  }

  // ---------------------------------------------------------------------------------------
  void updateSets() {
    for(uint32_t h = 0; h < 2; ++h)
    {
      for(uint32_t k = 0; k < 2; ++k)
      {
        VkDescriptorImageInfo colorInfo{nearestSampler, color.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkDescriptorImageInfo prevInfo{linearSampler, history[h].view, VK_IMAGE_LAYOUT_GENERAL};
        VkDescriptorImageInfo motionInfo{nearestSampler, motion.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        // Binding 3 is the one that differs: the preprocess writes the features, the composite reads the head.
        VkDescriptorBufferInfo bufferInfo{k == 0 ? features->buffer.buffer : graph->head().buffer.buffer, 0,
                                          VK_WHOLE_SIZE};
        VkDescriptorBufferInfo paramsInfo{params.buffer, 0, sizeof(Params)};
        VkDescriptorImageInfo outInfo{VK_NULL_HANDLE, output.view, VK_IMAGE_LAYOUT_GENERAL};
        VkDescriptorImageInfo nextInfo{VK_NULL_HANDLE, history[1 - h].view, VK_IMAGE_LAYOUT_GENERAL};

        VkWriteDescriptorSet writes[7]{};
        for(int i = 0; i < 7; ++i)
        {
          writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
          writes[i].dstSet = sets[h][k];
          writes[i].dstBinding = uint32_t(i);
          writes[i].descriptorCount = 1;
        }
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[0].pImageInfo = &colorInfo;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[1].pImageInfo = &prevInfo;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[2].pImageInfo = &motionInfo;
        writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[3].pBufferInfo = &bufferInfo;
        writes[4].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[4].pBufferInfo = &paramsInfo;
        writes[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[5].pImageInfo = &outInfo;
        writes[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[6].pImageInfo = &nextInfo;
        vkUpdateDescriptorSets(device, 7, writes, 0, nullptr);
      }
    }
  }

  // Feature preprocess -> network -> temporal composite, once per history parity. The graph's ~250 dispatches and
  // their descriptor sets cost milliseconds of CPU to record and nothing in them changes between frames: the
  // per-frame parameters live in a UBO written in the command stream, and the history ping-pongs between two
  // fixed images selected by the parity. So they are pre-recorded into secondary command buffers, which also
  // keeps the graph's descriptor sets alive for the lifetime of the buffer instead of exhausting the pool.
  bool buildCommands() {
    if(!check(vkResetCommandPool(device, commandPool, 0), "vkResetCommandPool"))
      return false;
    for(uint32_t h = 0; h < 2; ++h)
    {
      // The graph's descriptor sets of this parity live in the context's pool h for the command buffer's life.
      context->resetDescriptorPool(h);

      VkCommandBuffer cmd = commands[h];
      VkCommandBufferInheritanceInfo inheritance{VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO};
      VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      begin.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
      begin.pInheritanceInfo = &inheritance;
      VK_CHECK(vkBeginCommandBuffer(cmd, &begin));

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, preprocessPipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &sets[h][0], 0, nullptr);
      vkCmdDispatch(cmd, (geometry.fullWidth + 7) / 8, (geometry.fullHeight + 7) / 8, 1);
      context->computeBarrier(cmd);

      graph->record(cmd, *features);

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compositePipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &sets[h][1], 0, nullptr);
      vkCmdDispatch(cmd, (width + 7) / 8, (height + 7) / 8, 1);

      // The composite's storage writes have to be visible to the copy below and to the next frame's sampling of
      // the other history image.
      VkMemoryBarrier memory{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      memory.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memory.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory, 0,
                           nullptr, 0, nullptr);

      VK_CHECK(vkEndCommandBuffer(cmd));
    }
    return true;
  }

  // ---------------------------------------------------------------------------------------
  bool createSized(uint32_t inWidth, uint32_t inHeight) {
    width = inWidth;
    height = inHeight;
    try
    {
      geometry = nr::Geometry::fromValid(width, height);
      // DLSS5NR_UNFUSED=1 selects the unfused reference kernels instead of the fused 32-channel block. The fused
      // GLSL block is the one that works on 8x8 windows with four phase offsets, so this isolates window-phase
      // artifacts from the rest of the network.
      nr::Graph::Options options;
      options.fusedBlocks = std::getenv("DLSS5NR_UNFUSED") == nullptr;
      graph = std::make_unique<nr::Graph>(*context, *model, *kernels, geometry, options);
      features = graph->allocate("input features", geometry.fullWidth * geometry.fullHeight, 16, nr::Format::F32);
      // The graph allocates its other activations at the first record; do it now so the first visible frame does
      // not pay for it (and so the driver compiles the pipelines up front).
      context->fillZero(features->buffer);
      VkCommandBuffer warmup = context->beginCommands();
      graph->record(warmup, *features);
      context->endAndSubmit(warmup, true);
    }
    catch(const std::exception& e)
    {
      return fail(std::string("graph construction failed: ") + e.what());
    }

    history[0] = createImage(width, height, VK_FORMAT_R16G16B16A16_SFLOAT,
                             VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    history[1] = createImage(width, height, VK_FORMAT_R16G16B16A16_SFLOAT,
                             VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    // Dedicated RGBA8 output, never COLOR_AUX1: that is the RTX temporal state and has different semantics.
    output = createImage(width, height, VK_FORMAT_R8G8B8A8_UNORM,
                         VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

    // The storage images live in GENERAL for their whole life.
    VkCommandBuffer cmd = context->beginCommands();
    VkClearColorValue zero{};
    VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    for(Image* image : {&history[0], &history[1], &output})
    {
      transition(cmd, *image, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                 VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                 VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
      vkCmdClearColorImage(cmd, image->image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
    }
    context->endAndSubmit(cmd, true);

    color = External{};
    motion = External{};
    frames = 0;
    return true;
  }

  void destroySized() {
    if(!device)
      return;
    VK_CHECK(vkDeviceWaitIdle(device));
    VK_CHECK(vkResetCommandPool(device, commandPool, 0));
    for(Image* image : {&history[0], &history[1], &output})
      destroyImage(*image);
    for(External* external : {&color, &motion})
    {
      if(external->view)
        vkDestroyImageView(device, external->view, nullptr);
      *external = External{};
    }
    features = nullptr;
    graph.reset();
    width = height = 0;
  }
};

// -----------------------------------------------------------------------------------------
DlssNrPass::DlssNrPass() : m_impl(std::make_unique<Impl>()) {}

DlssNrPass::~DlssNrPass() { destroy(); }

bool DlssNrPass::init(VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device, uint32_t queueFamily,
                      uint32_t queueIndex, const std::string& shaderDir) {
  Impl& impl = *m_impl;
  if(impl.device)
    return true;
  impl.device = device;
  impl.shaderDir = shaderDir;
  try
  {
    impl.context = std::make_unique<vk::Context>(instance, physicalDevice, device, queueFamily, queueIndex);
  }
  catch(const std::exception& e)
  {
    return impl.fail(std::string("Vulkan context adoption failed: ") + e.what());
  }
  if(!((device != VK_NULL_HANDLE) && (physicalDevice != VK_NULL_HANDLE) && (instance != VK_NULL_HANDLE)))
  {
    return impl.fail("no Vulkan device");
  }

  VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  samplerInfo.magFilter = samplerInfo.minFilter = VK_FILTER_LINEAR;
  samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  samplerInfo.addressModeU = samplerInfo.addressModeV = samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerInfo.maxLod = 1.0F;
  if(!impl.check(vkCreateSampler(device, &samplerInfo, nullptr, &impl.linearSampler), "vkCreateSampler"))
    return false;
  samplerInfo.magFilter = samplerInfo.minFilter = VK_FILTER_NEAREST;
  if(!impl.check(vkCreateSampler(device, &samplerInfo, nullptr, &impl.nearestSampler), "vkCreateSampler"))
    return false;

  // The parameters are written into the command stream (vkCmdUpdateBuffer), so device-local + transfer dst.
  impl.params = impl.context->createBuffer(sizeof(Params), false, "dlss_nr params",
                                           VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

  VkDescriptorSetLayoutBinding bindings[7] = {
      {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
  VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  layoutInfo.bindingCount = 7;
  layoutInfo.pBindings = bindings;
  if(!impl.check(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &impl.setLayout), "vkCreateDescriptorSetLayout"))
    return false;

  VkPipelineLayoutCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pipelineInfo.setLayoutCount = 1;
  pipelineInfo.pSetLayouts = &impl.setLayout;
  if(!impl.check(vkCreatePipelineLayout(device, &pipelineInfo, nullptr, &impl.pipelineLayout), "vkCreatePipelineLayout"))
    return false;

  try
  {
    // Both passes share the layout: the preprocess simply does not read the two storage images.
    impl.preprocessPipeline = impl.computePipeline(shaderDir + "/dlss_nr_preprocess.spv");
    impl.compositePipeline = impl.computePipeline(shaderDir + "/dlss_nr_composite.spv");
  }
  catch(const std::exception& e)
  {
    return impl.fail(std::string("NR shader pipeline creation failed (are the .spv files next to the "
                                 "executable?): ") + e.what());
  }

  VkDescriptorPoolSize poolSizes[4] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 12},
                                       {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4},
                                       {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4},
                                       {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 8}};
  VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  poolInfo.maxSets = 4;
  poolInfo.poolSizeCount = 4;
  poolInfo.pPoolSizes = poolSizes;
  if(!impl.check(vkCreateDescriptorPool(device, &poolInfo, nullptr, &impl.pool), "vkCreateDescriptorPool"))
    return false;

  VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  alloc.descriptorPool = impl.pool;
  alloc.descriptorSetCount = 1;
  alloc.pSetLayouts = &impl.setLayout;
  for(auto& pair : impl.sets)
  {
    for(VkDescriptorSet& set : pair)
    {
      if(!impl.check(vkAllocateDescriptorSets(device, &alloc, &set), "vkAllocateDescriptorSets"))
        return false;
    }
  }

  VkCommandPoolCreateInfo cmdPoolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cmdPoolInfo.queueFamilyIndex = impl.context->queueFamily();
  if(!impl.check(vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &impl.commandPool), "vkCreateCommandPool"))
    return false;
  VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  allocate.commandPool = impl.commandPool;
  allocate.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
  allocate.commandBufferCount = 2;
  if(!impl.check(vkAllocateCommandBuffers(device, &allocate, impl.commands), "vkAllocateCommandBuffers"))
    return false;

  LOGI("DLSS-NR: initialized on %s\n", impl.context->deviceName().c_str());
  return true;
}

bool DlssNrPass::loadModel(const std::string& modelDir, const std::string& kernelDir) {
  Impl& impl = *m_impl;
  if(!impl.context)
    return impl.fail("init() must run first");
  if(impl.model)
    return true;
  if(modelDir.empty())
    return impl.fail("no model directory: pass --dlssNRModel <dir> with an OpenDLSS-NR model directory");

  try
  {
    // verifyHashes=false: the stage digests are verified by the extractor, and re-hashing 141 MiB on every start
    // costs seconds. Set DLSS5NR_VERIFY=1 to check them again.
    const bool verifyHashes = std::getenv("DLSS5NR_VERIFY") != nullptr;
    impl.model = std::make_unique<nr::Model>(*impl.context, modelDir, verifyHashes);
    LOGI("DLSS-NR: model loaded (%u blocks)\n", impl.model->blockCount());
    impl.kernels = std::make_unique<nr::Kernels>(*impl.context, kernelDir);
    LOGI("DLSS-NR: kernels loaded from %s\n", kernelDir.c_str());
    impl.kernels->setSiluTable(ref::siluTable());
  }
  catch(const std::exception& e)
  {
    impl.model.reset();
    return impl.fail(std::string("model load failed: ") + e.what());
  }

  // The learned per-frame blend scale, one f16.
  try
  {
    const nr::Tensor& blend = impl.model->tensor(70, 0, "blend_scale");
    if(blend.byteLength >= 2)
      impl.blendScale = num::f16ToF32(uint16_t(blend.bytes[0] | (blend.bytes[1] << 8)));
  }
  catch(const std::exception& e)
  {
    LOGW("DLSS-NR: no blend scale (%s), using 1.0\n", e.what());
  }
  LOGI("DLSS-NR: history blend scale %g\n", double(impl.blendScale));
  return true;
}

bool DlssNrPass::initialized() const { return m_impl->device != VK_NULL_HANDLE; }

bool DlssNrPass::ready() const { return m_impl->device != VK_NULL_HANDLE && m_impl->graph != nullptr && m_impl->width != 0; }

const std::string& DlssNrPass::error() const { return m_impl->error; }

bool DlssNrPass::resize(uint32_t width, uint32_t height) {
  Impl& impl = *m_impl;
  // Needs the model and the kernels; the graph is built by createSized() below.
  if(!impl.model || !impl.kernels)
    return impl.fail("resize() needs the model: call loadModel() first");
  if(width == impl.width && height == impl.height)
    return true;
  impl.destroySized();
  if(!impl.createSized(width, height))
    return false;
  return impl.buildCommands();
}

uint32_t DlssNrPass::width() const { return m_impl->width; }
uint32_t DlssNrPass::height() const { return m_impl->height; }
uint32_t DlssNrPass::fullWidth() const { return m_impl->geometry.fullWidth; }
uint32_t DlssNrPass::fullHeight() const { return m_impl->geometry.fullHeight; }

bool DlssNrPass::record(VkCommandBuffer cmd, const Frame& frame, const GpuImage& color, const GpuImage& motion) {
  Impl& impl = *m_impl;
  if(!frame.enabled || !ready())
    return false;
  if(!color.valid() || !motion.valid())
    return impl.fail("record(): the scene color or the motion image is missing");
  if(color.image == impl.output.image || motion.image == impl.output.image)
    return impl.fail("record(): the viewer's images must not alias the NR output");

  // Fitting a new size here keeps the caller from having to track it; it is a no-op unless it changed.
  if(uint32_t(color.format) == VK_FORMAT_UNDEFINED)
    return impl.fail("record(): the scene color has no format");
  if(impl.width == 0)
  {
    // The caller passes the size through the image extents; without an explicit resize the pass is not fitted yet.
    return impl.fail("record(): resize() must be called first");
  }

  bool changed = impl.bindExternal(impl.color, color);
  changed = impl.bindExternal(impl.motion, motion) || changed;
  if(changed && !(impl.updateSets(), impl.buildCommands()))
    return false;

  const uint32_t parity = impl.frames & 1U;
  impl.frames++;

  Params params;
  params.fullWidth = impl.geometry.fullWidth;
  params.fullHeight = impl.geometry.fullHeight;
  params.validWidth = impl.width;
  params.validHeight = impl.height;
  params.historyValid = frame.historyValid ? 1U : 0U;
  params.seed = frame.seed;
  params.blendScale = impl.blendScale;
  params.autoMask = frame.autoMask ? 1.0F : 0.0F;
  params.localTone = frame.localTone;
  params.localStructure = frame.localStructure;
  params.skinStructure = frame.skinStructure;
  params.style = float(frame.style);
  params.motionScale[0] = frame.motionScale[0];
  params.motionScale[1] = frame.motionScale[1];
  params.motionBias[0] = frame.motionBias[0];
  params.motionBias[1] = frame.motionBias[1];
  static const uint32_t debugStage = std::getenv("DLSS5NR_DEBUG") ? uint32_t(atoi(std::getenv("DLSS5NR_DEBUG"))) : 0U;
  params.debugStage = debugStage;
  params.intensity = frame.intensity;
  vkCmdUpdateBuffer(cmd, impl.params.buffer, 0, sizeof(Params), &params);
  VkMemoryBarrier paramsBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  paramsBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  paramsBarrier.dstAccessMask = VK_ACCESS_UNIFORM_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &paramsBarrier,
                       0, nullptr, 0, nullptr);

  // The viewer's scene and motion targets: its writes -> the pass's compute reads.
  impl.barrier(cmd, impl.color.image, color.layout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
               VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
               VK_ACCESS_SHADER_READ_BIT);
  impl.barrier(cmd, impl.motion.image, motion.layout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
               VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
               VK_ACCESS_SHADER_READ_BIT);

  vkCmdExecuteCommands(cmd, 1, &impl.commands[parity]);

  // The NR result replaces the viewer's scene target, so everything downstream (ImGui presentation, the XR copy)
  // keeps working unchanged.
  impl.barrier(cmd, impl.output.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
               VK_ACCESS_TRANSFER_READ_BIT);
  impl.barrier(cmd, impl.color.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
               VK_ACCESS_TRANSFER_WRITE_BIT);
  VkImageCopy region{};
  region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  region.extent = {impl.width, impl.height, 1};
  vkCmdCopyImage(cmd, impl.output.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, impl.color.image,
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
  impl.barrier(cmd, impl.color.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
               VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
               VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
  // output_ back to the layout its storage-image descriptor declares (GENERAL).
  impl.transition(cmd, impl.output, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT);
  return true;
}

double DlssNrPass::networkMs() const { return 0.0; }

const std::string& DlssNrPass::deviceName() const {
  static const std::string empty;
  return m_impl->context ? m_impl->context->deviceName() : empty;
}

void DlssNrPass::destroy() {
  Impl& impl = *m_impl;
  if(!impl.device)
    return;
  impl.destroySized();
  vkDeviceWaitIdle(impl.device);
  if(impl.commandPool)
    vkDestroyCommandPool(impl.device, impl.commandPool, nullptr);
  if(impl.pool)
    vkDestroyDescriptorPool(impl.device, impl.pool, nullptr);
  if(impl.preprocessPipeline)
    vkDestroyPipeline(impl.device, impl.preprocessPipeline, nullptr);
  if(impl.compositePipeline)
    vkDestroyPipeline(impl.device, impl.compositePipeline, nullptr);
  if(impl.pipelineLayout)
    vkDestroyPipelineLayout(impl.device, impl.pipelineLayout, nullptr);
  if(impl.setLayout)
    vkDestroyDescriptorSetLayout(impl.device, impl.setLayout, nullptr);
  if(impl.params.buffer)
    impl.context->destroyBuffer(impl.params);
  if(impl.linearSampler)
    vkDestroySampler(impl.device, impl.linearSampler, nullptr);
  if(impl.nearestSampler)
    vkDestroySampler(impl.device, impl.nearestSampler, nullptr);
  impl.kernels.reset();
  impl.model.reset();
  impl.context.reset();
  impl.device = VK_NULL_HANDLE;
  impl.commandPool = VK_NULL_HANDLE;
  impl.pool = VK_NULL_HANDLE;
  impl.preprocessPipeline = impl.compositePipeline = VK_NULL_HANDLE;
  impl.pipelineLayout = VK_NULL_HANDLE;
  impl.setLayout = VK_NULL_HANDLE;
  impl.linearSampler = impl.nearestSampler = VK_NULL_HANDLE;
}

#endif  // WITH_DLSS_NR
