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

// This file is included from gaussian_splatting.cpp - do not compile separately

namespace vk_gaussian_splatting {

void GaussianSplatting::initDescriptorSetPostProcessing()
{
  // Descriptor Bindings
  m_descriptorBindingsPostProcess.clear();
  m_descriptorBindingsPostProcess.addBinding(BINDING_FRAME_INFO_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  m_descriptorBindingsPostProcess.addBinding(POST_BINDING_MAIN_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  m_descriptorBindingsPostProcess.addBinding(POST_BINDING_AUX1_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  m_descriptorBindingsPostProcess.addBinding(POST_BINDING_DEPTH_TEXTURE, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  m_descriptorBindingsPostProcess.addBinding(POST_BINDING_DEPTH_SAMPLER, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  NVVK_CHECK(m_descriptorBindingsPostProcess.createDescriptorSetLayout(m_device, 0, &m_descriptorSetLayoutPostProcess));
  NVVK_DBG_NAME(m_descriptorSetLayoutPostProcess);

  // Descriptor Pool
  std::vector<VkDescriptorPoolSize> poolSize;
  m_descriptorBindingsPostProcess.appendPoolSizes(poolSize);
  VkDescriptorPoolCreateInfo poolInfo = {
      .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets       = 1,
      .poolSizeCount = uint32_t(poolSize.size()),
      .pPoolSizes    = poolSize.data(),
  };
  NVVK_CHECK(vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descriptorPoolPostProcess));
  NVVK_DBG_NAME(m_descriptorPoolPostProcess);

  // Descriptor Set
  VkDescriptorSetAllocateInfo allocInfo = {
      .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool     = m_descriptorPoolPostProcess,
      .descriptorSetCount = 1,
      .pSetLayouts        = &m_descriptorSetLayoutPostProcess,
  };
  NVVK_CHECK(vkAllocateDescriptorSets(m_device, &allocInfo, &m_descriptorSetPostProcess));
  NVVK_DBG_NAME(m_descriptorSetPostProcess);

  // Pipelne layout
  const VkPushConstantRange pcRanges = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT
                                            | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_COMPUTE_BIT,
                                        0, sizeof(shaderio::PushConstant)};

  VkPipelineLayoutCreateInfo plCreateInfo{
      .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount         = 1,
      .pSetLayouts            = &m_descriptorSetLayoutPostProcess,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges    = &pcRanges,
  };
  NVVK_CHECK(vkCreatePipelineLayout(m_device, &plCreateInfo, nullptr, &m_pipelineLayoutPostProcess));
  NVVK_DBG_NAME(m_pipelineLayoutPostProcess);

  // Writes
  nvvk::WriteSetContainer writeContainer;
  writeContainer.append(m_descriptorBindingsPostProcess.getWriteSet(BINDING_FRAME_INFO_UBO, m_descriptorSetPostProcess),
                        m_frameInfoBuffer);
  writeContainer.append(m_descriptorBindingsPostProcess.getWriteSet(POST_BINDING_MAIN_IMAGE, m_descriptorSetPostProcess),
                        m_gBuffers.getColorImageView(COLOR_MAIN), VK_IMAGE_LAYOUT_GENERAL);
  writeContainer.append(m_descriptorBindingsPostProcess.getWriteSet(POST_BINDING_AUX1_IMAGE, m_descriptorSetPostProcess),
                        m_gBuffers.getColorImageView(COLOR_AUX1), VK_IMAGE_LAYOUT_GENERAL);
  
  // Bind depth texture and sampler (if available)
  if(m_depthManager)
  {
    const auto& depthTexture = m_depthManager->getCurrentTexture();
    if(depthTexture.image.descriptor.imageView)
    {
      // Bind texture (SAMPLED_IMAGE - no sampler)
      writeContainer.append(m_descriptorBindingsPostProcess.getWriteSet(POST_BINDING_DEPTH_TEXTURE, m_descriptorSetPostProcess),
                            depthTexture.image.descriptor.imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_NULL_HANDLE);
      // Bind sampler separately (SAMPLER - no image)
      writeContainer.append(m_descriptorBindingsPostProcess.getWriteSet(POST_BINDING_DEPTH_SAMPLER, m_descriptorSetPostProcess),
                            VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, m_sampler);
    }
  }
  
  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writeContainer.size()), writeContainer.data(), 0, nullptr);
}

void GaussianSplatting::updateDescriptorSetPostProcessing()
{
  // update only if the descriptor set is already initialized
  if(m_descriptorSetPostProcess != VK_NULL_HANDLE)
  {
    nvvk::WriteSetContainer writeContainer;
    writeContainer.append(m_descriptorBindingsPostProcess.getWriteSet(POST_BINDING_MAIN_IMAGE, m_descriptorSetPostProcess),
                          m_gBuffers.getColorImageView(COLOR_MAIN), VK_IMAGE_LAYOUT_GENERAL);
    writeContainer.append(m_descriptorBindingsPostProcess.getWriteSet(POST_BINDING_AUX1_IMAGE, m_descriptorSetPostProcess),
                          m_gBuffers.getColorImageView(COLOR_AUX1), VK_IMAGE_LAYOUT_GENERAL);
    
    // Update depth texture and sampler bindings (if available)
    if(m_depthManager)
    {
      const auto& depthTexture = m_depthManager->getCurrentTexture();
      if(depthTexture.image.descriptor.imageView)
      {
        // Bind texture (SAMPLED_IMAGE - no sampler)
        writeContainer.append(m_descriptorBindingsPostProcess.getWriteSet(POST_BINDING_DEPTH_TEXTURE, m_descriptorSetPostProcess),
                              depthTexture.image.descriptor.imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_NULL_HANDLE);
        // Bind sampler separately (SAMPLER - no image)
        writeContainer.append(m_descriptorBindingsPostProcess.getWriteSet(POST_BINDING_DEPTH_SAMPLER, m_descriptorSetPostProcess),
                              VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, m_sampler);
      }
    }
    
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writeContainer.size()), writeContainer.data(), 0, nullptr);
  }
}

void GaussianSplatting::initPipelinePostProcessing()
{

  VkComputePipelineCreateInfo pipelineInfo{
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage =
          {
              .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
              .stage  = VK_SHADER_STAGE_COMPUTE_BIT,
              .module = m_shaders.postComputeShader,
              .pName  = "main",
          },
      .layout = m_pipelineLayoutPostProcess,
  };
  vkCreateComputePipelines(m_device, {}, 1, &pipelineInfo, nullptr, &m_computePipelinePostProcess);
  NVVK_DBG_NAME(m_computePipelinePostProcess);
}

void GaussianSplatting::postProcess(VkCommandBuffer cmd)
{
  NVVK_DBG_SCOPE(cmd);

  if(m_descriptorSetPostProcess == VK_NULL_HANDLE || m_pipelineLayoutPostProcess == VK_NULL_HANDLE 
     || m_computePipelinePostProcess == VK_NULL_HANDLE)
    return;

  auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, "Post process");

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_computePipelinePostProcess);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutPostProcess, 0, 1,
                          &m_descriptorSetPostProcess, 0, nullptr);

  uint32_t wgSize = 32;

  vkCmdDispatch(cmd, (uint32_t(m_viewSize.x) + wgSize - 1) / wgSize, (uint32_t(m_viewSize.y) + wgSize - 1) / wgSize, 1);
}

}  // namespace vk_gaussian_splatting
