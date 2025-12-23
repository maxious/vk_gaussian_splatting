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

// HDR10 encoding - converts linear BT.709 to BT.2020 + PQ
// This file is included from gaussian_splatting.cpp - do not compile separately

namespace vk_gaussian_splatting {

void GaussianSplatting::initDescriptorSetHdrEncode()
{
  // Descriptor Bindings
  m_descriptorBindingsHdrEncode.clear();
  m_descriptorBindingsHdrEncode.addBinding(BINDING_FRAME_INFO_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  m_descriptorBindingsHdrEncode.addBinding(HDR_BINDING_INPUT_IMAGE, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  m_descriptorBindingsHdrEncode.addBinding(HDR_BINDING_OUTPUT_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  NVVK_CHECK(m_descriptorBindingsHdrEncode.createDescriptorSetLayout(m_device, 0, &m_descriptorSetLayoutHdrEncode));
  NVVK_DBG_NAME(m_descriptorSetLayoutHdrEncode);

  // Descriptor Pool
  std::vector<VkDescriptorPoolSize> poolSize;
  m_descriptorBindingsHdrEncode.appendPoolSizes(poolSize);
  VkDescriptorPoolCreateInfo poolInfo = {
      .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets       = 1,
      .poolSizeCount = uint32_t(poolSize.size()),
      .pPoolSizes    = poolSize.data(),
  };
  NVVK_CHECK(vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descriptorPoolHdrEncode));
  NVVK_DBG_NAME(m_descriptorPoolHdrEncode);

  // Descriptor Set
  VkDescriptorSetAllocateInfo allocInfo = {
      .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool     = m_descriptorPoolHdrEncode,
      .descriptorSetCount = 1,
      .pSetLayouts        = &m_descriptorSetLayoutHdrEncode,
  };
  NVVK_CHECK(vkAllocateDescriptorSets(m_device, &allocInfo, &m_descriptorSetHdrEncode));
  NVVK_DBG_NAME(m_descriptorSetHdrEncode);

  // Pipeline layout
  VkPipelineLayoutCreateInfo plCreateInfo{
      .sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts    = &m_descriptorSetLayoutHdrEncode,
  };
  NVVK_CHECK(vkCreatePipelineLayout(m_device, &plCreateInfo, nullptr, &m_pipelineLayoutHdrEncode));
  NVVK_DBG_NAME(m_pipelineLayoutHdrEncode);

  // Writes - input is the main color buffer (linear HDR BT.709)
  // Output goes to the same buffer (in-place conversion for now)
  nvvk::WriteSetContainer writeContainer;
  writeContainer.append(m_descriptorBindingsHdrEncode.getWriteSet(BINDING_FRAME_INFO_UBO, m_descriptorSetHdrEncode),
                        m_frameInfoBuffer);
  writeContainer.append(m_descriptorBindingsHdrEncode.getWriteSet(HDR_BINDING_INPUT_IMAGE, m_descriptorSetHdrEncode),
                        m_gBuffers.getColorImageView(COLOR_MAIN), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  writeContainer.append(m_descriptorBindingsHdrEncode.getWriteSet(HDR_BINDING_OUTPUT_IMAGE, m_descriptorSetHdrEncode),
                        m_gBuffers.getColorImageView(COLOR_AUX1), VK_IMAGE_LAYOUT_GENERAL);
  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writeContainer.size()), writeContainer.data(), 0, nullptr);
}

void GaussianSplatting::updateDescriptorSetHdrEncode()
{
  if(m_descriptorSetHdrEncode != VK_NULL_HANDLE)
  {
    nvvk::WriteSetContainer writeContainer;
    writeContainer.append(m_descriptorBindingsHdrEncode.getWriteSet(HDR_BINDING_INPUT_IMAGE, m_descriptorSetHdrEncode),
                          m_gBuffers.getColorImageView(COLOR_MAIN), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    writeContainer.append(m_descriptorBindingsHdrEncode.getWriteSet(HDR_BINDING_OUTPUT_IMAGE, m_descriptorSetHdrEncode),
                          m_gBuffers.getColorImageView(COLOR_AUX1), VK_IMAGE_LAYOUT_GENERAL);
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writeContainer.size()), writeContainer.data(), 0, nullptr);
  }
}

void GaussianSplatting::initPipelineHdrEncode()
{
  VkComputePipelineCreateInfo pipelineInfo{
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage =
          {
              .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
              .stage  = VK_SHADER_STAGE_COMPUTE_BIT,
              .module = m_shaders.hdrEncodeComputeShader,
              .pName  = "main",
          },
      .layout = m_pipelineLayoutHdrEncode,
  };
  vkCreateComputePipelines(m_device, {}, 1, &pipelineInfo, nullptr, &m_computePipelineHdrEncode);
  NVVK_DBG_NAME(m_computePipelineHdrEncode);
}

void GaussianSplatting::hdrEncode(VkCommandBuffer cmd)
{
  NVVK_DBG_SCOPE(cmd);

  auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, "HDR10 encode");

  // Transition input image to shader read
  nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getColorImage(COLOR_MAIN),
                                    VK_IMAGE_ASPECT_COLOR_BIT,
                                    VK_ACCESS_SHADER_WRITE_BIT,
                                    VK_ACCESS_SHADER_READ_BIT,
                                    VK_IMAGE_LAYOUT_GENERAL,
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT});

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_computePipelineHdrEncode);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutHdrEncode, 0, 1,
                          &m_descriptorSetHdrEncode, 0, nullptr);

  uint32_t wgSize = 16;
  vkCmdDispatch(cmd, (uint32_t(m_viewSize.x) + wgSize - 1) / wgSize, (uint32_t(m_viewSize.y) + wgSize - 1) / wgSize, 1);

  // Transition output for reading and input back to general
  nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getColorImage(COLOR_MAIN),
                                    VK_IMAGE_ASPECT_COLOR_BIT,
                                    VK_ACCESS_SHADER_READ_BIT,
                                    VK_ACCESS_SHADER_WRITE_BIT,
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                    VK_IMAGE_LAYOUT_GENERAL,
                                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT});

  nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getColorImage(COLOR_AUX1),
                                    VK_IMAGE_ASPECT_COLOR_BIT,
                                    VK_ACCESS_SHADER_WRITE_BIT,
                                    VK_ACCESS_TRANSFER_READ_BIT,
                                    VK_IMAGE_LAYOUT_GENERAL,
                                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                    VK_PIPELINE_STAGE_TRANSFER_BIT});
}

}  // namespace vk_gaussian_splatting
