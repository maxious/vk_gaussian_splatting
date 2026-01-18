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

#include <nvutils/alignment.hpp> // Ensure this is included

namespace vk_viewer {

void VkViewer::initPipelines()
{

  nvvk::DescriptorBindings bindings;

  bindings.addBinding(BINDING_FRAME_INFO_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_ALL);
  bindings.addBinding(BINDING_DISTANCES_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
  bindings.addBinding(BINDING_INDICES_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
  bindings.addBinding(BINDING_INDIRECT_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_ALL);

  if(prmData.dataStorage == STORAGE_TEXTURES)

  {
    bindings.addBinding(BINDING_CENTERS_TEXTURE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(BINDING_SCALES_TEXTURE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(BINDING_ROTATIONS_TEXTURE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(BINDING_COVARIANCES_TEXTURE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);

    bindings.addBinding(BINDING_COLORS_TEXTURE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(BINDING_SH_TEXTURE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(BINDING_MOTION_TEXTURE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(BINDING_TIME_TEXTURE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);
  }
  else
  {
    bindings.addBinding(BINDING_CENTERS_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(BINDING_SCALES_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(BINDING_ROTATIONS_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(BINDING_COVARIANCES_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);

    bindings.addBinding(BINDING_COLORS_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(BINDING_SH_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(BINDING_MOTION_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(BINDING_TIME_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    
    // LCC packed storage buffer (for GPU-side decompression)
    bindings.addBinding(BINDING_LCC_PACKED_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
  }

  // Obj Mesh objectDescriptions
  bindings.addBinding(BINDING_MESH_DESCRIPTORS, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
  bindings.addBinding(BINDING_LIGHT_SET, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
  // Mesh textures (array of 16 textures)
  bindings.addBinding(BINDING_MESH_TEXTURES, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 16, VK_SHADER_STAGE_FRAGMENT_BIT);

  // VDZ depth mesh textures
  bindings.addBinding(BINDING_VDZ_VIDEO_TEXTURE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT);
  bindings.addBinding(BINDING_VDZ_DEPTH_TEXTURE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
  // Environment Depth Occlusion
  bindings.addBinding(BINDING_ENV_DEPTH_TEXTURE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT);
  // Hand mesh joint matrices (XR skinned hands)
  bindings.addBinding(BINDING_JOINT_MATRICES, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT);
  
  // Chunk-based hierarchical frustum culling buffers
  bindings.addBinding(BINDING_CHUNK_BOUNDS_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  bindings.addBinding(BINDING_VISIBLE_CHUNKS_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  bindings.addBinding(BINDING_VISIBLE_CHUNK_COUNT_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);

  //
  const VkPushConstantRange pcRanges = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT
                                            | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_COMPUTE_BIT,
                                        0, sizeof(shaderio::PushConstant)};

  NVVK_CHECK(bindings.createDescriptorSetLayout(m_device, 0, &m_descriptorSetLayout));
  NVVK_DBG_NAME(m_descriptorSetLayout);

  //
  std::vector<VkDescriptorPoolSize> poolSize;
  bindings.appendPoolSizes(poolSize);
  VkDescriptorPoolCreateInfo poolInfo = {
      .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets       = 1,
      .poolSizeCount = uint32_t(poolSize.size()),
      .pPoolSizes    = poolSize.data(),
  };
  NVVK_CHECK(vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descriptorPool));
  NVVK_DBG_NAME(m_descriptorPool);

  VkDescriptorSetAllocateInfo allocInfo = {
      .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool     = m_descriptorPool,
      .descriptorSetCount = 1,
      .pSetLayouts        = &m_descriptorSetLayout,
  };
  NVVK_CHECK(vkAllocateDescriptorSets(m_device, &allocInfo, &m_descriptorSet));
  NVVK_DBG_NAME(m_descriptorSet);

  VkPipelineLayoutCreateInfo plCreateInfo{
      .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount         = 1,
      .pSetLayouts            = &m_descriptorSetLayout,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges    = &pcRanges,
  };
  NVVK_CHECK(vkCreatePipelineLayout(m_device, &plCreateInfo, nullptr, &m_pipelineLayout));
  NVVK_DBG_NAME(m_pipelineLayout);

  // Write descriptors for the buffers and textures
  nvvk::WriteSetContainer writeContainer;

  // add common buffers
  // Dynamic offset for Frame Info UBO
  VkDescriptorBufferInfo frameInfoDesc{m_frameInfoBuffer.buffer, 0, sizeof(shaderio::FrameInfo)};
  VkWriteDescriptorSet   frameInfoWrite = bindings.getWriteSet(BINDING_FRAME_INFO_UBO, m_descriptorSet);
  frameInfoWrite.pBufferInfo            = &frameInfoDesc;
  // Use direct update to handle dynamic buffer properly
  vkUpdateDescriptorSets(m_device, 1, &frameInfoWrite, 0, nullptr);

  writeContainer.append(bindings.getWriteSet(BINDING_DISTANCES_BUFFER, m_descriptorSet), m_splatDistancesDevice);

  writeContainer.append(bindings.getWriteSet(BINDING_INDICES_BUFFER, m_descriptorSet), m_splatIndicesDevice);
  
  // Dynamic offset for indirect buffer (range is size of one struct)
  VkDescriptorBufferInfo indirectInfo{m_indirect.buffer, 0, sizeof(shaderio::IndirectParams)};
  VkWriteDescriptorSet   indirectWrite = bindings.getWriteSet(BINDING_INDIRECT_BUFFER, m_descriptorSet);
  indirectWrite.pBufferInfo            = &indirectInfo;
  
  // Directly update descriptor set to avoid WriteSetContainer append issue with custom write
  vkUpdateDescriptorSets(m_device, 1, &indirectWrite, 0, nullptr);
  
  // Only add splat data descriptors if splats are loaded
  if(m_splatSet.size() > 0)
  {
    if(prmData.dataStorage == STORAGE_TEXTURES)
    {
      // add data texture maps
      writeContainer.append(bindings.getWriteSet(BINDING_CENTERS_TEXTURE, m_descriptorSet), m_splatSetVk.centersMap);
      writeContainer.append(bindings.getWriteSet(BINDING_SCALES_TEXTURE, m_descriptorSet), m_splatSetVk.scalesMap);
      writeContainer.append(bindings.getWriteSet(BINDING_ROTATIONS_TEXTURE, m_descriptorSet), m_splatSetVk.rotationsMap);
      writeContainer.append(bindings.getWriteSet(BINDING_COVARIANCES_TEXTURE, m_descriptorSet), m_splatSetVk.covariancesMap);

      writeContainer.append(bindings.getWriteSet(BINDING_COLORS_TEXTURE, m_descriptorSet), m_splatSetVk.colorsMap);
      writeContainer.append(bindings.getWriteSet(BINDING_SH_TEXTURE, m_descriptorSet), m_splatSetVk.sphericalHarmonicsMap);
      writeContainer.append(bindings.getWriteSet(BINDING_MOTION_TEXTURE, m_descriptorSet), m_splatSetVk.motionMap);
      writeContainer.append(bindings.getWriteSet(BINDING_TIME_TEXTURE, m_descriptorSet), m_splatSetVk.timeMap);
    }
    else if(prmData.dataStorage == STORAGE_LCC_PACKED)
    {
      // LCC packed storage: single buffer with raw 32-byte packed splat data
      if(m_splatSetVk.lccPackedBuffer.buffer != VK_NULL_HANDLE)
      {
        writeContainer.append(bindings.getWriteSet(BINDING_LCC_PACKED_BUFFER, m_descriptorSet), m_splatSetVk.lccPackedBuffer);
      }
    }
    else
    {
      // add data buffers (STORAGE_BUFFERS mode)
      writeContainer.append(bindings.getWriteSet(BINDING_CENTERS_BUFFER, m_descriptorSet), m_splatSetVk.centersBuffer);
      writeContainer.append(bindings.getWriteSet(BINDING_SCALES_BUFFER, m_descriptorSet), m_splatSetVk.scalesBuffer);
      writeContainer.append(bindings.getWriteSet(BINDING_ROTATIONS_BUFFER, m_descriptorSet), m_splatSetVk.rotationsBuffer);
      writeContainer.append(bindings.getWriteSet(BINDING_COVARIANCES_BUFFER, m_descriptorSet), m_splatSetVk.covariancesBuffer);

      writeContainer.append(bindings.getWriteSet(BINDING_COLORS_BUFFER, m_descriptorSet), m_splatSetVk.colorsBuffer);
      if(m_splatSetVk.sphericalHarmonicsBuffer.buffer != NULL)
        writeContainer.append(bindings.getWriteSet(BINDING_SH_BUFFER, m_descriptorSet), m_splatSetVk.sphericalHarmonicsBuffer);
        
      writeContainer.append(bindings.getWriteSet(BINDING_MOTION_BUFFER, m_descriptorSet), m_splatSetVk.motionBuffer);
      writeContainer.append(bindings.getWriteSet(BINDING_TIME_BUFFER, m_descriptorSet), m_splatSetVk.timeBuffer);
    }
  }

  if(m_meshSetVk.instances.size())
  {
    writeContainer.append(bindings.getWriteSet(BINDING_MESH_DESCRIPTORS, m_descriptorSet),
                          m_meshSetVk.objectDescriptionsBuffer.buffer);
  }

  if(m_lightSet.size())
  {
    writeContainer.append(bindings.getWriteSet(BINDING_LIGHT_SET, m_descriptorSet), m_lightSet.lightsBuffer);
  }

  // Initialize with dummy texture to ensure valid bindings
  VkDescriptorImageInfo dummyInfo = {
      .sampler     = m_sampler,
      .imageView   = m_dummyTextureArray.view,
      .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

  // Environment Depth Occlusion
  writeContainer.append(bindings.getWriteSet(BINDING_ENV_DEPTH_TEXTURE, m_descriptorSet), &dummyInfo);

  // VDZ depth mesh textures
  // Initialize with dummy first to prevent validation errors
  writeContainer.append(bindings.getWriteSet(BINDING_VDZ_DEPTH_TEXTURE, m_descriptorSet), &dummyInfo);
  writeContainer.append(bindings.getWriteSet(BINDING_VDZ_VIDEO_TEXTURE, m_descriptorSet), &dummyInfo);

  if(m_depthManager)
  {
    const auto& depthTexture = m_depthManager->getCurrentTexture();
    if(depthTexture.image.descriptor.imageView)
    {
      writeContainer.append(bindings.getWriteSet(BINDING_VDZ_DEPTH_TEXTURE, m_descriptorSet),
                            depthTexture.image.descriptor.imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, m_sampler);
      // Currently using depth texture as video texture placeholder
      writeContainer.append(bindings.getWriteSet(BINDING_VDZ_VIDEO_TEXTURE, m_descriptorSet),
                            depthTexture.image.descriptor.imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, m_sampler);
    }
  }

  // Mesh textures - initialize all 16 slots with dummy textures first
  std::vector<VkDescriptorImageInfo> meshTexInfos(16, dummyInfo);
  
  // Fill in actual mesh textures if available
  if(!m_meshSetVk.meshes.empty())
  {
    for(const auto& mesh : m_meshSetVk.meshes)
    {
      for(size_t i = 0; i < mesh.textures.size() && i < 16; ++i)
      {
        const auto& tex = mesh.textures[i];
        if(tex.view && tex.sampler)
        {
          meshTexInfos[i] = {
              .sampler     = tex.sampler,
              .imageView   = tex.view,
              .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
          };
        }
      }
    }
  }
  
  VkWriteDescriptorSet meshTexWrite = bindings.getWriteSet(BINDING_MESH_TEXTURES, m_descriptorSet);
  meshTexWrite.descriptorCount = 16;
  meshTexWrite.pImageInfo = meshTexInfos.data();
  vkUpdateDescriptorSets(m_device, 1, &meshTexWrite, 0, nullptr);

  // Chunk culling buffers (bind even if empty to avoid validation errors)
  if(m_chunkBoundsBuffer.buffer != VK_NULL_HANDLE)
  {
    writeContainer.append(bindings.getWriteSet(BINDING_CHUNK_BOUNDS_BUFFER, m_descriptorSet), m_chunkBoundsBuffer);
    writeContainer.append(bindings.getWriteSet(BINDING_VISIBLE_CHUNKS_BUFFER, m_descriptorSet), m_visibleChunksBuffer);
    writeContainer.append(bindings.getWriteSet(BINDING_VISIBLE_CHUNK_COUNT_BUFFER, m_descriptorSet), m_visibleChunkCountBuffer);
  }

  // write
  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writeContainer.size()), writeContainer.data(), 0, nullptr);

  // Create the pipeline to run the compute shader for distance & culling
  {
    VkComputePipelineCreateInfo pipelineInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage =
            {
                .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage  = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = m_shaders.distShader,
                .pName  = "main",
            },
        .layout = m_pipelineLayout,
    };
    vkCreateComputePipelines(m_device, {}, 1, &pipelineInfo, nullptr, &m_computePipelineGsDistCull);
    NVVK_DBG_NAME(m_computePipelineGsDistCull);
  }
  // Create the GS rasterization pipelines
  {
    // Preparing the common states
    nvvk::GraphicsPipelineState pipelineState;
    pipelineState.rasterizationState.cullMode = VK_CULL_MODE_NONE;

    // activates blending and set blend func
    pipelineState.colorBlendEnables[0]                       = VK_TRUE;
    pipelineState.colorBlendEquations[0].alphaBlendOp        = VK_BLEND_OP_ADD;
    pipelineState.colorBlendEquations[0].colorBlendOp        = VK_BLEND_OP_ADD;
    pipelineState.colorBlendEquations[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;  //VK_BLEND_FACTOR_SRC_ALPHA;
    pipelineState.colorBlendEquations[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;  //VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    pipelineState.colorBlendEquations[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    pipelineState.colorBlendEquations[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;

    // By default disable depth write and test for the pipeline
    // Since splats are sorted, screen aligned, and rendered back to front
    // we do not need depth test/write, which leads to faster rendering
    // however since CPU sorting mode is costly we disable it when not visualizing with alpha,
    // only in this case we will use depth test/write. this will be changed dynamically at rendering.
    pipelineState.rasterizationState.cullMode        = VK_CULL_MODE_NONE;
    pipelineState.depthStencilState.depthWriteEnable = VK_FALSE;
    pipelineState.depthStencilState.depthTestEnable  = VK_FALSE;

    // create the pipeline that uses mesh shaders for 3DGS
    {
      nvvk::GraphicsPipelineCreator creator;
      creator.pipelineInfo.layout                  = m_pipelineLayout;
      creator.colorFormats                         = {m_colorFormat};
      creator.renderingState.depthAttachmentFormat = m_depthFormat;
      // The dynamic state is used to change the depth test state dynamically
      creator.dynamicStateValues.push_back(VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE);
      creator.dynamicStateValues.push_back(VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE);

      creator.addShader(VK_SHADER_STAGE_MESH_BIT_EXT, "main", m_shaders.meshShader);
      creator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main_mesh", m_shaders.fragmentShader);

      creator.createGraphicsPipeline(m_device, nullptr, pipelineState, &m_graphicsPipelineGsMesh);
      NVVK_DBG_NAME(m_graphicsPipelineGsMesh);
    }

    // create the pipeline that uses mesh shaders for 3DGUT
    {
      nvvk::GraphicsPipelineCreator creator;
      creator.pipelineInfo.layout                  = m_pipelineLayout;
      creator.colorFormats                         = {m_colorFormat};
      creator.renderingState.depthAttachmentFormat = m_depthFormat;
      // The dynamic state is used to change the depth test state dynamically
      creator.dynamicStateValues.push_back(VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE);
      creator.dynamicStateValues.push_back(VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE);

      creator.addShader(VK_SHADER_STAGE_MESH_BIT_EXT, "main", m_shaders.threedgutMeshShader);
      creator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main", m_shaders.threedgutFragmentShader);

      creator.createGraphicsPipeline(m_device, nullptr, pipelineState, &m_graphicsPipeline3dgutMesh);
      NVVK_DBG_NAME(m_graphicsPipeline3dgutMesh);
    }

    // create the pipeline that uses vertex shaders for 3DGS
    {
      const auto BINDING_ATTR_POSITION    = 0;
      const auto BINDING_ATTR_SPLAT_INDEX = 1;

      pipelineState.vertexBindings   = {{// 3 component per vertex position
                                         .binding = BINDING_ATTR_POSITION,
                                         .stride  = 3 * sizeof(float),
                                       //.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
                                         .divisor = 1},
                                        {// All the vertices of each splat instance will get the same index
                                         .binding   = BINDING_ATTR_SPLAT_INDEX,
                                         .stride    = sizeof(uint32_t),
                                         .inputRate = VK_VERTEX_INPUT_RATE_INSTANCE,
                                         .divisor   = 1}};
      pipelineState.vertexAttributes = {
          {.location = ATTRIBUTE_LOC_POSITION, .binding = BINDING_ATTR_POSITION, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = 0},
          {.location = ATTRIBUTE_LOC_SPLAT_INDEX, .binding = BINDING_ATTR_SPLAT_INDEX, .format = VK_FORMAT_R32_UINT, .offset = 0}};

      nvvk::GraphicsPipelineCreator creator;
      creator.pipelineInfo.layout                  = m_pipelineLayout;
      creator.colorFormats                         = {m_colorFormat};
      creator.renderingState.depthAttachmentFormat = m_depthFormat;
      // The dynamic state is used to change the depth test state dynamically
      creator.dynamicStateValues.push_back(VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE);
      creator.dynamicStateValues.push_back(VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE);

      creator.addShader(VK_SHADER_STAGE_VERTEX_BIT, "main", m_shaders.vertexShader);
      creator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main", m_shaders.fragmentShader);

      creator.createGraphicsPipeline(m_device, nullptr, pipelineState, &m_graphicsPipelineGsVert);
      NVVK_DBG_NAME(m_graphicsPipelineGsVert);
    }

#ifdef WITH_OPENXR
    // Create multiview variants with viewMask = 0x3 (both eyes)
    {
      nvvk::GraphicsPipelineCreator creator;
      creator.pipelineInfo.layout                  = m_pipelineLayout;
      creator.colorFormats                         = {m_colorFormat};
      creator.renderingState.depthAttachmentFormat = m_depthFormat;
      creator.renderingState.viewMask              = 0x3;  // Render to both views
      creator.dynamicStateValues.push_back(VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE);
      creator.dynamicStateValues.push_back(VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE);

      creator.addShader(VK_SHADER_STAGE_MESH_BIT_EXT, "main", m_shaders.meshShader);
      creator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main_mesh", m_shaders.fragmentShader);

      creator.createGraphicsPipeline(m_device, nullptr, pipelineState, &m_graphicsPipelineGsMeshMultiview);
      NVVK_DBG_NAME(m_graphicsPipelineGsMeshMultiview);
    }

    {
      nvvk::GraphicsPipelineCreator creator;
      creator.pipelineInfo.layout                  = m_pipelineLayout;
      creator.colorFormats                         = {m_colorFormat};
      creator.renderingState.depthAttachmentFormat = m_depthFormat;
      creator.renderingState.viewMask              = 0x3;
      creator.dynamicStateValues.push_back(VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE);
      creator.dynamicStateValues.push_back(VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE);

      creator.addShader(VK_SHADER_STAGE_MESH_BIT_EXT, "main", m_shaders.threedgutMeshShader);
      creator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main", m_shaders.threedgutFragmentShader);

      creator.createGraphicsPipeline(m_device, nullptr, pipelineState, &m_graphicsPipeline3dgutMeshMultiview);
      NVVK_DBG_NAME(m_graphicsPipeline3dgutMeshMultiview);
    }

    {
      nvvk::GraphicsPipelineCreator creator;
      creator.pipelineInfo.layout                  = m_pipelineLayout;
      creator.colorFormats                         = {m_colorFormat};
      creator.renderingState.depthAttachmentFormat = m_depthFormat;
      creator.renderingState.viewMask              = 0x3;
      creator.dynamicStateValues.push_back(VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE);
      creator.dynamicStateValues.push_back(VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE);

      pipelineState.vertexBindings   = {{.binding = 0, .stride = 3 * sizeof(float), .divisor = 1},
                                        {.binding = 1, .stride = sizeof(uint32_t), .inputRate = VK_VERTEX_INPUT_RATE_INSTANCE, .divisor = 1}};
      pipelineState.vertexAttributes = {
          {.location = ATTRIBUTE_LOC_POSITION, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = 0},
          {.location = ATTRIBUTE_LOC_SPLAT_INDEX, .binding = 1, .format = VK_FORMAT_R32_UINT, .offset = 0}};

      creator.addShader(VK_SHADER_STAGE_VERTEX_BIT, "main", m_shaders.vertexShader);
      creator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main", m_shaders.fragmentShader);

      creator.createGraphicsPipeline(m_device, nullptr, pipelineState, &m_graphicsPipelineGsVertMultiview);
      NVVK_DBG_NAME(m_graphicsPipelineGsVertMultiview);
    }
#endif
  }
  // Create the 3D mesh rasterization pipeline
  {

    // Preparing the pipeline states
    nvvk::GraphicsPipelineState pipelineState;
    pipelineState.rasterizationState.cullMode = VK_CULL_MODE_NONE;

    // deactivates blending and set blend func
    pipelineState.colorBlendEnables[0]                       = VK_FALSE;
    pipelineState.colorBlendEquations[0].alphaBlendOp        = VK_BLEND_OP_ADD;
    pipelineState.colorBlendEquations[0].colorBlendOp        = VK_BLEND_OP_ADD;
    pipelineState.colorBlendEquations[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    pipelineState.colorBlendEquations[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    pipelineState.colorBlendEquations[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    pipelineState.colorBlendEquations[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;

    // TODOC
    pipelineState.rasterizationState.cullMode        = VK_CULL_MODE_NONE;
    pipelineState.depthStencilState.depthWriteEnable = VK_TRUE;
    pipelineState.depthStencilState.depthTestEnable  = VK_TRUE;

    // create the pipeline
    const auto BINDING_ATTR_VERTEX = 0;

    pipelineState.vertexBindings   = {{// 3 pos + 3 nrm + 2 texCoord per vertex
                                       .binding = BINDING_ATTR_VERTEX,
                                       .stride  = sizeof(ObjVertex),
                                       .divisor = 1}};
    pipelineState.vertexAttributes = {{.location = ATTRIBUTE_LOC_MESH_POSITION,
                                       .binding  = BINDING_ATTR_VERTEX,
                                       .format   = VK_FORMAT_R32G32B32_SFLOAT,
                                       .offset   = static_cast<uint32_t>(offsetof(ObjVertex, pos))},
                                      {.location = ATTRIBUTE_LOC_MESH_NORMAL,
                                       .binding  = BINDING_ATTR_VERTEX,
                                       .format   = VK_FORMAT_R32G32B32_SFLOAT,
                                       .offset   = static_cast<uint32_t>(offsetof(ObjVertex, nrm))},
                                      {.location = ATTRIBUTE_LOC_MESH_TEXCOORD,
                                       .binding  = BINDING_ATTR_VERTEX,
                                       .format   = VK_FORMAT_R32G32_SFLOAT,
                                       .offset   = static_cast<uint32_t>(offsetof(ObjVertex, texCoord))}};

    nvvk::GraphicsPipelineCreator creator;
    creator.pipelineInfo.layout                  = m_pipelineLayout;
    creator.colorFormats                         = {m_colorFormat};
    creator.renderingState.depthAttachmentFormat = m_depthFormat;
    // The dynamic state is used to change the depth test state dynamically
    creator.dynamicStateValues.push_back(VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE);
    creator.dynamicStateValues.push_back(VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE);

    creator.addShader(VK_SHADER_STAGE_VERTEX_BIT, "main", m_shaders.meshVertexShader);
    creator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main", m_shaders.meshFragmentShader);

    creator.createGraphicsPipeline(m_device, nullptr, pipelineState, &m_graphicsPipelineMesh);
    NVVK_DBG_NAME(m_graphicsPipelineMesh);
  }

  // Create the VDZ depth mesh rasterization pipeline
  {
    nvvk::GraphicsPipelineState pipelineState;
    pipelineState.rasterizationState.cullMode = VK_CULL_MODE_NONE;

    // No blending for VDZ mesh
    pipelineState.colorBlendEnables[0] = VK_FALSE;

    // No depth testing for camera-attached mesh
    pipelineState.depthStencilState.depthWriteEnable = VK_FALSE;
    pipelineState.depthStencilState.depthTestEnable  = VK_FALSE;

    // VDZ mesh vertex layout: position (vec3) + uv (vec2)
    pipelineState.vertexBindings   = {{.binding = 0,
                                       .stride  = sizeof(float) * 3 + sizeof(float) * 2,
                                       .inputRate = VK_VERTEX_INPUT_RATE_VERTEX}};
                                       
    pipelineState.vertexAttributes = {{.location = 0,
                                       .binding  = 0,
                                       .format   = VK_FORMAT_R32G32B32_SFLOAT,
                                       .offset   = 0},
                                      {.location = 1,
                                       .binding  = 0,
                                       .format   = VK_FORMAT_R32G32_SFLOAT,
                                       .offset   = sizeof(float) * 3}};

    nvvk::GraphicsPipelineCreator creator;
    creator.pipelineInfo.layout                  = m_pipelineLayout;
    creator.colorFormats                         = {m_colorFormat};
    creator.renderingState.depthAttachmentFormat = m_depthFormat;

    creator.addShader(VK_SHADER_STAGE_VERTEX_BIT, "main", m_shaders.vdzMeshVertexShader);
    creator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main", m_shaders.vdzMeshFragmentShader);

    creator.createGraphicsPipeline(m_device, nullptr, pipelineState, &m_graphicsPipelineVdzMesh);
    NVVK_DBG_NAME(m_graphicsPipelineVdzMesh);
  }

  // Create the VDZ hybrid rasterization pipeline (mesh + POM)
  {
    nvvk::GraphicsPipelineState pipelineState;
    pipelineState.rasterizationState.cullMode = VK_CULL_MODE_NONE;

    // No blending for VDZ hybrid
    pipelineState.colorBlendEnables[0] = VK_FALSE;

    // No depth testing for camera-attached mesh
    pipelineState.depthStencilState.depthWriteEnable = VK_FALSE;
    pipelineState.depthStencilState.depthTestEnable  = VK_FALSE;

    // VDZ hybrid mesh vertex layout: position (vec3) + uv (vec2)
    pipelineState.vertexBindings   = {{.binding = 0,
                                       .stride  = sizeof(float) * 3 + sizeof(float) * 2,
                                       .inputRate = VK_VERTEX_INPUT_RATE_VERTEX}};
                                       
    pipelineState.vertexAttributes = {{.location = 0,
                                       .binding  = 0,
                                       .format   = VK_FORMAT_R32G32B32_SFLOAT,
                                       .offset   = 0},
                                      {.location = 1,
                                       .binding  = 0,
                                       .format   = VK_FORMAT_R32G32_SFLOAT,
                                       .offset   = sizeof(float) * 3}};

    nvvk::GraphicsPipelineCreator creator;
    creator.pipelineInfo.layout                  = m_pipelineLayout;
    creator.colorFormats                         = {m_colorFormat};
    creator.renderingState.depthAttachmentFormat = m_depthFormat;

    creator.addShader(VK_SHADER_STAGE_VERTEX_BIT, "main", m_shaders.vdzHybridVertexShader);
    creator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main", m_shaders.vdzHybridFragmentShader);

    creator.createGraphicsPipeline(m_device, nullptr, pipelineState, &m_graphicsPipelineVdzHybrid);
    NVVK_DBG_NAME(m_graphicsPipelineVdzHybrid);
  }

  // Create the hand mesh pipeline (XR skinned hands)
  {
    nvvk::GraphicsPipelineState pipelineState;

    // Alpha blending for translucent hands
    pipelineState.colorBlendEnables[0]                        = VK_TRUE;
    pipelineState.colorBlendEquations[0].colorBlendOp        = VK_BLEND_OP_ADD;
    pipelineState.colorBlendEquations[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    pipelineState.colorBlendEquations[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    pipelineState.colorBlendEquations[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    pipelineState.colorBlendEquations[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;

    pipelineState.rasterizationState.cullMode        = VK_CULL_MODE_NONE;
    pipelineState.depthStencilState.depthWriteEnable = VK_TRUE;
    pipelineState.depthStencilState.depthTestEnable  = VK_TRUE;

    // Hand mesh vertex layout: position(vec3) + normal(vec3) + uv(vec2) + blendIndices(ivec4) + blendWeights(vec4)
    pipelineState.vertexBindings = {{.binding   = 0,
                                     .stride    = sizeof(float) * 3 + sizeof(float) * 3 + sizeof(float) * 2 +
                                                  sizeof(int32_t) * 4 + sizeof(float) * 4,
                                     .inputRate = VK_VERTEX_INPUT_RATE_VERTEX}};

    pipelineState.vertexAttributes = {
        {.location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = 0},                          // position
        {.location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = sizeof(float) * 3},          // normal
        {.location = 2, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT, .offset = sizeof(float) * 6},             // uv
        {.location = 3, .binding = 0, .format = VK_FORMAT_R32G32B32A32_SINT, .offset = sizeof(float) * 8},         // blendIndices
        {.location = 4, .binding = 0, .format = VK_FORMAT_R32G32B32A32_SFLOAT, .offset = sizeof(float) * 8 + sizeof(int32_t) * 4}  // blendWeights
    };

    nvvk::GraphicsPipelineCreator creator;
    creator.pipelineInfo.layout                  = m_pipelineLayout;
    creator.colorFormats                         = {m_colorFormat};
    creator.renderingState.depthAttachmentFormat = m_depthFormat;
    creator.dynamicStateValues.push_back(VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE);
    creator.dynamicStateValues.push_back(VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE);

    creator.addShader(VK_SHADER_STAGE_VERTEX_BIT, "main", m_shaders.handMeshVertexShader);
    creator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main", m_shaders.handMeshFragmentShader);

    creator.createGraphicsPipeline(m_device, nullptr, pipelineState, &m_graphicsPipelineHandMesh);
    NVVK_DBG_NAME(m_graphicsPipelineHandMesh);

#ifdef WITH_OPENXR
    // Create multiview variant for XR stereo rendering
    {
      nvvk::GraphicsPipelineCreator multiviewCreator;
      multiviewCreator.pipelineInfo.layout                  = m_pipelineLayout;
      multiviewCreator.colorFormats                         = {m_colorFormat};
      multiviewCreator.renderingState.depthAttachmentFormat = m_depthFormat;
      multiviewCreator.renderingState.viewMask              = 0x3;  // Render to both views
      multiviewCreator.dynamicStateValues.push_back(VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE);
      multiviewCreator.dynamicStateValues.push_back(VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE);

      multiviewCreator.addShader(VK_SHADER_STAGE_VERTEX_BIT, "main", m_shaders.handMeshVertexShader);
      multiviewCreator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main", m_shaders.handMeshFragmentShader);

      multiviewCreator.createGraphicsPipeline(m_device, nullptr, pipelineState, &m_graphicsPipelineHandMeshMultiview);
      NVVK_DBG_NAME(m_graphicsPipelineHandMeshMultiview);
    }
#endif
  }
  
  // Initialize chunk culling pipeline
  initChunkCullingPipeline();
}

// include RTX one
void VkViewer::deinitPipelines()
{
  if(m_graphicsPipelineGsVert == VK_NULL_HANDLE)
    return;

  TEST_DESTROY_AND_RESET(m_graphicsPipelineGsVert, vkDestroyPipeline(m_device, m_graphicsPipelineGsVert, nullptr));
  TEST_DESTROY_AND_RESET(m_graphicsPipelineGsMesh, vkDestroyPipeline(m_device, m_graphicsPipelineGsMesh, nullptr));
  TEST_DESTROY_AND_RESET(m_graphicsPipeline3dgutMesh, vkDestroyPipeline(m_device, m_graphicsPipeline3dgutMesh, nullptr));
  TEST_DESTROY_AND_RESET(m_graphicsPipelineMesh, vkDestroyPipeline(m_device, m_graphicsPipelineMesh, nullptr));
  TEST_DESTROY_AND_RESET(m_graphicsPipelineVdzMesh, vkDestroyPipeline(m_device, m_graphicsPipelineVdzMesh, nullptr));
  TEST_DESTROY_AND_RESET(m_graphicsPipelineVdzHybrid, vkDestroyPipeline(m_device, m_graphicsPipelineVdzHybrid, nullptr));
  TEST_DESTROY_AND_RESET(m_graphicsPipelineHandMesh, vkDestroyPipeline(m_device, m_graphicsPipelineHandMesh, nullptr));
  TEST_DESTROY_AND_RESET(m_computePipelineGsDistCull, vkDestroyPipeline(m_device, m_computePipelineGsDistCull, nullptr));
  
  // Chunk culling pipeline cleanup
  deinitChunkCullingPipeline();
#ifdef WITH_OPENXR
  TEST_DESTROY_AND_RESET(m_graphicsPipelineGsVertMultiview, vkDestroyPipeline(m_device, m_graphicsPipelineGsVertMultiview, nullptr));
  TEST_DESTROY_AND_RESET(m_graphicsPipelineGsMeshMultiview, vkDestroyPipeline(m_device, m_graphicsPipelineGsMeshMultiview, nullptr));
  TEST_DESTROY_AND_RESET(m_graphicsPipeline3dgutMeshMultiview, vkDestroyPipeline(m_device, m_graphicsPipeline3dgutMeshMultiview, nullptr));
  TEST_DESTROY_AND_RESET(m_graphicsPipelineHandMeshMultiview, vkDestroyPipeline(m_device, m_graphicsPipelineHandMeshMultiview, nullptr));
#endif

  TEST_DESTROY_AND_RESET(m_pipelineLayout, vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr));
  TEST_DESTROY_AND_RESET(m_descriptorSetLayout, vkDestroyDescriptorSetLayout(m_device, m_descriptorSetLayout, nullptr));
  TEST_DESTROY_AND_RESET(m_descriptorPool, vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr));
  m_descriptorSet = VK_NULL_HANDLE;

  // RTX TODO move this in rtDeinitPipeline and invoke in proper location
  TEST_DESTROY_AND_RESET(m_rtPipeline, vkDestroyPipeline(m_device, m_rtPipeline, nullptr));

  TEST_DESTROY_AND_RESET(m_rtPipelineLayout, vkDestroyPipelineLayout(m_device, m_rtPipelineLayout, nullptr));
  TEST_DESTROY_AND_RESET(m_rtDescriptorPool, vkDestroyDescriptorPool(m_device, m_rtDescriptorPool, nullptr));
  m_rtDescriptorSet = VK_NULL_HANDLE;

  TEST_DESTROY_AND_RESET(m_rtDescriptorSetLayout, vkDestroyDescriptorSetLayout(m_device, m_rtDescriptorSetLayout, nullptr));

  m_alloc.destroyBuffer(m_rtSBTBuffer);
  m_rtShaderGroups.clear();

  // Post process
  TEST_DESTROY_AND_RESET(m_computePipelinePostProcess, vkDestroyPipeline(m_device, m_computePipelinePostProcess, nullptr));

  TEST_DESTROY_AND_RESET(m_pipelineLayoutPostProcess, vkDestroyPipelineLayout(m_device, m_pipelineLayoutPostProcess, nullptr));
  TEST_DESTROY_AND_RESET(m_descriptorPoolPostProcess, vkDestroyDescriptorPool(m_device, m_descriptorPoolPostProcess, nullptr));
  m_descriptorSetPostProcess = VK_NULL_HANDLE;

  TEST_DESTROY_AND_RESET(m_descriptorSetLayoutPostProcess,
                         vkDestroyDescriptorSetLayout(m_device, m_descriptorSetLayoutPostProcess, nullptr));


}

void VkViewer::initRendererBuffers()
{
  // Use packed splat count when in LCC packed mode, otherwise use m_splatSet size
  const auto splatCount = (prmData.dataStorage == STORAGE_LCC_PACKED && m_lccPackedSplatCount > 0)
                          ? m_lccPackedSplatCount : (uint32_t)m_splatSet.size();

  // Skip reallocation if current capacity is sufficient (streaming optimization)
  if(m_rendererBufferCapacity >= splatCount && m_splatIndicesDevice.buffer != VK_NULL_HANDLE)
  {
    return;  // Buffers are already large enough
  }

  // Grow with 1.5x headroom to reduce reallocations during streaming
  uint32_t newCapacity = std::max(splatCount, m_rendererBufferCapacity * 3 / 2);
  newCapacity = std::max(newCapacity, 1024u);  // Minimum capacity

  // All this block for the sorting
  {
    // Vrdx sorter
    VrdxSorterCreateInfo gpuSorterInfo{.physicalDevice = m_app->getPhysicalDevice(), .device = m_app->getDevice()};
    vrdxCreateSorter(&gpuSorterInfo, &m_gpuSorter);

    {  // Create some buffer for GPU and/or CPU sorting
      // shall use minStorageBufferOffsetAlignment
      // Use minimum size of 16 bytes when no splats (for valid descriptor bindings)
      const VkDeviceSize bufferSize = std::max((VkDeviceSize)16, ((newCapacity * sizeof(uint32_t) + 15) / 16) * 16);

      m_alloc.createBuffer(m_splatIndicesHost, bufferSize, VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                           VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

      m_alloc.createBuffer(m_splatIndicesDevice, bufferSize,
                           VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT
                               | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_VERTEX_BUFFER_BIT,
                           VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

      m_alloc.createBuffer(m_splatDistancesDevice, bufferSize,
                           VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT
                               | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_VERTEX_BUFFER_BIT,
                           VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

      VrdxSorterStorageRequirements requirements;
      vrdxGetSorterKeyValueStorageRequirements(m_gpuSorter, newCapacity, &requirements);
      m_alloc.createBuffer(m_vrdxStorageDevice, requirements.size, requirements.usage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

      // Track new capacity
      m_rendererBufferCapacity = newCapacity;

      // for stats reporting only
      m_renderMemoryStats.allocVdrxInternal = (uint32_t)requirements.size;

      // generate debug information for buffers
      NVVK_DBG_NAME(m_splatIndicesHost.buffer);
      NVVK_DBG_NAME(m_splatIndicesDevice.buffer);
      NVVK_DBG_NAME(m_splatDistancesDevice.buffer);
      NVVK_DBG_NAME(m_vrdxStorageDevice.buffer);
    }
  }

  // create the device buffer for indirect parameters
  // Double-buffered to prevent race conditions during heavy load (OpenXR)
  // Use the application's frame cycle size to determine buffer count
  uint32_t frameCount = m_app->getFrameCycleSize();
  m_indirectStride = nvutils::align_up(sizeof(shaderio::IndirectParams), 
                                         m_physicalDeviceInfo.properties10.limits.minStorageBufferOffsetAlignment);
  m_alloc.createBuffer(m_indirect, m_indirectStride * frameCount,
                       VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT
                           | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_INDIRECT_BUFFER_BIT,
                       VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

  // for statistics readback

  m_alloc.createBuffer(m_indirectReadbackHost, sizeof(shaderio::IndirectParams),
                       VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                       VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

  NVVK_DBG_NAME(m_indirect.buffer);
  NVVK_DBG_NAME(m_indirectReadbackHost.buffer);

  // We create a command buffer in order to perform the copy to VRAM
  VkCommandBuffer cmd = m_app->createTempCmdBuffer();

  // The Quad
  const std::vector<uint16_t> indices  = {0, 2, 1, 2, 0, 3};
  const std::vector<float>    vertices = {-1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 1.0, 0.0, -1.0, 1.0, 0.0};

  // create the quad buffers
  m_alloc.createBuffer(m_quadVertices, vertices.size() * sizeof(float), VK_BUFFER_USAGE_2_VERTEX_BUFFER_BIT,
                       VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
  m_alloc.createBuffer(m_quadIndices, indices.size() * sizeof(uint16_t), VK_BUFFER_USAGE_2_INDEX_BUFFER_BIT,
                       VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

  NVVK_DBG_NAME(m_quadVertices.buffer);
  NVVK_DBG_NAME(m_quadIndices.buffer);

  // buffers are small so we use vkCmdUpdateBuffer for the transfers
  vkCmdUpdateBuffer(cmd, m_quadVertices.buffer, 0, vertices.size() * sizeof(float), vertices.data());
  vkCmdUpdateBuffer(cmd, m_quadIndices.buffer, 0, indices.size() * sizeof(uint16_t), indices.data());
  m_app->submitAndWaitTempCmdBuffer(cmd);

  // Uniform buffer
  // Allocate 4 slots per frame (Sort, Draw/Multiview, Left, Right) to avoid UBO races
  m_frameInfoStride = nvutils::align_up(sizeof(shaderio::FrameInfo), m_physicalDeviceInfo.properties10.limits.minUniformBufferOffsetAlignment);
  m_alloc.createBuffer(m_frameInfoBuffer, m_frameInfoStride * frameCount * 4,
                       VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT,
                       VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
  NVVK_DBG_NAME(m_frameInfoBuffer.buffer);

}

void VkViewer::deinitRendererBuffers()
{
  // TODO can we rather move this to pipelines creation/deletion ?
  if(m_gpuSorter != VK_NULL_HANDLE)
  {
    vrdxDestroySorter(m_gpuSorter);
    m_gpuSorter = VK_NULL_HANDLE;
  }

  m_alloc.destroyBuffer(m_splatDistancesDevice);
  m_alloc.destroyBuffer(m_splatIndicesDevice);
  m_alloc.destroyBuffer(m_splatIndicesHost);
  m_alloc.destroyBuffer(m_vrdxStorageDevice);
  m_rendererBufferCapacity = 0;  // Reset capacity on full deinit

  m_alloc.destroyBuffer(m_indirect);
  m_alloc.destroyBuffer(m_indirectReadbackHost);

  m_alloc.destroyBuffer(m_quadVertices);
  m_alloc.destroyBuffer(m_quadIndices);

  m_alloc.destroyBuffer(m_frameInfoBuffer);
}

}  // namespace vk_viewer
