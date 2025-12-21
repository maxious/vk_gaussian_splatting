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

//--------------------------------------------------------------------------------------------------
// This descriptor set holds the Acceleration structure and the output image
//
void GaussianSplatting::initRtDescriptorSet()
{
  //SCOPED_TIMER(__FUNCTION__"\n");

  //////////////////////
  // Bindings

  m_rtDescriptorBindings.clear();

  m_rtDescriptorBindings.addBinding(RTX_BINDING_OUTIMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR);
  m_rtDescriptorBindings.addBinding(RTX_BINDING_AUX1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR);
  m_rtDescriptorBindings.addBinding(RTX_BINDING_OUTDEPTH, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR);

  m_rtDescriptorBindings.addBinding(RTX_BINDING_TLAS_SPLATS, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1,
                                    VK_SHADER_STAGE_RAYGEN_BIT_KHR);
  m_rtDescriptorBindings.addBinding(RTX_BINDING_TLAS_MESH, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1,
                                    VK_SHADER_STAGE_RAYGEN_BIT_KHR);

  NVVK_CHECK(m_rtDescriptorBindings.createDescriptorSetLayout(m_device, 0, &m_rtDescriptorSetLayout));
  NVVK_DBG_NAME(m_rtDescriptorSetLayout);

  //
  std::vector<VkDescriptorPoolSize> poolSize;
  m_rtDescriptorBindings.appendPoolSizes(poolSize);
  VkDescriptorPoolCreateInfo poolInfo = {
      .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets       = 1,
      .poolSizeCount = uint32_t(poolSize.size()),
      .pPoolSizes    = poolSize.data(),
  };
  NVVK_CHECK(vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_rtDescriptorPool));
  NVVK_DBG_NAME(m_rtDescriptorPool);

  VkDescriptorSetAllocateInfo allocInfo = {
      .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool     = m_rtDescriptorPool,
      .descriptorSetCount = 1,
      .pSetLayouts        = &m_rtDescriptorSetLayout,
  };
  NVVK_CHECK(vkAllocateDescriptorSets(m_device, &allocInfo, &m_rtDescriptorSet));
  NVVK_DBG_NAME(m_rtDescriptorSet);

  //////////////////////
  // Writes

  nvvk::WriteSetContainer writeContainer;

  // Output image buffer
  writeContainer.append(m_rtDescriptorBindings.getWriteSet(RTX_BINDING_OUTIMAGE, m_rtDescriptorSet),
                        m_gBuffers.getColorImageView(COLOR_MAIN), VK_IMAGE_LAYOUT_GENERAL);
  writeContainer.append(m_rtDescriptorBindings.getWriteSet(RTX_BINDING_AUX1, m_rtDescriptorSet),
                        m_gBuffers.getColorImageView(COLOR_AUX1), VK_IMAGE_LAYOUT_GENERAL);
  writeContainer.append(m_rtDescriptorBindings.getWriteSet(RTX_BINDING_OUTDEPTH, m_rtDescriptorSet),
                        m_gBuffers.getDepthImageView(), VK_IMAGE_LAYOUT_GENERAL);

  // splats TLAS
  if(m_splatSetVk.rtAccelerationStructures.tlas.accel != NULL)
    writeContainer.append(m_rtDescriptorBindings.getWriteSet(RTX_BINDING_TLAS_SPLATS, m_rtDescriptorSet),
                          m_splatSetVk.rtAccelerationStructures.tlas);
  // mesh TLAS
  if(m_meshSetVk.instances.size() && (m_meshSetVk.rtAccelerationStructures.tlas.accel != NULL))
  {
    writeContainer.append(m_rtDescriptorBindings.getWriteSet(RTX_BINDING_TLAS_MESH, m_rtDescriptorSet),
                          m_meshSetVk.rtAccelerationStructures.tlas);
  }

  // actually write
  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writeContainer.size()), writeContainer.data(), 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Writes the output image to the descriptor set
// - Required when changing resolution
//
void GaussianSplatting::updateRtDescriptorSet()
{
  //SCOPED_TIMER(__FUNCTION__"\n");

  // update only if the descriptor set is already initialized
  if(m_rtDescriptorSet != VK_NULL_HANDLE)
  {
    nvvk::WriteSetContainer writeContainer;

    // Output image buffer
    writeContainer.append(m_rtDescriptorBindings.getWriteSet(RTX_BINDING_OUTIMAGE, m_rtDescriptorSet),
                          m_gBuffers.getColorImageView(COLOR_MAIN), VK_IMAGE_LAYOUT_GENERAL);
    writeContainer.append(m_rtDescriptorBindings.getWriteSet(RTX_BINDING_AUX1, m_rtDescriptorSet),
                          m_gBuffers.getColorImageView(COLOR_AUX1), VK_IMAGE_LAYOUT_GENERAL);
    writeContainer.append(m_rtDescriptorBindings.getWriteSet(RTX_BINDING_OUTDEPTH, m_rtDescriptorSet),
                          m_gBuffers.getDepthImageView(), VK_IMAGE_LAYOUT_GENERAL);
    // let's update
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writeContainer.size()), writeContainer.data(), 0, nullptr);
  }
}

//--------------------------------------------------------------------------------------------------
// Pipeline for the ray tracer: all shaders, raygen, chit, miss
//
void GaussianSplatting::initRtPipeline()
{
  //SCOPED_TIMER(__FUNCTION__"\n");

  enum StageIndices
  {
    eRaygen,
    eMiss,
    eMiss2,
    eClosestHit,
    eAnyHit,
    eIntersection,
    eStageIndicesCount
  };

  // if not using AABBs we do not use the intersection shader (last stage listed)
  uint32_t stagesCount = prmRtxData.useAABBs ? eStageIndicesCount : eStageIndicesCount - 1;

  // All stages
  std::array<VkPipelineShaderStageCreateInfo, eStageIndicesCount> stages{};
  VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stage.pName = "main";  // All the same entry point
  // Raygen
  stage.module    = m_shaders.rtxRgenShader;
  stage.stage     = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
  stages[eRaygen] = stage;
  // Miss
  stage.module  = m_shaders.rtxRmissShader;
  stage.stage   = VK_SHADER_STAGE_MISS_BIT_KHR;
  stages[eMiss] = stage;
  // The second miss shader is invoked when a shadow ray misses the geometry. It simply indicates that no occlusion has been found
  stage.module   = m_shaders.rtxRmiss2Shader;
  stage.stage    = VK_SHADER_STAGE_MISS_BIT_KHR;
  stages[eMiss2] = stage;
  // Hit Group - Closest Hit
  stage.module        = m_shaders.rtxRchitShader;
  stage.stage         = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
  stages[eClosestHit] = stage;
  // Hit Group - Any Hit
  stage.module    = m_shaders.rtxRahitShader;
  stage.stage     = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
  stages[eAnyHit] = stage;
  // Hit Group - Intersection (used only if useAABBs is true)
  stage.module          = m_shaders.rtxRintShader;
  stage.stage           = VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
  stages[eIntersection] = stage;

  // Shader groups
  VkRayTracingShaderGroupCreateInfoKHR group{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
  group.anyHitShader       = VK_SHADER_UNUSED_KHR;
  group.closestHitShader   = VK_SHADER_UNUSED_KHR;
  group.generalShader      = VK_SHADER_UNUSED_KHR;
  group.intersectionShader = VK_SHADER_UNUSED_KHR;

  // Raygen
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eRaygen;
  m_rtShaderGroups.push_back(group);

  // Miss
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eMiss;
  m_rtShaderGroups.push_back(group);

  // Shadow Miss
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eMiss2;
  m_rtShaderGroups.push_back(group);

  if(prmRtxData.useAABBs)
  {
    // Hit 0 any hit shader with procedural intersections
    group.type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR;
    group.generalShader      = VK_SHADER_UNUSED_KHR;
    group.closestHitShader   = VK_SHADER_UNUSED_KHR;
    group.anyHitShader       = eAnyHit;
    group.intersectionShader = eIntersection;
    m_rtShaderGroups.push_back(group);
  }
  else
  {
    // Hit 0 any hit shader with mesh ICOSA
    group.type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    group.generalShader      = VK_SHADER_UNUSED_KHR;
    group.closestHitShader   = VK_SHADER_UNUSED_KHR;
    group.intersectionShader = VK_SHADER_UNUSED_KHR;
    group.anyHitShader       = eAnyHit;
    m_rtShaderGroups.push_back(group);
  }

  // Hit 1 Closest-hit only (for eMeshTlas)
  group.type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
  group.generalShader      = VK_SHADER_UNUSED_KHR;
  group.anyHitShader       = VK_SHADER_UNUSED_KHR;
  group.intersectionShader = VK_SHADER_UNUSED_KHR;
  group.closestHitShader   = eClosestHit;
  m_rtShaderGroups.push_back(group);

  // Push constant: we want to be able to update constants used by the shaders
  VkPushConstantRange pushConstant{VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR
                                       | VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
                                   0, sizeof(shaderio::PushConstantRay)};


  VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
  pipelineLayoutCreateInfo.pPushConstantRanges    = &pushConstant;

  // Descriptor sets: one specific to ray tracing, and one shared with the rasterization pipeline
  std::vector<VkDescriptorSetLayout> rtDescSetLayouts = {m_descriptorSetLayout, m_rtDescriptorSetLayout};
  pipelineLayoutCreateInfo.setLayoutCount             = static_cast<uint32_t>(rtDescSetLayouts.size());
  pipelineLayoutCreateInfo.pSetLayouts                = rtDescSetLayouts.data();

  vkCreatePipelineLayout(m_device, &pipelineLayoutCreateInfo, nullptr, &m_rtPipelineLayout);

  // Assemble the shader stages and recursion depth info into the ray tracing pipeline
  VkRayTracingPipelineCreateInfoKHR rayPipelineInfo{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
  rayPipelineInfo.stageCount = stagesCount;  // Stages are shaders
  rayPipelineInfo.pStages    = stages.data();

  // In this case, m_rtShaderGroups.size() == 4: we have one raygen group,
  // two miss shader groups, and one hit group.
  rayPipelineInfo.groupCount = static_cast<uint32_t>(m_rtShaderGroups.size());
  rayPipelineInfo.pGroups    = m_rtShaderGroups.data();

  // The ray tracing process can shoot rays from the camera, and a shadow ray can be shot from the
  // hit points of the camera rays, hence a recursion level of 2. This number should be kept as low
  // as possible for performance reasons. Even recursive ray tracing should be flattened into a loop
  // in the ray generation to avoid deep recursion.
  rayPipelineInfo.maxPipelineRayRecursionDepth = 2;  // Ray depth
  rayPipelineInfo.layout                       = m_rtPipelineLayout;

  vkCreateRayTracingPipelinesKHR(m_device, {}, {}, 1, &rayPipelineInfo, nullptr, &m_rtPipeline);


  // Spec only guarantees 1 level of "recursion". Check for that sad possibility here.
  if(m_rtProperties.maxRayRecursionDepth <= 1)
  {
    throw std::runtime_error("Device fails to support ray recursion (m_rtProperties.maxRayRecursionDepth <= 1)");
  }

  // Creating the SBT
  {
    // Shader Binding Table (SBT) setup
    nvvk::SBTGenerator sbtGenerator;
    sbtGenerator.init(m_app->getDevice(), m_rtProperties);

    // Prepare SBT data from ray pipeline
    size_t bufferSize = sbtGenerator.calculateSBTBufferSize(m_rtPipeline, rayPipelineInfo);

    // Create SBT buffer using the size from above
    NVVK_CHECK(m_alloc.createBuffer(m_rtSBTBuffer, bufferSize, VK_BUFFER_USAGE_2_SHADER_BINDING_TABLE_BIT_KHR, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                    VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
                                    sbtGenerator.getBufferAlignment()));
    NVVK_DBG_NAME(m_rtSBTBuffer.buffer);

    // Pass the manual mapped pointer to fill the sbt data
    NVVK_CHECK(sbtGenerator.populateSBTBuffer(m_rtSBTBuffer.address, bufferSize, m_rtSBTBuffer.mapping));

    // Retrieve the regions, which are using addresses based on the m_sbtBuffer.address
    m_sbtRegions = sbtGenerator.getSBTRegions();

    sbtGenerator.deinit();
  }
}

//--------------------------------------------------------------------------------------------------
// Ray Tracing the scene
//
void GaussianSplatting::raytrace(const VkCommandBuffer& cmdBuf, bool meshDepthOnly, glm::ivec2 viewportOffset, glm::ivec2 viewportSize)
{
  NVVK_DBG_SCOPE(cmdBuf);

  const std::string name = meshDepthOnly ? "Raytracing prepass" : "Raytracing";

  auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmdBuf, name);

  // Use provided viewport size or fall back to full viewport
  const uint32_t traceWidth  = (viewportSize.x > 0) ? static_cast<uint32_t>(viewportSize.x) : static_cast<uint32_t>(m_viewSize.x);
  const uint32_t traceHeight = (viewportSize.y > 0) ? static_cast<uint32_t>(viewportSize.y) : static_cast<uint32_t>(m_viewSize.y);

  // Initializing push constant values
  m_pcRay.modelMatrix        = m_splatSetVk.transform;
  m_pcRay.modelMatrixInverse = m_splatSetVk.transformInverse;
  // cast to mat3 extracts only the rot/scale part of the transform
  m_pcRay.modelMatrixRotScaleInverse = glm::inverse(glm::mat3(m_splatSetVk.transform));
  m_pcRay.meshDepthOnly              = meshDepthOnly;
  m_pcRay.viewportOffset             = viewportOffset;

  std::vector<VkDescriptorSet> descSets{m_descriptorSet, m_rtDescriptorSet};
  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipeline);
  vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipelineLayout, 0,
                          (uint32_t)descSets.size(), descSets.data(), 0, nullptr);

  m_pcRay.vertexAddress = m_splatSetVk.m_splatModel.vertexBuffer.address;
  m_pcRay.indexAddress  = m_splatSetVk.m_splatModel.indexBuffer.address;

  vkCmdPushConstants(cmdBuf, m_rtPipelineLayout,
                     VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR
                         | VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
                     0, sizeof(shaderio::PushConstantRay), &m_pcRay);


  vkCmdTraceRaysKHR(cmdBuf, &m_sbtRegions.raygen, &m_sbtRegions.miss, &m_sbtRegions.hit, &m_sbtRegions.callable,
                    traceWidth, traceHeight, 1);
}


bool GaussianSplatting::updateFrameCounter()
{
  static float     ref_fov{0};
  static glm::mat4 ref_cam_matrix;

  const auto& m   = cameraManip->getViewMatrix();
  const auto  fov = cameraManip->getFov();

  if(ref_cam_matrix != m || ref_fov != fov)
  {
    resetFrameCounter();
    ref_cam_matrix = m;
    ref_fov        = fov;
  }

  if(prmFrame.frameSampleId >= prmFrame.frameSampleMax)
  {
    return false;
  }
  prmFrame.frameSampleId++;
  return true;
}

}  // namespace vk_gaussian_splatting
