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

  // Note: BINDING_FRAME_INFO_UBO is in Set 0 (m_descriptorSet), so we don't need to add it here (Set 1).
  
  m_rtDescriptorBindings.addBinding(RTX_BINDING_TLAS_SPLATS, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1,
                                    VK_SHADER_STAGE_RAYGEN_BIT_KHR);
  m_rtDescriptorBindings.addBinding(RTX_BINDING_TLAS_MESH, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1,
                                    VK_SHADER_STAGE_RAYGEN_BIT_KHR);

#ifdef WITH_DLSS_RR
  m_rtDescriptorBindings.addBinding(RTX_BINDING_DLSS_DIFFUSE_ALBEDO, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR);
  m_rtDescriptorBindings.addBinding(RTX_BINDING_DLSS_SPECULAR_ALBEDO, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR);
  m_rtDescriptorBindings.addBinding(RTX_BINDING_DLSS_NORMAL_ROUGH, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR);
  m_rtDescriptorBindings.addBinding(RTX_BINDING_DLSS_MOTION, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR);
  m_rtDescriptorBindings.addBinding(RTX_BINDING_DLSS_LINEAR_DEPTH, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR);
  m_rtDescriptorBindings.addBinding(RTX_BINDING_DLSS_SPEC_HIT_DIST, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR);
#else
  // Add motion binding for Space Warp
  m_rtDescriptorBindings.addBinding(RTX_BINDING_DLSS_MOTION, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR);
#endif

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

#ifdef WITH_DLSS_RR
  // DLSS-RR G-buffer outputs
  writeContainer.append(m_rtDescriptorBindings.getWriteSet(RTX_BINDING_DLSS_DIFFUSE_ALBEDO, m_rtDescriptorSet),
                        m_gBuffers.getColorImageView(COLOR_DLSS_DIFFUSE_ALBEDO), VK_IMAGE_LAYOUT_GENERAL);
  writeContainer.append(m_rtDescriptorBindings.getWriteSet(RTX_BINDING_DLSS_SPECULAR_ALBEDO, m_rtDescriptorSet),
                        m_gBuffers.getColorImageView(COLOR_DLSS_SPECULAR_ALBEDO), VK_IMAGE_LAYOUT_GENERAL);
  writeContainer.append(m_rtDescriptorBindings.getWriteSet(RTX_BINDING_DLSS_NORMAL_ROUGH, m_rtDescriptorSet),
                        m_gBuffers.getColorImageView(COLOR_DLSS_NORMAL_ROUGH), VK_IMAGE_LAYOUT_GENERAL);
  writeContainer.append(m_rtDescriptorBindings.getWriteSet(RTX_BINDING_DLSS_MOTION, m_rtDescriptorSet),
                        m_gBuffers.getColorImageView(COLOR_DLSS_MOTION), VK_IMAGE_LAYOUT_GENERAL);
  writeContainer.append(m_rtDescriptorBindings.getWriteSet(RTX_BINDING_DLSS_LINEAR_DEPTH, m_rtDescriptorSet),
                        m_gBuffers.getColorImageView(COLOR_DLSS_LINEAR_DEPTH), VK_IMAGE_LAYOUT_GENERAL);
  writeContainer.append(m_rtDescriptorBindings.getWriteSet(RTX_BINDING_DLSS_SPEC_HIT_DIST, m_rtDescriptorSet),
                        m_gBuffers.getColorImageView(COLOR_DLSS_SPEC_HIT_DIST), VK_IMAGE_LAYOUT_GENERAL);
#else
  // Bind motion buffer for Space Warp support
  // COLOR_MOTION (index 2) is always present in the GBuffer color formats
  writeContainer.append(m_rtDescriptorBindings.getWriteSet(RTX_BINDING_DLSS_MOTION, m_rtDescriptorSet),
                        m_gBuffers.getColorImageView(COLOR_MOTION), VK_IMAGE_LAYOUT_GENERAL);
#endif

  // actually write
  if (writeContainer.size() > 0) {

    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writeContainer.size()), writeContainer.data(), 0, nullptr);
  }
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

#ifdef WITH_DLSS_RR

    // DLSS-RR G-buffer outputs
    writeContainer.append(m_rtDescriptorBindings.getWriteSet(RTX_BINDING_DLSS_DIFFUSE_ALBEDO, m_rtDescriptorSet),
                          m_gBuffers.getColorImageView(COLOR_DLSS_DIFFUSE_ALBEDO), VK_IMAGE_LAYOUT_GENERAL);
    writeContainer.append(m_rtDescriptorBindings.getWriteSet(RTX_BINDING_DLSS_SPECULAR_ALBEDO, m_rtDescriptorSet),
                          m_gBuffers.getColorImageView(COLOR_DLSS_SPECULAR_ALBEDO), VK_IMAGE_LAYOUT_GENERAL);
    writeContainer.append(m_rtDescriptorBindings.getWriteSet(RTX_BINDING_DLSS_NORMAL_ROUGH, m_rtDescriptorSet),
                          m_gBuffers.getColorImageView(COLOR_DLSS_NORMAL_ROUGH), VK_IMAGE_LAYOUT_GENERAL);
    writeContainer.append(m_rtDescriptorBindings.getWriteSet(RTX_BINDING_DLSS_MOTION, m_rtDescriptorSet),
                          m_gBuffers.getColorImageView(COLOR_DLSS_MOTION), VK_IMAGE_LAYOUT_GENERAL);
    writeContainer.append(m_rtDescriptorBindings.getWriteSet(RTX_BINDING_DLSS_LINEAR_DEPTH, m_rtDescriptorSet),
                          m_gBuffers.getColorImageView(COLOR_DLSS_LINEAR_DEPTH), VK_IMAGE_LAYOUT_GENERAL);
    writeContainer.append(m_rtDescriptorBindings.getWriteSet(RTX_BINDING_DLSS_SPEC_HIT_DIST, m_rtDescriptorSet),
                          m_gBuffers.getColorImageView(COLOR_DLSS_SPEC_HIT_DIST), VK_IMAGE_LAYOUT_GENERAL);
#else
    // Bind motion buffer for Space Warp support
    // COLOR_MOTION (index 2) is always present in the GBuffer color formats
    writeContainer.append(m_rtDescriptorBindings.getWriteSet(RTX_BINDING_DLSS_MOTION, m_rtDescriptorSet),
                          m_gBuffers.getColorImageView(COLOR_MOTION), VK_IMAGE_LAYOUT_GENERAL);
#endif

    // let's update
    if (writeContainer.size() > 0) {
      vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writeContainer.size()), writeContainer.data(), 0, nullptr);
    }
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

  if(m_descriptorSet == VK_NULL_HANDLE || m_rtDescriptorSet == VK_NULL_HANDLE 
     || m_rtPipeline == VK_NULL_HANDLE || m_rtPipelineLayout == VK_NULL_HANDLE)
    return;

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
  #ifdef WITH_OPENXR
  m_pcRay.useNdcMotion               = (m_xr && m_xr->isSpaceWarpSupported()) ? 1 : 0;
#else
  m_pcRay.useNdcMotion               = 0;
#endif

  // Dynamic offsets for descriptor sets:
  // Set 0 (Raster): [FrameInfo, Indirect] - But wait, initRtPipeline set up descSets{m_descriptorSet, m_rtDescriptorSet}
  // m_descriptorSet layout has FrameInfo at binding 0, Indirect at binding 3.
  // m_rtDescriptorSet layout has FrameInfo at binding BINDING_FRAME_INFO_UBO.
  //
  // However, vkCmdBindDescriptorSets takes ONE array of dynamic offsets that applies to all dynamic descriptors in the specified sets sequentially.
  // We are binding TWO sets: Set 0 (m_descriptorSet) and Set 1 (m_rtDescriptorSet).
  // Set 0 has 2 dynamic buffers: FrameInfo (binding 0) and Indirect (binding 3).
  // Set 1 has 1 dynamic buffer: FrameInfo (binding BINDING_FRAME_INFO_UBO).
  //
  // The offsets array must contain offsets for ALL dynamic buffers in the bound sets, in set order, then binding order.
  // Order: Set 0 Binding 0, Set 0 Binding 3, Set 1 Binding BINDING_FRAME_INFO_UBO.
  
  uint32_t frameInfoOffset = m_currentFrameInfoOffset; 
  // We need to use the offset that was just written to.
  // Since raytrace() is called after updateAndUploadFrameInfoUBO(), and update... increments the offset,
  // we should use m_lastFrameInfoOffset which stores the offset used for the current frame/pass.
  uint32_t currentFrameInfoOffset = m_lastFrameInfoOffset;
  
  uint32_t indirectOffset = static_cast<uint32_t>(m_frameIndex * m_indirectStride);
  
  // Combined offsets for Set 0 (Set 1 has no dynamic buffers)
  std::vector<uint32_t> dynamicOffsets = {
      currentFrameInfoOffset, // Set 0, Binding 0 (FrameInfo)
      indirectOffset          // Set 0, Binding 3 (Indirect)
  };

  std::vector<VkDescriptorSet> descSets{m_descriptorSet, m_rtDescriptorSet};
  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipeline);
  
  // Bind both sets at once with the combined dynamic offsets array
  vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipelineLayout, 0,
                          (uint32_t)descSets.size(), descSets.data(), 
                          (uint32_t)dynamicOffsets.size(), dynamicOffsets.data());

  m_pcRay.vertexAddress = m_splatSetVk.m_splatModel.vertexBuffer.address;
  m_pcRay.indexAddress  = m_splatSetVk.m_splatModel.indexBuffer.address;

  vkCmdPushConstants(cmdBuf, m_rtPipelineLayout,
                     VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR
                         | VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
                     0, sizeof(shaderio::PushConstantRay), &m_pcRay);


  vkCmdTraceRaysKHR(cmdBuf, &m_sbtRegions.raygen, &m_sbtRegions.miss, &m_sbtRegions.hit, &m_sbtRegions.callable,
                    traceWidth, traceHeight, 1);
}

//--------------------------------------------------------------------------------------------------
// Ray Tracing with VK_KHR_multiview support (mobile VR optimization)
// Single raytrace call renders both eyes efficiently for side-by-side stereo
//
void GaussianSplatting::raytraceMultiview(const VkCommandBuffer& cmdBuf, bool meshDepthOnly,
                                          const glm::mat4& leftViewMat, const glm::mat4& leftProjMat,
                                          const glm::mat4& rightViewMat, const glm::mat4& rightProjMat,
                                          const glm::vec3& leftEyePos, const glm::vec3& rightEyePos,
                                          glm::ivec2 viewportSize)
{
  NVVK_DBG_SCOPE(cmdBuf);

  if(m_descriptorSet == VK_NULL_HANDLE || m_rtDescriptorSet == VK_NULL_HANDLE 
     || m_rtPipeline == VK_NULL_HANDLE || m_rtPipelineLayout == VK_NULL_HANDLE)
    return;

  const std::string name = meshDepthOnly ? "Raytracing multiview prepass" : "Raytracing multiview";

  auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmdBuf, name);

  // Use full SBS dimensions for multiview rendering
  const uint32_t traceWidth  = (viewportSize.x > 0) ? static_cast<uint32_t>(viewportSize.x) : static_cast<uint32_t>(m_viewSize.x);
  const uint32_t traceHeight = (viewportSize.y > 0) ? static_cast<uint32_t>(viewportSize.y) : static_cast<uint32_t>(m_viewSize.y);

  // Store both eye matrices in arrays for multiview shader access
  prmFrame.viewMatrixArray[0] = leftViewMat;
  prmFrame.viewMatrixArray[1] = rightViewMat;
  prmFrame.viewInverseArray[0] = glm::inverse(leftViewMat);
  prmFrame.viewInverseArray[1] = glm::inverse(rightViewMat);
  prmFrame.projectionMatrixArray[0] = leftProjMat;
  prmFrame.projectionMatrixArray[1] = rightProjMat;
  prmFrame.projInverseArray[0] = glm::inverse(leftProjMat);
  prmFrame.projInverseArray[1] = glm::inverse(rightProjMat);
  prmFrame.cameraPositionArray[0] = leftEyePos;
  prmFrame.cameraPositionArray[1] = rightEyePos;
  // Per-eye sensor pose for 3DGUT projection
  for(int eyeIdx = 0; eyeIdx < 2; ++eyeIdx)
  {
    glm::quat viewQuat         = glm::quat_cast(prmFrame.viewMatrixArray[eyeIdx]);
    prmFrame.viewQuatArray[eyeIdx]  = glm::vec4(viewQuat.x, viewQuat.y, viewQuat.z, viewQuat.w);
    prmFrame.viewTransArray[eyeIdx] = prmFrame.viewMatrixArray[eyeIdx][3];
  }
  prmFrame.multiviewEnabled = 1;

  // Upload frame info UBO using vkCmdUpdateBuffer (same pattern as updateAndUploadFrameInfoUBO)
  vkCmdUpdateBuffer(cmdBuf, m_frameInfoBuffer.buffer, 0, sizeof(shaderio::FrameInfo), &prmFrame);

  VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask   = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask   = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
  vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR, 0, 1, &barrier, 0, NULL, 0, NULL);

  // Initializing push constant values
  m_pcRay.modelMatrix        = m_splatSetVk.transform;
  m_pcRay.modelMatrixInverse = m_splatSetVk.transformInverse;
  m_pcRay.modelMatrixRotScaleInverse = glm::inverse(glm::mat3(m_splatSetVk.transform));
  m_pcRay.meshDepthOnly = meshDepthOnly;
  m_pcRay.viewportOffset = glm::ivec2(0, 0);
#ifdef WITH_OPENXR
  m_pcRay.useNdcMotion = (m_xr && m_xr->isSpaceWarpSupported()) ? 1 : 0;
#else
  m_pcRay.useNdcMotion = 0;
#endif

  // Dynamic offsets
  uint32_t indirectOffset = static_cast<uint32_t>(m_frameIndex * m_indirectStride);
  uint32_t currentFrameInfoOffset = m_lastFrameInfoOffset;

  // Combined offsets for Set 0 (Set 1 has no dynamic buffers)
  std::vector<uint32_t> dynamicOffsets = {
      currentFrameInfoOffset, // Set 0: FrameInfo
      indirectOffset          // Set 0: Indirect
  };

  std::vector<VkDescriptorSet> descSets{m_descriptorSet, m_rtDescriptorSet};
  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipeline);

  
  // Bind both sets at once
  vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipelineLayout, 0,
                          (uint32_t)descSets.size(), descSets.data(), 
                          (uint32_t)dynamicOffsets.size(), dynamicOffsets.data());

  m_pcRay.vertexAddress = m_splatSetVk.m_splatModel.vertexBuffer.address;
  m_pcRay.indexAddress = m_splatSetVk.m_splatModel.indexBuffer.address;

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
