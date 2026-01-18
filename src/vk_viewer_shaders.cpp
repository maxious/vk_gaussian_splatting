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

namespace vk_viewer {

void VkViewer::updateSlangMacros()
{
  m_shaderMacros =  // comment to force clang new line and better indent
      {{"PIPELINE", std::to_string(prmSelectedPipeline)},
       {"HYBRID_ENABLED", std::to_string((int)(prmSelectedPipeline == PIPELINE_HYBRID || prmSelectedPipeline == PIPELINE_HYBRID_3DGUT))},
       {"CAMERA_TYPE", std::to_string(m_cameraSet.getCamera().model)},
       {"VISUALIZE", std::to_string((int)prmRender.visualize)},
       {"DISABLE_OPACITY_GAUSSIAN", std::to_string((int)prmRender.opacityGaussianDisabled)},
       {"FRUSTUM_CULLING_MODE", std::to_string(prmRaster.frustumCulling)},
       // Disabled, TODO do we enable ortho cam in the UI/camera controller
       {"ORTHOGRAPHIC_MODE", "0"},
       {"SHOW_SH_ONLY", std::to_string((int)prmRender.showShOnly)},
       {"MAX_SH_DEGREE", std::to_string(prmRender.maxShDegree)},
       {"DATA_STORAGE", std::to_string(prmData.dataStorage)},
       {"SH_FORMAT", std::to_string(prmData.shFormat)},
       {"POINT_CLOUD_MODE", std::to_string((int)prmRaster.pointCloudModeEnabled)},
       {"USE_BARYCENTRIC", std::to_string((int)prmRaster.fragmentBarycentric)},
       {"WIREFRAME", std::to_string((int)prmRender.wireframe)},
       {"DISTANCE_COMPUTE_WORKGROUP_SIZE", std::to_string((int)prmRaster.distShaderWorkgroupSize)},
       {"RASTER_MESH_WORKGROUP_SIZE", std::to_string((int)prmRaster.meshShaderWorkgroupSize)},
       {"MS_ANTIALIASING", std::to_string((int)prmRaster.msAntialiasing)},
       {"EXTENT_METHOD", std::to_string((int)prmRaster.extentProjection)},
       // RTX
       {"TEMPORAL_SAMPLING", std::to_string((int)prmRtx.temporalSampling)},
       {"KERNEL_DEGREE", std::to_string(prmRtx.kernelDegree)},
       {"KERNEL_MIN_RESPONSE", std::to_string(prmRtx.kernelMinResponse)},
       {"KERNEL_ADAPTIVE_CLAMPING", std::to_string((int)prmRtx.kernelAdaptiveClamping)},
       {"PAYLOAD_ARRAY_SIZE", std::to_string(prmRtx.payloadArraySize)},
       {"RTX_USE_INSTANCES", std::to_string((int)prmRtxData.useTlasInstances)},
       {"RTX_USE_AABBS", std::to_string((int)prmRtxData.useAABBs)},
       {"RTX_USE_MESHES", std::to_string((int)m_meshSetVk.instances.size())},
       {"RTX_DOF_ENABLED", std::to_string((int)m_cameraSet.getCamera().dofEnabled)},
       // VK_KHR_multiview support for mobile VR (stereo rendering optimization)
       {"MULTIVIEW_ENABLED", "1"},
#ifdef WITH_DLSS_RR
       {"WITH_DLSS_RR", std::to_string((int)m_dlssRREnabled)}
#endif
      };

  m_slangCompiler.clearMacros();

  // then provide the char* strings to the compiler
  for(auto& macro : m_shaderMacros)
  {
    m_slangCompiler.addMacro({macro.first.c_str(), macro.second.c_str()});
  }
}

bool VkViewer::compileSlangShader(const std::string& filename, VkShaderModule& module)
{

  if(!m_slangCompiler.compileFile(filename))
  {
    return false;
  }

  if(module != VK_NULL_HANDLE)
    vkDestroyShaderModule(m_device, module, nullptr);

  // Create the VK module
  VkShaderModuleCreateInfo createInfo{.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                      .codeSize = m_slangCompiler.getSpirvSize(),
                                      .pCode    = m_slangCompiler.getSpirv()};

  if(m_slangCompiler.getSpirvSize() == 0)
  {
    LOGE("Missing entry point in shader %s\n", filename.c_str());
    return false;
  }
  NVVK_CHECK(vkCreateShaderModule(m_device, &createInfo, nullptr, &module));
  NVVK_DBG_NAME(module);

  m_shaders.modules.emplace_back(&module);

  return true;
}

bool VkViewer::initShaders(void)
{
  auto startTime = std::chrono::high_resolution_clock::now();

  bool success = true;

  updateSlangMacros();

  // Particles distance to viewpoint and frustum culling
  success &= compileSlangShader("dist.comp.slang", m_shaders.distShader);
  // Chunk-based hierarchical frustum culling
  success &= compileSlangShader("chunk_cull.comp.slang", m_shaders.chunkCullShader);
  // 3DGS raster
  success &= compileSlangShader("threedgs_raster.vert.slang", m_shaders.vertexShader);
  success &= compileSlangShader("threedgs_raster.mesh.slang", m_shaders.meshShader);
  success &= compileSlangShader("threedgs_raster.frag.slang", m_shaders.fragmentShader);
  // 3DGUT raster
  success &= compileSlangShader("threedgut_raster.mesh.slang", m_shaders.threedgutMeshShader);
  success &= compileSlangShader("threedgut_raster.frag.slang", m_shaders.threedgutFragmentShader);
  // Mesh raster
  success &= compileSlangShader("threedmesh_raster.vert.slang", m_shaders.meshVertexShader);
  success &= compileSlangShader("threedmesh_raster.frag.slang", m_shaders.meshFragmentShader);
  // Ray trace
  success &= compileSlangShader("threedgrt_raytrace.rgen.slang", m_shaders.rtxRgenShader);
  success &= compileSlangShader("threedgrt_raytrace.rmiss.slang", m_shaders.rtxRmissShader);
  success &= compileSlangShader("threedgrt_raytrace_shadow.rmiss.slang", m_shaders.rtxRmiss2Shader);
  success &= compileSlangShader("threedgrt_raytrace.rchit.slang", m_shaders.rtxRchitShader);
  success &= compileSlangShader("threedgrt_raytrace.rahit.slang", m_shaders.rtxRahitShader);
  success &= compileSlangShader("threedgrt_raytrace.rint.slang", m_shaders.rtxRintShader);
  // Post processings
  success &= compileSlangShader("post.comp.slang", m_shaders.postComputeShader);
  // VDZ depth mesh
  success &= compileSlangShader("vdz_mesh.vert.slang", m_shaders.vdzMeshVertexShader);
  success &= compileSlangShader("vdz_mesh.frag.slang", m_shaders.vdzMeshFragmentShader);
  // VDZ hybrid rendering (mesh + POM)
  success &= compileSlangShader("vdz_hybrid.vert.slang", m_shaders.vdzHybridVertexShader);
  success &= compileSlangShader("vdz_hybrid.frag.slang", m_shaders.vdzHybridFragmentShader);
  // Hand mesh (XR skinned hands)
  success &= compileSlangShader("hand_mesh.vert.slang", m_shaders.handMeshVertexShader);
  success &= compileSlangShader("hand_mesh.frag.slang", m_shaders.handMeshFragmentShader);

  if(!success)
    return (m_shaders.valid = false);

  auto      endTime   = std::chrono::high_resolution_clock::now();
  long long buildTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
  LOGI("Shaders updated in %lldms\n", buildTime);

  return (m_shaders.valid = true);
}

void VkViewer::deinitShaders(void)
{
  for(auto& shader : m_shaders.modules)
  {
    vkDestroyShaderModule(m_device, *shader, nullptr);
    *shader = VK_NULL_HANDLE;
  }

  m_shaders.valid = false;
  m_shaders.modules.clear();
}

}  // namespace vk_viewer
