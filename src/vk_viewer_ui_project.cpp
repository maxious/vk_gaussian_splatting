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

// This file is included from vk_viewer_ui.cpp - do not compile separately
// Contains: Project save/load functionality (loadProjectIfNeeded, saveProject)

#include <iomanip>

// Helper macros for JSON loading
#define LOAD1(val, item, name)                                                                                         \
  if((item).contains(name))                                                                                            \
  (val) = (item)[name]
#define LOAD3(val, item, name)                                                                                         \
  if((item).contains(name))                                                                                            \
  (val) = glm::vec3((item)[name][0], (item)[name][1], (item)[name][2])
#define LOAD4_FROM3(val, item, name)                                                                                   \
  if((item).contains(name))                                                                                            \
  (val) = glm::vec4((item)[name][0], (item)[name][1], (item)[name][2], 0.0f)

bool VkViewerUI::loadProjectIfNeeded()
{
  // Nothing to load
  if(prmScene.projectToLoadFilename.empty())
    return true;

  auto path = prmScene.projectToLoadFilename.string();

  // load the json and set loading status
  if(!loadingProject)
  {
    if(!m_radianceFields.empty())
      ImGui::OpenPopup("Load .vkg project file ?");

    // Always center this window when appearing
    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

    bool doReset = true;

    if(ImGui::BeginPopupModal("Load .vkg project file ?", NULL, ImGuiWindowFlags_AlwaysAutoResize))
    {
      doReset = false;

      ImGui::Text("The current project will be entirely replaced.\nThis operation cannot be undone!");
      ImGui::Separator();

      if(ImGui::Button("OK", ImVec2(120, 0)))
      {
        doReset = true;
        ImGui::CloseCurrentPopup();
      }
      ImGui::SetItemDefaultFocus();
      ImGui::SameLine();
      if(ImGui::Button("Cancel", ImVec2(120, 0)))
      {
        // cancel any request leading to a reset
        prmScene.sceneToLoadFilename   = "";
        prmScene.projectToLoadFilename = "";
        ImGui::CloseCurrentPopup();
      }
      ImGui::EndPopup();
    }

    if(doReset)
    {
      LOGI("Opening project file %s\n", path.c_str());

      std::ifstream i(path);
      if(!i.is_open())
      {
        LOGE("Error : unable to open project file %s\n", path.c_str());
        prmScene.projectToLoadFilename = "";
        return false;
      }

      try
      {
        i >> data;
      }
      catch(...)
      {
        LOGE("Error : invalid project file %s\n", path.c_str());
        prmScene.projectToLoadFilename = "";
        return false;
      }
      i.close();

      loadingProject = true;

      // Initiate SplatSet loading
      if(!data["splats"].empty())
      {
        const auto& item             = data["splats"][0];
        prmScene.sceneToLoadFilename = makeAbsolutePath(std::filesystem::path(path).parent_path(), item["path"]);
        
        // Warn if project has multiple splat sets (only first will be loaded initially)
        if(data["splats"].size() > 1)
        {
          LOGW("Project contains %zu radiance fields. Only the first will be loaded initially. "
               "Use File > Add to load additional files.\n", data["splats"].size());
        }
      }
    }

    // Will do the rest of the work on next call when splatset is loaded
    return true;
  }

  // we skip until the splat set is being loaded
  if(m_splatLoader.getStatus() != SplatLoaderAsync::State::STATE_READY)
    return true;

  // we finalize
  guiAddToRecentProjects(prmScene.projectToLoadFilename);
  loadingProject                 = false;
  prmScene.projectToLoadFilename = "";

  try
  {
    // Renderer
    if(data.contains("renderer"))
    {
      const auto& item = data["renderer"];

      if(item.contains("vsync"))
        m_app->setVsync(item["vsync"]);

      LOAD1(prmSelectedPipeline, item, "pipeline");

      LOAD1(prmRender.maxShDegree, item, "maxShDegree");
      LOAD1(prmRender.opacityGaussianDisabled, item, "opacityGaussianDisabled");
      LOAD1(prmRender.showShOnly, item, "showShOnly");
      LOAD1(prmRender.visualize, item, "visualize");
      LOAD1(prmRender.wireframe, item, "wireframe");

      LOAD1(prmRaster.cpuLazySort, item, "cpuLazySort");
      LOAD1(prmRaster.distShaderWorkgroupSize, item, "distShaderWorkgroupSize");
      LOAD1(prmRaster.fragmentBarycentric, item, "fragmentBarycentric");
      LOAD1(prmRaster.frustumCulling, item, "frustumCulling");
      LOAD1(prmRaster.meshShaderWorkgroupSize, item, "meshShaderWorkgroupSize");
      LOAD1(prmRaster.pointCloudModeEnabled, item, "pointCloudModeEnabled");
      LOAD1(prmRaster.sortingMethod, item, "sortingMethod");

      LOAD1(prmRtx.temporalSampling, item, "temporalSampling");
      LOAD1(prmFrame.frameSampleMax, item, "temporalSamplesCount");
      LOAD1(prmRtx.kernelAdaptiveClamping, item, "kernelAdaptiveClamping");
      LOAD1(prmRtx.kernelDegree, item, "kernelDegree");
      LOAD1(prmRtx.kernelMinResponse, item, "kernelMinResponse");
      LOAD1(prmRtx.payloadArraySize, item, "payloadArraySize");
    }
    // Splat global options
    if(data.contains("splatsGlobals"))
    {
      const auto& item = data["splatsGlobals"];

      LOAD1(prmData.dataStorage, item, "dataStorage");
      LOAD1(prmData.shFormat, item, "shFormat");

      LOAD1(prmRtxData.compressBlas, item, "compressBlas");
      LOAD1(prmRtxData.useAABBs, item, "useAABBs");
      LOAD1(prmRtxData.useSpheres, item, "useSpheres");
      LOAD1(prmRtxData.useTlasInstances, item, "useTlasInstances");

      m_requestUpdateSplatData = true;
      m_requestUpdateSplatAs   = true;
    }
    // Parse splat settings
    if(data.contains("splats"))
    {
      if(!data["splats"].empty())
      {
        const auto& item = data["splats"][0];
        LOAD3(m_splatSetVk.translation, item, "position");
        LOAD3(m_splatSetVk.rotation, item, "rotation");
        LOAD3(m_splatSetVk.scale, item, "scale");

        computeTransform(m_splatSetVk.scale, m_splatSetVk.rotation, m_splatSetVk.translation, m_splatSetVk.transform,
                         m_splatSetVk.transformInverse);

        // delay update of Acceleration Structures if not using ray tracing
        m_requestDelayedUpdateSplatAs = true;
      }
    }

    // Load all the meshes
    if(data.contains("meshes"))
    {
      auto meshId = 0;
      for(const auto& item : data["meshes"])
      {
        std::string relPath;
        LOAD1(relPath, item, "path");
        if(relPath.empty())
          continue;

        auto meshPath = makeAbsolutePath(std::filesystem::path(path).parent_path(), relPath);
        if(!m_meshSetVk.loadModel(meshPath.string()))
        {
          meshId++;
          continue;
        }
        // Access to newly created mesh/instance
        auto& instance = m_meshSetVk.instances.back();
        auto& mesh     = m_meshSetVk.meshes[instance.objIndex];

        // Transform
        LOAD3(instance.translation, item, "position");
        LOAD3(instance.rotation, item, "rotation");
        LOAD3(instance.scale, item, "scale");
        computeTransform(instance.scale, instance.rotation, instance.translation, instance.transform, instance.transformInverse);

        // Materials
        if(item.contains("materials"))
        {
          auto matId = 0;
          for(const auto& matItem : item["materials"])
          {
            auto& mat = mesh.materials[matId];
            LOAD4_FROM3(mat.ambient, matItem, "ambient");
            LOAD4_FROM3(mat.diffuse, matItem, "diffuse");
            LOAD1(mat.illum, matItem, "illum");
            LOAD1(mat.ior, matItem, "ior");
            LOAD1(mat.shininess, matItem, "shininess");
            LOAD4_FROM3(mat.specular, matItem, "specular");
            LOAD4_FROM3(mat.transmittance, matItem, "transmittance");

            matId++;
          }
          m_meshSetVk.updateObjMaterialsBuffer(meshId);
        }

        meshId++;
      }
      m_requestUpdateMeshData = true;
      m_requestUpdateShaders  = true;
    }

    // Parse camera
    if(data.contains("camera"))
    {
      auto&  item = data["camera"];
      Camera cam;
      LOAD1(cam.model, item, "model");
      LOAD3(cam.ctr, item, "ctr");
      LOAD3(cam.eye, item, "eye");
      LOAD3(cam.up, item, "up");
      LOAD1(cam.fov, item, "fov");
      LOAD1(cam.dofEnabled, item, "dofEnabled");
      LOAD1(cam.focusDist, item, "focusDist");
      LOAD1(cam.aperture, item, "aperture");
      m_cameraSet.setCamera(cam);
    }
    // Parse camera presets
    if(data.contains("cameras"))
    {
      for(const auto& item : data["cameras"])
      {
        Camera cam;
        LOAD1(cam.model, item, "model");
        LOAD3(cam.ctr, item, "ctr");
        LOAD3(cam.eye, item, "eye");
        LOAD3(cam.up, item, "up");
        LOAD1(cam.fov, item, "fov");
        LOAD1(cam.dofEnabled, item, "dofEnabled");
        LOAD1(cam.focusDist, item, "focusDist");
        LOAD1(cam.aperture, item, "aperture");
        m_cameraSet.createPreset(cam);
      }
    }
    // Parse lights
    if(data.contains("lights"))
    {
      bool defaultLight = true;
      for(const auto& item : data["lights"])
      {
        // A default light already exists, we only modify it
        uint64_t id = 0;
        if(!defaultLight)
        {
          id = m_lightSet.createLight();
        }
        auto& light = m_lightSet.getLight(id);
        LOAD1(light.type, item, "type");
        LOAD3(light.position, item, "position");
        LOAD1(light.intensity, item, "intensity");
        defaultLight = false;
      }
      m_requestUpdateLightsBuffer = true;
    }

    return true;
  }
  catch(...)
  {
    return false;
  }
}

bool VkViewerUI::saveProject(std::string path)
{
  std::ofstream o(path);
  if(!o.is_open())
    return false;

  try
  {
    json data;

    // Renderer
    {
      json item;

      item["vsync"] = m_app->isVsync();

      item["pipeline"] = prmSelectedPipeline;

      item["maxShDegree"]             = prmRender.maxShDegree;
      item["opacityGaussianDisabled"] = prmRender.opacityGaussianDisabled;
      item["showShOnly"]              = prmRender.showShOnly;
      item["visualize"]               = prmRender.visualize;
      item["wireframe"]               = prmRender.wireframe;

      item["cpuLazySort"]             = prmRaster.cpuLazySort;
      item["distShaderWorkgroupSize"] = prmRaster.distShaderWorkgroupSize;
      item["fragmentBarycentric"]     = prmRaster.fragmentBarycentric;
      item["frustumCulling"]          = prmRaster.frustumCulling;
      item["meshShaderWorkgroupSize"] = prmRaster.meshShaderWorkgroupSize;
      item["pointCloudModeEnabled"]   = prmRaster.pointCloudModeEnabled;
      item["sortingMethod"]           = prmRaster.sortingMethod;

      item["temporalSampling"]       = prmRtx.temporalSampling;
      item["temporalSamplesCount"]   = prmFrame.frameSampleMax;
      item["kernelAdaptiveClamping"] = prmRtx.kernelAdaptiveClamping;
      item["kernelDegree"]           = prmRtx.kernelDegree;
      item["kernelMinResponse"]      = prmRtx.kernelMinResponse;
      item["payloadArraySize"]       = prmRtx.payloadArraySize;

      data["renderer"] = item;
    }

    // Active Camera
    {
      const auto& cam = m_cameraSet.getCamera();
      json        item;
      item["model"]      = cam.model;
      item["ctr"]        = {cam.ctr.x, cam.ctr.y, cam.ctr.z};
      item["eye"]        = {cam.eye.x, cam.eye.y, cam.eye.z};
      item["up"]         = {cam.up.x, cam.up.y, cam.up.z};
      item["fov"]        = cam.fov;
      item["dofEnabled"] = cam.dofEnabled;
      item["focusDist"]  = cam.focusDist;
      item["aperture"]   = cam.aperture;

      data["camera"] = item;
    }

    // Camera presets
    data["cameras"] = json::array();
    for(auto camId = 0; camId < m_cameraSet.size(); ++camId)
    {
      auto cam = m_cameraSet.getPreset(camId);

      json item;
      item["model"]      = cam.model;
      item["ctr"]        = {cam.ctr.x, cam.ctr.y, cam.ctr.z};
      item["eye"]        = {cam.eye.x, cam.eye.y, cam.eye.z};
      item["up"]         = {cam.up.x, cam.up.y, cam.up.z};
      item["fov"]        = cam.fov;
      item["dofEnabled"] = cam.dofEnabled;
      item["focusDist"]  = cam.focusDist;
      item["aperture"]   = cam.aperture;

      data["cameras"].push_back(item);
    }

    // Lights
    data["lights"] = json::array();
    for(auto lightId = 0; lightId < m_lightSet.numLights; ++lightId)
    {
      const auto& light = m_lightSet.getLight(lightId);

      json item;
      item["type"]      = light.type;
      item["position"]  = {light.position.x, light.position.y, light.position.z};
      item["intensity"] = light.intensity;

      data["lights"].push_back(item);
    }

    // Splat global options
    {
      json item;
      item["dataStorage"] = prmData.dataStorage;
      item["shFormat"]    = prmData.shFormat;

      item["compressBlas"]     = prmRtxData.compressBlas;
      item["useAABBs"]         = prmRtxData.useAABBs;
      item["useSpheres"]       = prmRtxData.useSpheres;
      item["useTlasInstances"] = prmRtxData.useTlasInstances;

      data["splatsGlobals"] = item;
    }

    // Splat sets - save all radiance fields
    data["splats"] = json::array();
    for(const auto& field : m_radianceFields)
    {
      json item;
      item["path"]     = getRelativePath(std::filesystem::path(path).parent_path(), field.filename);
      item["position"] = {m_splatSetVk.translation.x, m_splatSetVk.translation.y, m_splatSetVk.translation.z};
      item["rotation"] = {m_splatSetVk.rotation.x, m_splatSetVk.rotation.y, m_splatSetVk.rotation.z};
      item["scale"]    = {m_splatSetVk.scale.x, m_splatSetVk.scale.y, m_splatSetVk.scale.z};

      data["splats"].push_back(item);
    }

    // Meshes
    data["meshes"] = json::array();
    for(auto instId = 0; instId < m_meshSetVk.instances.size(); ++instId)
    {
      const auto& instance = m_meshSetVk.instances[instId];
      const auto& mesh     = m_meshSetVk.meshes[instance.objIndex];

      json item;
      item["path"] = getRelativePath(std::filesystem::path(path).parent_path(), mesh.path);
      item["name"] = mesh.name;

      // Transform
      item["position"] = {instance.translation.x, instance.translation.y, instance.translation.z};
      item["rotation"] = {instance.rotation.x, instance.rotation.y, instance.rotation.z};
      item["scale"]    = {instance.scale.x, instance.scale.y, instance.scale.z};

      // Material override
      item["materials"] = json::array();

      for(auto matId = 0; matId < mesh.matNames.size(); ++matId)
      {
        json matItem;

        const auto& name = mesh.matNames[matId];
        const auto& mat  = mesh.materials[matId];

        matItem["name"]          = name;
        matItem["ambient"]       = {mat.ambient.x, mat.ambient.y, mat.ambient.z};
        matItem["diffuse"]       = {mat.diffuse.x, mat.diffuse.y, mat.diffuse.z};
        matItem["illum"]         = mat.illum;
        matItem["ior"]           = mat.ior;
        matItem["shininess"]     = mat.shininess;
        matItem["specular"]      = {mat.specular.x, mat.specular.y, mat.specular.z};
        matItem["transmittance"] = {mat.transmittance.x, mat.transmittance.y, mat.transmittance.z};

        item["materials"].push_back(matItem);
      }

      data["meshes"].push_back(item);
    }

    o << std::setw(4) << data << std::endl;
    o.close();
    return true;
  }
  catch(...)
  {
    return false;
  }
}

#undef LOAD1
#undef LOAD3
#undef LOAD4_FROM3
