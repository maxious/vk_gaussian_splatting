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

#include "parameters.h"
#include <nvutils/logger.hpp>

namespace vk_viewer {

// no reset function on purpose
SceneParameters prmScene{};

VramDataParameters    prmData{};
RtxVramDataParameters prmRtxData{};

// no reset function on purpose
uint32_t            prmSelectedPipeline = PIPELINE_MESH;
shaderio::FrameInfo prmFrame{};
RenderParameters    prmRender{};
RasterParameters       prmRaster{};
RtxParameters          prmRtx{};
StochasticParameters   prmStochastic{};

// Storage for respective default values

static VramDataParameters    prmDataDefault{};
static RtxVramDataParameters prmRtxDataDefault{};

static shaderio::FrameInfo prmFrameDefault{};
static RenderParameters    prmRenderDefault{};
static RasterParameters    prmRasterDefault{};
static RtxParameters       prmRtxDefault{};

void storeDefaultParameters()
{
  prmDataDefault    = prmData;
  prmRtxDataDefault = prmRtxData;

  prmFrameDefault  = prmFrame;
  prmRenderDefault = prmRender;
  prmRasterDefault = prmRaster;
  prmRtxDefault    = prmRtx;
}

void resetDataParameters()
{
  prmData = prmDataDefault;
}
void resetRtxDataParameters()
{
  prmRtxData = prmRtxDataDefault;
}
void resetFrameParameters()
{
  prmFrame = prmFrameDefault;
}
void resetRenderParameters()
{
  prmRender = prmRenderDefault;
}
void resetRasterParameters()
{
  prmRaster = prmRasterDefault;
}
void resetRtxParameters()
{
  prmRtx = prmRtxDefault;
}

void registerCommandLineParameters(nvutils::ParameterRegistry* parameterRegistry)
{
  // Scene - unified input that auto-detects based on extension
  parameterRegistry->add({"inputFile", "load a scene file (ply, spz, rad, sog, lod-meta.json, 4dv, obj, glb, gltf, or metadata.json for depth video)"},
                         {".ply", ".spz", ".rad", ".sog", ".4dv", ".obj", ".glb", ".gltf", ".json"}, &prmScene.sceneToLoadFilename);
  parameterRegistry->add({"inputMesh", "load a mesh file (obj, glb, gltf)"}, {".obj", ".glb", ".gltf"}, &prmScene.meshToImportFilename);
#ifdef WITH_DEFAULT_SCENE_FEATURE
  parameterRegistry->add({"loadDefaultScene", "0=disable the load of a default scene when no ply file is provided"},
                         &prmScene.enableDefaultScene);
#endif
  // Projects
  parameterRegistry->add({"inputProject", "load a vkgs project file"}, {".vkgs"}, &prmScene.projectToLoadFilename);

  // Data
  parameterRegistry->add({"shformat", "0=fp32 1=fp16 2=uint8"}, &prmData.shFormat);
  parameterRegistry->add({"useAABBs", "0=use icosahedron 3D mesh and built-in triangle/ray intersection (Default), 1=use AABBs and parametric intersection shader. DO NOT COMBINE with useTlasInstances=0."},
                         &prmRtxData.useAABBs);
  parameterRegistry->add({"useTlasInstances", "1=use one TLAS instance per particle and a small unit particle BLAS (default). 0=use one TLAS entry and a large BLAS."},
                         &prmRtxData.useTlasInstances);
  parameterRegistry->add({"compressBlas", "1=compress BLAS (default). 0=diabled."}, &prmRtxData.compressBlas);

  // Pipelines
  parameterRegistry->add({"pipeline", "0=3dgs-vert 1=3dgs-mesh(default) 2=3dgrt 3=hybrid-3dgs 4=3dgut 5=hybrid-3dgut 6=stochastic-gs"},
                         &prmSelectedPipeline);
  parameterRegistry->add({"maxShDegree", "max sh degree used for rendering in [0,1,2,3]"}, &prmRender.maxShDegree);
  parameterRegistry->add({"extentProjection", "particle extent projection method [0=Eigen (default),1=Conic]"},
                         &prmRaster.extentProjection);
  parameterRegistry->add({"kernelDegree", "kernel degree used by 3DGRT, 3DGUT and Hybrid 3DGUT pipelines in [0,1,2(default),3,4,5]"},
                         &prmRtx.kernelDegree);

  // Stochastic GS
  parameterRegistry->add({"stochasticSamplesPerPixel", "samples per pixel for stochastic GS (1=interactive, 64=converged)"},
                         &prmStochastic.stochasticSamplesPerPixel);
  parameterRegistry->add({"stochasticMaxSamples", "max accumulation samples before auto-reset"},
                         &prmStochastic.stochasticMaxSamples);
  parameterRegistry->add({"stochasticSupersamplingFactor", "supersampling factor (1, 2, 4)"},
                         &prmStochastic.stochasticSupersamplingFactor);
  parameterRegistry->add({"stochasticUseGps", "0=ST, 1=GPS mode"},
                         &prmStochastic.stochasticUseGps);
  parameterRegistry->add({"stochasticEnableDof", "0=no DOF, 1=DOF enabled"},
                         &prmStochastic.stochasticEnableDof);

  // Scene loading options
  parameterRegistry->add({"mortonReorder", "1=reorder splats using Morton/Z-order curve for cache coherency (default), 0=disabled"},
                         &prmScene.mortonReorder);

  // Migration guard: if a saved config has pipeline > HYBRID_3DGUT (e.g. stochastic-gs),
  // reset to mesh until the pipeline is fully wired
  if (prmSelectedPipeline > PIPELINE_HYBRID_3DGUT)
  {
    LOGW("Reset prmSelectedPipeline from %d to PIPELINE_MESH (%d)\n", (int)prmSelectedPipeline, (int)PIPELINE_MESH);
    prmSelectedPipeline = PIPELINE_MESH;
  }
}

}  // namespace vk_viewer
