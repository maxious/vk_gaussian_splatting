/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sogxt_loader.h"

#include <fstream>
#include <cmath>
#include <algorithm>
#include <future>
#include <cstring>

#include <nvutils/logger.hpp>
#include <nvutils/parallel_work.hpp>
#include <tinygltf/json.hpp>

#include <webp/decode.h>

using nlohmann::json;
using namespace vk_viewer;

namespace {

// Read an entire file into a byte vector
std::vector<uint8_t> readFile(const std::filesystem::path& path)
{
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if(!file)
    return {};
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<uint8_t> buffer(size);
  if(!file.read(reinterpret_cast<char*>(buffer.data()), size))
    return {};
  return buffer;
}

// Inverse signed-log transform: reverses sign(x) * ln(|x| + 1)
inline float invSignedLog(float v)
{
  float a = std::abs(v);
  float e = std::exp(a) - 1.0f;
  return v < 0.0f ? -e : e;
}

// Per-channel observed min/max over the first `channels` RGBA channels.
// Values are read from `rgba` for pixel row-major index `i` (stride 4).
inline void channelMinMax(const std::vector<uint8_t>& rgba, uint32_t width, uint32_t height,
                          uint32_t channels, uint32_t gridSide, float& gmin, float& gmax)
{
  gmin = std::numeric_limits<float>::max();
  gmax = std::numeric_limits<float>::lowest();
  const uint32_t pixels = width * height;
  for(uint32_t i = 0; i < pixels; ++i)
  {
    for(uint32_t c = 0; c < channels; ++c)
    {
      const float v = static_cast<float>(rgba[i * 4 + c]);
      gmin = std::min(gmin, v);
      gmax = std::max(gmax, v);
    }
  }
  if(gmax - gmin < 1e-8f)
    gmax = gmin + 1.0f;
}

// dequantize: normalize observed [gmin,gmax] -> [0,1], then affine to [min,max].
inline float dequantizeScalar(uint8_t v, float gmin, float gmax, float vmin, float vmax)
{
  const float norm = (static_cast<float>(v) - gmin) / (gmax - gmin);
  return norm * (vmax - vmin) + vmin;
}

}  // namespace

bool SogXtLoader::parseMeta(const std::vector<uint8_t>& jsonData, SogXtMeta& meta)
{
  try
  {
    json j = json::parse(jsonData.begin(), jsonData.end());

    meta.version  = j.value("version", 0u);
    meta.count    = j.value("count", 0u);
    meta.gridSide = j.value("gridSide", 0u);

    if(j.value("format", std::string("")) != "sog-xt")
    {
      LOGE("Not a SOG-XT container (missing format: sog-xt)\n");
      return false;
    }

    auto parseField = [](const json& j, FieldInfo& info) {
      if(j.contains("mins"))
      {
        if(j["mins"].is_number())
          info.mins.push_back(j["mins"].get<float>());
        else
          info.mins = j["mins"].get<std::vector<float>>();
      }
      if(j.contains("maxs"))
      {
        if(j["maxs"].is_number())
          info.maxs.push_back(j["maxs"].get<float>());
        else
          info.maxs = j["maxs"].get<std::vector<float>>();
      }
      if(j.contains("files"))
        info.files = j["files"].get<std::vector<std::string>>();
      info.normalize = j.value("normalize", std::string("observed-minmax"));
      info.encoding  = j.value("encoding", std::string(""));
    };

    if(j.contains("mask"))
      parseField(j["mask"], meta.mask);
    if(j.contains("means"))
      parseField(j["means"], meta.means);
    if(j.contains("opacities"))
      parseField(j["opacities"], meta.opacities);
    if(j.contains("scales"))
      parseField(j["scales"], meta.scales);
    if(j.contains("quats"))
      parseField(j["quats"], meta.quats);
    if(j.contains("sh0"))
      parseField(j["sh0"], meta.sh0);

    if(j.contains("shN"))
    {
      auto& shN     = j["shN"];
      meta.shN.coeffs       = shN.value("coeffs", 15u);
      meta.shN.centroidSide = shN.value("centroidSide", 0u);
      meta.shN.tileRows     = shN.value("tileRows", 3u);
      meta.shN.tileCols     = shN.value("tileCols", 5u);
      if(shN.contains("centroidsMins"))
        meta.shN.centroidsMins = shN["centroidsMins"].get<std::vector<float>>();
      if(shN.contains("centroidsMaxs"))
        meta.shN.centroidsMaxs = shN["centroidsMaxs"].get<std::vector<float>>();
      if(shN.contains("files"))
        meta.shN.files = shN["files"].get<std::vector<std::string>>();
    }

    if(meta.gridSide == 0 || meta.means.files.size() < 2)
    {
      LOGE("SOG-XT meta missing gridSide or means planes\n");
      return false;
    }
    return true;
  }
  catch(const std::exception& e)
  {
    LOGE("Failed to parse SOG-XT meta.json: %s\n", e.what());
    return false;
  }
}

bool SogXtLoader::decodeWebP(const std::vector<uint8_t>& webpData, WebPImage& output)
{
  WebPDecoderConfig config;
  if(!WebPInitDecoderConfig(&config))
  {
    LOGE("Failed to initialize WebP decoder config\n");
    return false;
  }

  if(WebPGetFeatures(webpData.data(), webpData.size(), &config.input) != VP8_STATUS_OK)
  {
    LOGE("Failed to get WebP image features\n");
    return false;
  }

  output.width  = static_cast<uint32_t>(config.input.width);
  output.height = static_cast<uint32_t>(config.input.height);
  output.rgba.resize(output.width * output.height * 4);

  config.output.colorspace         = MODE_RGBA;
  config.output.u.RGBA.rgba        = output.rgba.data();
  config.output.u.RGBA.stride      = output.width * 4;
  config.output.u.RGBA.size        = output.rgba.size();
  config.output.is_external_memory = 1;
  config.options.use_threads       = 1;

  VP8StatusCode status = WebPDecode(webpData.data(), webpData.size(), &config);
  WebPFreeDecBuffer(&config.output);

  if(status != VP8_STATUS_OK)
  {
    LOGE("Failed to decode WebP image (status %d)\n", status);
    return false;
  }
  return true;
}

void SogXtLoader::decodeMask(const WebPImage& mask, uint32_t gridSide, SplatSet& output,
                             std::vector<bool>& active)
{
  active.assign(gridSide * gridSide, false);
  const uint32_t pixels = std::min<uint32_t>(mask.width * mask.height, gridSide * gridSide);
  for(uint32_t i = 0; i < pixels; ++i)
  {
    active[i] = mask.rgba[i * 4 + 0] > 0;
  }
}

void SogXtLoader::decodeMeans(const WebPImage& low, const WebPImage& high, const SogXtMeta& meta,
                              SplatSet& output)
{
  const uint32_t gridSide = meta.gridSide;
  const uint32_t pixels   = gridSide * gridSide;

  // Reference decoder: combined = low + 256*high (16-bit), observed global
  // min/max over the whole plane, then scalar affine, then inverse signed log.
  std::vector<uint32_t> combined(pixels * 3);
  uint32_t gmin = std::numeric_limits<uint32_t>::max();
  uint32_t gmax = 0;
  for(uint32_t i = 0; i < pixels; ++i)
  {
    for(uint32_t c = 0; c < 3; ++c)
    {
      uint32_t v = low.rgba[i * 4 + c] + 256u * high.rgba[i * 4 + c];
      combined[i * 3 + c] = v;
      gmin = std::min(gmin, v);
      gmax = std::max(gmax, v);
    }
  }
  if(gmax == gmin)
    gmax = gmin + 1;

  const float vmin = meta.means.mins.empty() ? 0.0f : meta.means.mins[0];
  const float vmax = meta.means.maxs.empty() ? 1.0f : meta.means.maxs[0];

  output.positions.resize(pixels * 3);
  nvutils::parallel_ranges_pooled<1024>(pixels, [&](uint64_t start, uint64_t end, uint32_t threadIdx) {
    for(uint64_t i = start; i < end; ++i)
    {
      for(uint32_t c = 0; c < 3; ++c)
      {
        const float norm = (static_cast<float>(combined[i * 3 + c]) - static_cast<float>(gmin)) /
                           (static_cast<float>(gmax) - static_cast<float>(gmin));
        const float scaled = norm * (vmax - vmin) + vmin;
        output.positions[i * 3 + c] = invSignedLog(scaled);
      }
    }
  });
}

void SogXtLoader::decodeOpacities(const WebPImage& img, const FieldInfo& info, uint32_t gridSide,
                                  SplatSet& output)
{
  const uint32_t pixels = gridSide * gridSide;
  float          gmin, gmax;
  channelMinMax(img.rgba, img.width, img.height, 1, gridSide, gmin, gmax);
  const float vmin = info.mins.empty() ? 0.0f : info.mins[0];
  const float vmax = info.maxs.empty() ? 1.0f : info.maxs[0];

  output.opacity.resize(pixels);
  nvutils::parallel_ranges_pooled<1024>(pixels, [&](uint64_t start, uint64_t end, uint32_t threadIdx) {
    for(uint64_t i = start; i < end; ++i)
    {
      // Container stores opacity in logit space; SplatSet stores logit too.
      output.opacity[i] = dequantizeScalar(img.rgba[i * 4 + 0], gmin, gmax, vmin, vmax);
    }
  });
}

void SogXtLoader::decodeScales(const WebPImage& img, const FieldInfo& info, uint32_t gridSide,
                               SplatSet& output)
{
  const uint32_t pixels = gridSide * gridSide;
  float          gmin, gmax;
  channelMinMax(img.rgba, img.width, img.height, 3, gridSide, gmin, gmax);

  output.scale.resize(pixels * 3);
  nvutils::parallel_ranges_pooled<1024>(pixels, [&](uint64_t start, uint64_t end, uint32_t threadIdx) {
    for(uint64_t i = start; i < end; ++i)
    {
      for(uint32_t c = 0; c < 3; ++c)
      {
        const float vmin = info.mins.empty() ? 0.0f : info.mins[std::min(c, (uint32_t)info.mins.size() - 1)];
        const float vmax = info.maxs.empty() ? 1.0f : info.maxs[std::min(c, (uint32_t)info.maxs.size() - 1)];
        // Container stores scale in log space; SplatSet stores log space too.
        output.scale[i * 3 + c] = dequantizeScalar(img.rgba[i * 4 + c], gmin, gmax, vmin, vmax);
      }
    }
  });
}

void SogXtLoader::decodeQuats(const WebPImage& img, const FieldInfo& info, uint32_t gridSide,
                              SplatSet& output)
{
  const uint32_t pixels = gridSide * gridSide;
  float          gmin, gmax;
  channelMinMax(img.rgba, img.width, img.height, 4, gridSide, gmin, gmax);

  output.rotation.resize(pixels * 4);
  nvutils::parallel_ranges_pooled<1024>(pixels, [&](uint64_t start, uint64_t end, uint32_t threadIdx) {
    for(uint64_t i = start; i < end; ++i)
    {
      for(uint32_t c = 0; c < 4; ++c)
      {
        const float vmin = info.mins.empty() ? 0.0f : info.mins[std::min(c, (uint32_t)info.mins.size() - 1)];
        const float vmax = info.maxs.empty() ? 1.0f : info.maxs[std::min(c, (uint32_t)info.maxs.size() - 1)];
        output.rotation[i * 4 + c] = dequantizeScalar(img.rgba[i * 4 + c], gmin, gmax, vmin, vmax);
      }
    }
  });
}

void SogXtLoader::decodeSh0(const WebPImage& img, const FieldInfo& info, uint32_t gridSide,
                            SplatSet& output)
{
  const uint32_t pixels = gridSide * gridSide;
  float          gmin, gmax;
  channelMinMax(img.rgba, img.width, img.height, 3, gridSide, gmin, gmax);

  output.f_dc.resize(pixels * 3);
  nvutils::parallel_ranges_pooled<1024>(pixels, [&](uint64_t start, uint64_t end, uint32_t threadIdx) {
    for(uint64_t i = start; i < end; ++i)
    {
      for(uint32_t c = 0; c < 3; ++c)
      {
        const float vmin = info.mins.empty() ? 0.0f : info.mins[std::min(c, (uint32_t)info.mins.size() - 1)];
        const float vmax = info.maxs.empty() ? 1.0f : info.maxs[std::min(c, (uint32_t)info.maxs.size() - 1)];
        output.f_dc[i * 3 + c] = dequantizeScalar(img.rgba[i * 4 + c], gmin, gmax, vmin, vmax);
      }
    }
  });
}

void SogXtLoader::decodeShN(const WebPImage& centroidsImg, const WebPImage& labelsImg,
                            const ShNInfo& shN, uint32_t gridSide, SplatSet& output)
{
  if(shN.centroidSide == 0)
    return;

  const uint32_t pixels      = gridSide * gridSide;
  const uint32_t centroidSide = shN.centroidSide;
  const uint32_t tileRows    = std::max(shN.tileRows, 1u);
  const uint32_t tileCols    = std::max(shN.tileCols, 1u);
  const uint32_t numTiles    = tileRows * tileCols;   // 15
  const uint32_t perTileCh   = 3;                      // RGB
  const uint32_t channels45  = perTileCh * numTiles;   // 45

  // ---- Labels: index = v * centroidSide + u, channel 0 = u, channel 1 = v
  std::vector<uint32_t> labels(pixels);
  for(uint32_t i = 0; i < pixels; ++i)
  {
    const uint32_t u = labelsImg.rgba[i * 4 + 0];
    const uint32_t v = labelsImg.rgba[i * 4 + 1];
    labels[i] = v * centroidSide + u;
  }

  // ---- Centroids: tiled (tileRows, tileCols) grid of RGB tiles.
  // Observed global min/max over the whole tiled image, then per-channel affine.
  const uint32_t cenPixels = centroidsImg.width * centroidsImg.height;
  float          gmin = std::numeric_limits<float>::max();
  float          gmax = std::numeric_limits<float>::lowest();
  for(uint32_t i = 0; i < cenPixels; ++i)
  {
    for(uint32_t c = 0; c < 3; ++c)
    {
      const float v = static_cast<float>(centroidsImg.rgba[i * 4 + c]);
      gmin = std::min(gmin, v);
      gmax = std::max(gmax, v);
    }
  }
  if(gmax - gmin < 1e-8f)
    gmax = gmin + 1.0f;

  const uint32_t tileH = centroidsImg.height / tileRows;
  const uint32_t tileW = centroidsImg.width / tileCols;

  auto centroidAt = [&](uint32_t u, uint32_t v, uint32_t ch, uint32_t tr, uint32_t tc) -> float {
    // Tiled image: tile (tr, tc) holds RGB for SH coefficient (tr*tileCols + tc).
    const uint32_t px = (tr * tileH + v) * centroidsImg.width + (tc * tileW + u);
    const uint8_t  byte = centroidsImg.rgba[px * 4 + ch];
    return dequantizeScalar(byte, gmin, gmax, 0.0f, 1.0f);
  };

  // Output channel ordering matches the reference untile:
  // out[c] with c = ch * numTiles + (tr * tileCols + tc).
  output.f_rest.resize(pixels * channels45);
  nvutils::parallel_ranges_pooled<512>(pixels, [&](uint64_t start, uint64_t end, uint32_t threadIdx) {
    for(uint64_t i = start; i < end; ++i)
    {
      const uint32_t centroidIdx = labels[i];
      if(centroidIdx >= centroidSide * centroidSide)
      {
        std::fill(output.f_rest.begin() + i * channels45, output.f_rest.begin() + (i + 1) * channels45, 0.0f);
        continue;
      }
      const uint32_t cu = centroidIdx % centroidSide;
      const uint32_t cv = centroidIdx / centroidSide;
      for(uint32_t c = 0; c < channels45; ++c)
      {
        const uint32_t ch  = c / numTiles;
        const uint32_t rem = c % numTiles;
        const uint32_t tr  = rem / tileCols;
        const uint32_t tc  = rem % tileCols;
        const float    norm = centroidAt(cu, cv, ch, tr, tc);  // [0,1]
        const float    vmin = shN.centroidsMins.empty() ? 0.0f : shN.centroidsMins[c];
        const float    vmax = shN.centroidsMaxs.empty() ? 1.0f : shN.centroidsMaxs[c];
        output.f_rest[i * channels45 + c] = norm * (vmax - vmin) + vmin;
      }
    }
  });
}

void SogXtLoader::applyActiveMask(SplatSet& output, const std::vector<bool>& active)
{
  const uint32_t gridSide = static_cast<uint32_t>(std::sqrt((double)active.size()));
  const uint32_t cells    = gridSide * gridSide;

  // Count active
  uint32_t count = 0;
  for(bool a : active)
    count += a ? 1 : 0;

  std::vector<float> newPos(count * 3), newFDc(count * 3), newFRest(count * 45),
      newOpacity(count), newScale(count * 3), newRotation(count * 4);

  const size_t shPerSplat = output.f_rest.size() / std::max<size_t>(cells, 1);
  const size_t restCh     = std::min<size_t>(45, shPerSplat);

  uint32_t idx = 0;
  for(uint32_t i = 0; i < cells; ++i)
  {
    if(!active[i])
      continue;
    for(uint32_t c = 0; c < 3; ++c)
    {
      newPos[idx * 3 + c]      = output.positions[i * 3 + c];
      newFDc[idx * 3 + c]      = output.f_dc[i * 3 + c];
      newScale[idx * 3 + c]    = output.scale[i * 3 + c];
      newRotation[idx * 4 + c] = output.rotation[i * 4 + c];
    }
    newRotation[idx * 4 + 3] = output.rotation[i * 4 + 3];
    newOpacity[idx]          = output.opacity[i];
    for(size_t c = 0; c < restCh; ++c)
      newFRest[idx * 45 + c] = output.f_rest[i * restCh + c];
    ++idx;
  }

  output.positions = std::move(newPos);
  output.f_dc      = std::move(newFDc);
  output.f_rest    = std::move(newFRest);
  output.opacity   = std::move(newOpacity);
  output.scale     = std::move(newScale);
  output.rotation  = std::move(newRotation);
}

bool SogXtLoader::loadWithReader(const SogXtMeta& meta, FileReader reader, SplatSet& output,
                                 std::function<void(float)> progressCallback)
{
  const uint32_t gridSide = meta.gridSide;
  if(gridSide == 0)
  {
    LOGE("SOG-XT container has no gridSide\n");
    return false;
  }

  if(progressCallback)
    progressCallback(0.1f);

  auto decodeFile = [&](const std::string& filename) -> WebPImage {
    WebPImage img;
    if(!filename.empty())
      decodeWebP(reader(filename), img);
    return img;
  };

  std::future<WebPImage> maskFut = std::async(std::launch::async, [&]() {
    return decodeFile(meta.mask.files.empty() ? "" : meta.mask.files[0]);
  });
  std::future<WebPImage> meansLowFut = std::async(std::launch::async, [&]() {
    return decodeFile(meta.means.files.size() > 0 ? meta.means.files[0] : "");
  });
  std::future<WebPImage> meansHighFut = std::async(std::launch::async, [&]() {
    return decodeFile(meta.means.files.size() > 1 ? meta.means.files[1] : "");
  });
  std::future<WebPImage> opacitiesFut = std::async(std::launch::async, [&]() {
    return decodeFile(meta.opacities.files.empty() ? "" : meta.opacities.files[0]);
  });
  std::future<WebPImage> scalesFut = std::async(std::launch::async, [&]() {
    return decodeFile(meta.scales.files.empty() ? "" : meta.scales.files[0]);
  });
  std::future<WebPImage> quatsFut = std::async(std::launch::async, [&]() {
    return decodeFile(meta.quats.files.empty() ? "" : meta.quats.files[0]);
  });
  std::future<WebPImage> sh0Fut = std::async(std::launch::async, [&]() {
    return decodeFile(meta.sh0.files.empty() ? "" : meta.sh0.files[0]);
  });
  std::future<WebPImage> shNCenFut = std::async(std::launch::async, [&]() {
    return decodeFile(meta.shN.files.size() > 0 ? meta.shN.files[0] : "");
  });
  std::future<WebPImage> shNLabFut = std::async(std::launch::async, [&]() {
    return decodeFile(meta.shN.files.size() > 1 ? meta.shN.files[1] : "");
  });

  std::vector<bool> active;
  decodeMask(maskFut.get(), gridSide, output, active);
  if(progressCallback)
    progressCallback(0.25f);

  WebPImage low = meansLowFut.get();
  WebPImage high = meansHighFut.get();
  if(!low.rgba.empty() && !high.rgba.empty())
    decodeMeans(low, high, meta, output);
  if(progressCallback)
    progressCallback(0.4f);

  WebPImage opacities = opacitiesFut.get();
  if(!opacities.rgba.empty())
    decodeOpacities(opacities, meta.opacities, gridSide, output);
  if(progressCallback)
    progressCallback(0.55f);

  WebPImage scales = scalesFut.get();
  if(!scales.rgba.empty())
    decodeScales(scales, meta.scales, gridSide, output);
  if(progressCallback)
    progressCallback(0.65f);

  WebPImage quats = quatsFut.get();
  if(!quats.rgba.empty())
    decodeQuats(quats, meta.quats, gridSide, output);
  if(progressCallback)
    progressCallback(0.75f);

  WebPImage sh0 = sh0Fut.get();
  if(!sh0.rgba.empty())
    decodeSh0(sh0, meta.sh0, gridSide, output);
  if(progressCallback)
    progressCallback(0.85f);

  WebPImage shNCen = shNCenFut.get();
  WebPImage shNLab = shNLabFut.get();
  if(!shNCen.rgba.empty() && !shNLab.rgba.empty())
    decodeShN(shNCen, shNLab, meta.shN, gridSide, output);
  if(progressCallback)
    progressCallback(0.95f);

  applyActiveMask(output, active);

  // Convert coordinate system from RDF (INRIA PLY convention, same as SOG) to
  // the viewer's RUB convention.
  output.convertCoordinates(spz::CoordinateSystem::RDF, spz::CoordinateSystem::RUB);

  if(progressCallback)
    progressCallback(1.0f);

  LOGI("Loaded SOG-XT container: %zu splats (grid %ux%u)\n", output.size(), gridSide, gridSide);
  return true;
}

bool SogXtLoader::isSogXtManifest(const std::filesystem::path& path)
{
  // Directory container
  if(std::filesystem::is_directory(path))
  {
    std::vector<uint8_t> dirMeta = readFile(path / "meta.json");
    if(dirMeta.empty())
      return false;
    SogXtMeta meta;
    return parseMeta(dirMeta, meta);
  }

  const std::string fname = path.filename().string();
  std::filesystem::path metaPath = path;

  if(fname == "scene.json")
  {
    // Resolve the metadata url from the manifest.
    std::vector<uint8_t> manifest = readFile(path);
    if(manifest.empty())
      return false;
    try
    {
      json scene = json::parse(manifest.begin(), manifest.end());
      std::string metaUrl = scene.value("metadata", json::object()).value("url", std::string("meta.json"));
      metaPath = path.parent_path() / metaUrl;
    }
    catch(const std::exception&)
    {
      return false;
    }
  }

  std::vector<uint8_t> metaData = readFile(metaPath);
  if(metaData.empty())
    return false;
  SogXtMeta meta;
  return parseMeta(metaData, meta);
}

bool SogXtLoader::load(const std::filesystem::path& filename, SplatSet& output,
                       std::function<void(float)> progressCallback)
{
  // Resolve the container directory + meta.json from various inputs.
  std::filesystem::path containerDir;
  std::filesystem::path metaPath;

  if(std::filesystem::is_directory(filename))
  {
    containerDir = filename;
    metaPath     = containerDir / "meta.json";
  }
  else
  {
    containerDir = filename.parent_path();
    const std::string fname = filename.filename().string();
    if(fname == "scene.json")
    {
      // Resolve the metadata url from the manifest.
      std::vector<uint8_t> manifest = readFile(filename);
      if(manifest.empty())
        return false;
      try
      {
        json scene = json::parse(manifest.begin(), manifest.end());
        std::string metaUrl = scene.value("metadata", json::object()).value("url", std::string("meta.json"));
        metaPath = containerDir / metaUrl;
      }
      catch(const std::exception& e)
      {
        LOGE("Failed to parse scene.json: %s\n", e.what());
        return false;
      }
    }
    else
    {
      metaPath = filename;
    }
  }

  std::vector<uint8_t> metaData = readFile(metaPath);
  if(metaData.empty())
  {
    if(std::filesystem::is_directory(filename))
    {
      return false;  // Not a SOG-XT container; caller falls through to other loaders.
    }
    LOGE("Failed to read SOG-XT meta.json: %s\n", metaPath.string().c_str());
    return false;
  }

  SogXtMeta meta;
  if(!parseMeta(metaData, meta))
    return false;

  FileReader reader = [&containerDir](const std::string& name) -> std::vector<uint8_t> {
    return readFile(containerDir / name);
  };

  return loadWithReader(meta, reader, output, progressCallback);
}