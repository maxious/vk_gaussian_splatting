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

//
#include <functional>
#include <fstream>
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>

#include <nvutils/logger.hpp>

#include "splat_loader_fast.h"
// 3rd party spz library
#include "load-spz.h"

//
#include "splat_loader_async.h"
#include "sog_loader.h"
#include "sogxt_loader.h"
#include "supersplat_client.h"
#include "lod_loader.h"
#include "fourdv_loader.h"
#include "npz_loader.h"
#include "lcc_loader.h"
#include "rad_loader.h"
#include "utilities.h"

#include <ixwebsocket/IXHttpClient.h>
#include <ixwebsocket/IXNetSystem.h>

using namespace vk_viewer;

// Type alias for SogMeta used in SuperSplat download
using SogMeta = SogLoader::SogMeta;

// Helper to extract ID from SuperSplat URL
std::string extractSuperSplatId(const std::string& url)
{
  // Support formats:
  // https://superspl.at/view?id=bd964899
  // https://superspl.at/s?id=bd964899

  std::string idKey = "?id=";
  size_t      pos   = url.find(idKey);
  if(pos == std::string::npos)
    return "";

  std::string id = url.substr(pos + idKey.length());
  // Truncate at next parameter if any
  size_t endPos = id.find('&');
  if(endPos != std::string::npos)
  {
    id = id.substr(0, endPos);
  }
  return id;
}

// Helper to read file into vector
std::vector<uint8_t> readFileLocal(const std::filesystem::path& path)
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

namespace {
using ProgressCallback = std::function<void(size_t downloaded, size_t total)>;

bool downloadFile(const std::string& url, const std::filesystem::path& destPath, ProgressCallback callback = nullptr)
{
  ix::HttpClient httpClient;
  auto           args = httpClient.createRequest(url, ix::HttpClient::kGet);

  std::ofstream outFile(destPath, std::ios::binary);
  if(!outFile)
  {
    LOGE("Failed to create file: %s\n", destPath.string().c_str());
    return false;
  }

  args->onChunkCallback = [&](const std::string& chunk) {
    outFile.write(chunk.data(), chunk.size());
    return true;
  };

  if(callback)
  {
    args->onProgressCallback = [&](size_t downloaded, size_t total) {
      callback(downloaded, total);
      return true;
    };
  }

  auto res = httpClient.get(url, args);
  outFile.close();

  if(res->errorCode != ix::HttpErrorCode::Ok)
  {
    LOGE("Download failed: %s\n", res->errorMsg.c_str());
    return false;
  }

  if(res->statusCode != 200)
  {
    LOGE("Download failed: HTTP %d\n", res->statusCode);
    return false;
  }

  return true;
}
}  // namespace

bool SplatLoaderAsync::loadScene(std::filesystem::path filename, SplatSet& output)
{
  std::lock_guard<std::mutex> lock(m_mutex);
  if(m_status != STATE_READY)
  {
    return false;
  }

  // setup load info and wakeup the thread
  m_filename = filename;
  m_output   = &output;
  m_loadCV.notify_all();

  return true;
}

bool SplatLoaderAsync::loadSceneAtLod(std::filesystem::path filename, SplatSet& output, int lodLevel)
{
  std::lock_guard<std::mutex> lock(m_mutex);
  if(m_status != STATE_READY)
  {
    return false;
  }

  if(!LccLoader::canLoad(filename))
  {
    return false;
  }

  // setup load info and wakeup the thread
  m_filename         = filename;
  m_output           = &output;
  m_targetLod        = lodLevel;
  m_lodReloadPending = true;
  m_loadCV.notify_all();

  return true;
}

bool SplatLoaderAsync::initialize()
{
  ix::initNetSystem();

  // original state shall be shutdown
  std::unique_lock<std::mutex> lock(m_mutex);
  if(m_status != STATE_SHUTDOWN)
    return false;  // will unlock through lock destructor
  else
    lock.unlock();

  // starts the thread
  m_loader = std::thread([this]() {
    //
    std::unique_lock<std::mutex> lock(m_mutex);
    m_status = STATE_READY;
    lock.unlock();
    //
    while(true)
    {
      // wait to load new scene
      std::unique_lock<std::mutex> lock(m_mutex);
      m_loadCV.wait(lock, [this] { return m_shutdownRequested || m_output != nullptr; });
      bool shutdown = m_shutdownRequested;
      lock.unlock();
      // if request is not a shutdown do the job
      if(!shutdown)
      {
        // let's load
        std::unique_lock<std::mutex> lock(m_mutex);
        m_status = STATE_LOADING;
        lock.unlock();
        if(m_output != nullptr && innerLoad(m_filename, *m_output))
        {
          std::lock_guard<std::mutex> lock(m_mutex);
          m_status   = STATE_LOADED;
          m_output   = nullptr;
          m_filename = "";
        }
        else
        {
          std::lock_guard<std::mutex> lock(m_mutex);
          m_status   = STATE_FAILURE;
          m_output   = nullptr;
          m_filename = "";
        }
      }
      else
      {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_status            = STATE_SHUTDOWN;
        m_shutdownRequested = false;
        m_output            = nullptr;
        m_filename          = "";
        return true;
      }
    }
  });

  return true;
}

void SplatLoaderAsync::cancel()
{
  // does nothing for the time beeing
}

SplatLoaderAsync::State SplatLoaderAsync::getStatus()
{
  std::lock_guard<std::mutex> lock(m_mutex);
  return m_status;
}

bool SplatLoaderAsync::reset()
{
  std::lock_guard<std::mutex> lock(m_mutex);
  if(m_status == STATE_LOADED || m_status == STATE_FAILURE)
  {
    m_progress = 0.0;
    m_status   = STATE_READY;
    return true;
  }
  else
  {
    return false;
  }
}

bool SplatLoaderAsync::innerLoad(std::filesystem::path filename, SplatSet& output)
{
  auto startTime = std::chrono::high_resolution_clock::now();

  std::string pathStr = filename.string();
  if(pathStr.find("http://") == 0 || pathStr.find("https://") == 0)
  {
    std::filesystem::path cacheDir = std::filesystem::temp_directory_path() / "vk_viewer_cache";

    if(!std::filesystem::exists(cacheDir))
    {
      std::filesystem::create_directories(cacheDir);
    }

    if(SupersplatClient::isSuperSplatUrl(pathStr))
    {
      std::string contentUrl;
      if(!SupersplatClient::resolveContentUrlSync(pathStr, contentUrl))
      {
        LOGE("Failed to resolve SuperSplat content URL\n");
        return false;
      }

      size_t      lastSlash = contentUrl.find_last_of('/');
      std::string baseUrl   = contentUrl.substr(0, lastSlash + 1);
      std::string metaUrl   = baseUrl + "meta.json";

      std::string cacheKey;
      {
        std::string superSplatId = extractSuperSplatId(pathStr);
        cacheKey                 = superSplatId.empty() ? "scene" : superSplatId;
      }

      LOGI("Resolved content URL, base: %s\n", baseUrl.c_str());
      std::filesystem::path sceneDir = cacheDir / cacheKey;
      if(!std::filesystem::exists(sceneDir))
      {
        std::filesystem::create_directories(sceneDir);
      }

      std::filesystem::path metaPath = sceneDir / "meta.json";

      if(!downloadFile(metaUrl, metaPath))
      {
        LOGE("Failed to download meta.json from SuperSplat\n");
        return false;
      }

      std::vector<uint8_t> metaData = readFileLocal(metaPath);
      SogMeta              meta;
      if(SogLoader::parseMeta(metaData, meta))
      {
        std::vector<std::string> filesToDownload;
        auto                     addFiles = [&](const std::vector<std::string>& files) {
          filesToDownload.insert(filesToDownload.end(), files.begin(), files.end());
        };

        addFiles(meta.means.files);
        addFiles(meta.scales.files);
        addFiles(meta.quats.files);
        addFiles(meta.sh0.files);
        addFiles(meta.shN.files);
        addFiles(meta.motion.files);
        addFiles(meta.t.files);
        addFiles(meta.t_scale.files);

        int total   = static_cast<int>(filesToDownload.size());
        int current = 0;

        {
          std::lock_guard<std::mutex> lock(m_mutex);
          m_downloadFileCount = total;
        }

        for(const auto& file : filesToDownload)
        {
          current++;
          std::filesystem::path localFilePath = sceneDir / file;

          if(std::filesystem::exists(localFilePath))
          {
            continue;
          }

          std::string fileUrl = baseUrl + file;
          LOGI("Downloading %s (%d/%d)...\n", file.c_str(), current, total);

          setProgress(static_cast<float>(current) / static_cast<float>(total));

          {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_downloadFileIndex       = current;
            m_currentDownloadingFile  = file;
            m_currentDownloadSize     = 0;
            m_currentDownloadProgress = 0;
          }

          auto progressCallback = [&](size_t downloaded, size_t totalBytes) {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_currentDownloadSize     = totalBytes;
            m_currentDownloadProgress = downloaded;
          };

          if(!downloadFile(fileUrl, localFilePath, progressCallback))
          {
            LOGE("Failed to download file: %s\n", file.c_str());
            return false;
          }
        }

        filename = metaPath;
        LOGI("SuperSplat scene download complete.\n");
      }
      else
      {
        LOGE("Failed to parse downloaded meta.json.\n");
        return false;
      }
    }
    else
    {
      std::string tempFileName = "downloaded_scene";

      size_t lastDot = pathStr.find_last_of('.');
      if(lastDot != std::string::npos && lastDot < pathStr.length() - 1)
      {
        std::string ext = pathStr.substr(lastDot);
        if(ext.length() <= 5)
        {
          tempFileName += ext;
        }
        else
        {
          tempFileName += ".sog";
        }
      }
      else
      {
        tempFileName += ".sog";
      }

      std::filesystem::path destPath = cacheDir / tempFileName;

      LOGI("Downloading %s to %s...\n", pathStr.c_str(), destPath.string().c_str());

      {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_downloadFileCount       = 1;
        m_downloadFileIndex       = 1;
        m_currentDownloadingFile  = pathStr;
        m_currentDownloadSize     = 0;
        m_currentDownloadProgress = 0;
      }

      auto progressCallback = [&](size_t downloaded, size_t totalBytes) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_currentDownloadSize     = totalBytes;
        m_currentDownloadProgress = downloaded;
        // For single file, use download progress as overall progress
        if(totalBytes > 0)
        {
          m_progress = static_cast<float>(downloaded) / static_cast<float>(totalBytes);
        }
      };

      if(downloadFile(pathStr, destPath, progressCallback))
      {
        filename = destPath;
        LOGI("Download complete. Proceeding to load...\n");
      }
      else
      {
        LOGE("Failed to download file from URL.\n");
        return false;
      }
    }
  }

  // LOD format (lod-meta.json with octree + multi-level SOG files)
  // Reference: https://developer.playcanvas.com/user-manual/gaussian-splatting/editing/splat-transform/
  if(filename.filename() == "lod-meta.json")
  {
    bool success = LodScene::loadAtLevel(filename, 0, output, [this](float progress) { setProgress(progress); });
    if(success)
    {
      auto      endTime  = std::chrono::high_resolution_clock::now();
      long long loadTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
      LOGI("LOD scene loaded in %lldms (%zu splats)\n", loadTime, output.size());
    }
    return success;
  }

  // SOG-XT container (KISS-GS format: directory with meta.json/scene.json manifest)
  // Try SogXtLoader first: it validates format: "sog-xt" in meta.json.
  if(std::filesystem::is_directory(filename) || filename.filename() == "scene.json"
     || filename.filename() == "meta.json")
  {
    SplatSet xtOutput;
    bool     isSogXt = SogXtLoader::load(filename, xtOutput, [this](float progress) { setProgress(progress); });
    if(isSogXt)
    {
      output = std::move(xtOutput);
      auto      endTime  = std::chrono::high_resolution_clock::now();
      long long loadTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
      LOGI("SOG-XT container loaded in %lldms (%zu splats)\n", loadTime, output.size());
      return true;
    }
  }

  // SOG format (bundled .sog or unbundled meta.json)
  if(hasExtension(filename, ".sog") || filename.filename() == "meta.json")
  {
    bool success = SogLoader::load(filename, output, [this](float progress) { setProgress(progress); });
    if(success)
    {
      auto      endTime  = std::chrono::high_resolution_clock::now();
      long long loadTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
      LOGI("SOG file loaded in %lldms\n", loadTime);
    }
    return success;
  }

  if(hasExtension(filename, ".rad"))
  {
    bool success = RadLoader::load(filename, output, [this](float progress) { setProgress(progress); });
    if(success)
    {
      auto      endTime  = std::chrono::high_resolution_clock::now();
      long long loadTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
      LOGI("RAD file loaded in %lldms\n", loadTime);
      return true;
    }
    LOGE("Error: RAD loader failed for file: %s\n", filename.string().c_str());
    return false;
  }

  // we use spz library for .spz extensions
  if(hasExtension(filename, ".spz"))
  {
    // Converts to RUB coordinate system
    spz::UnpackOptions options{.to = spz::CoordinateSystem::RUB};
    // let's load
    spz::GaussianCloud cloud = spz::loadSpz(filename.string(), options);
    // convert to INRIA representation
    output.positions.swap(cloud.positions);
    output.rotation.resize(cloud.rotations.size());
    const uint32_t numSplats = uint32_t(output.positions.size() / 3);
    for(uint32_t i = 0; i < numSplats; i++)
    {
      const uint32_t offset       = i * 4;
      output.rotation[offset + 0] = cloud.rotations[offset + 3];
      output.rotation[offset + 1] = cloud.rotations[offset + 0];
      output.rotation[offset + 2] = cloud.rotations[offset + 1];
      output.rotation[offset + 3] = cloud.rotations[offset + 2];
    }
    output.scale.swap(cloud.scales);
    output.opacity.swap(cloud.alphas);
    output.f_dc = cloud.colors;
    // reorganize SH per components to match INRIA
    const size_t shCoefsCount = cloud.sh.size() / numSplats / 3;
    output.f_rest.resize(cloud.sh.size());
    for(size_t i = 0; i < numSplats; i++)
    {
      const size_t offset = i * shCoefsCount * 3;

      // Spherical harmonics: Interleave so the coefficients are the fastest-changing axis and
      // the channel (r, g, b) is slower-changing axis.
      for(size_t j = 0; j < shCoefsCount; j++)
      {
        output.f_rest[offset + j] = cloud.sh[(i * shCoefsCount + j) * 3];
      }
      for(size_t j = 0; j < shCoefsCount; j++)
      {
        output.f_rest[offset + shCoefsCount + j] = cloud.sh[(i * shCoefsCount + j) * 3 + 1];
      }
      for(size_t j = 0; j < shCoefsCount; j++)
      {
        output.f_rest[offset + shCoefsCount * 2 + j] = cloud.sh[(i * shCoefsCount + j) * 3 + 2];
      }
    }
    //
    auto      endTime  = std::chrono::high_resolution_clock::now();
    long long loadTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    LOGI("File loaded in %lldms\n", loadTime);
    //
    return cloud.numPoints != 0;
  }

  // 4DV Loader
  if(hasExtension(filename, ".4dv"))
  {
    bool success = FourDvLoader::load(filename, output, [this](float progress) { setProgress(progress); });
    if(success)
    {
      auto      endTime  = std::chrono::high_resolution_clock::now();
      long long loadTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
      LOGI("4DV file loaded in %lldms\n", loadTime);
    }
    return success;
  }

  // Antimatter15 .splat format (32 bytes/gaussian, little-endian)
  //   bytes 0-11:  position (3xf32)
  //   bytes 12-23: scale (3xf32)
  //   bytes 24-27: RGBA (4xu8)
  //   bytes 28-31: rotation (4xu8, 128-centered quaternion w,x,y,z)
  if(hasExtension(filename, ".splat"))
  {
    LOGI("Loading antimatter15 .splat format: %s\n", pathStr.c_str());
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if(!file)
    {
      LOGE("Failed to open .splat file: %s\n", pathStr.c_str());
      return false;
    }
    const std::streamsize size = file.tellg();
    file.seekg(0);
    std::vector<uint8_t> data(static_cast<size_t>(size));
    if(!file.read(reinterpret_cast<char*>(data.data()), size))
    {
      LOGE("Failed to read .splat file: %s\n", pathStr.c_str());
      return false;
    }
    if(data.empty() || data.size() % 32 != 0)
    {
      LOGE("Invalid .splat file: size=%zu (expected multiple of 32)\n", data.size());
      return false;
    }
    const size_t n_gaussians = data.size() / 32;
    output.positions.resize(n_gaussians * 3);
    output.scale.resize(n_gaussians * 3);
    output.rotation.resize(n_gaussians * 4);
    output.opacity.resize(n_gaussians);
    output.f_dc.resize(n_gaussians * 3);
    // SplatSet stores SH DC coefficients (f_dc), not direct colors.
    // color = 0.5 + SH_C0 * f_dc  =>  f_dc = (color - 0.5) / SH_C0
    constexpr float SH_C0 = 0.28209479177387814f;
    for(size_t i = 0; i < n_gaussians; ++i)
    {
      const uint8_t* p = data.data() + i * 32;
      // position (3xf32 little-endian) - .splat uses RDF (Right-Down-Forward)
      // viewer expects RUB (Right-Up-Back), so negate Y and Z
      std::memcpy(&output.positions[i * 3], p, 12);
      output.positions[i * 3 + 1] = -output.positions[i * 3 + 1];
      output.positions[i * 3 + 2] = -output.positions[i * 3 + 2];
      // scale (3xf32 little-endian) - .splat stores linear scales (exp(log_scale))
      // but SplatSet expects log-space scales (renderer applies exp() in all pipelines)
      float rawScale[3];
      std::memcpy(rawScale, p + 12, 12);
      output.scale[i * 3 + 0] = std::log(std::max(rawScale[0], 1e-7f));
      output.scale[i * 3 + 1] = std::log(std::max(rawScale[1], 1e-7f));
      output.scale[i * 3 + 2] = std::log(std::max(rawScale[2], 1e-7f));
      // color (4 xu8: r,g,b,a) → f_dc SH DC coefficients
      const float r = p[24] / 255.0f;
      const float g = p[25] / 255.0f;
      const float b = p[26] / 255.0f;
      const float a = p[27] / 255.0f;
      output.f_dc[i * 3 + 0] = (r - 0.5f) / SH_C0;
      output.f_dc[i * 3 + 1] = (g - 0.5f) / SH_C0;
      output.f_dc[i * 3 + 2] = (b - 0.5f) / SH_C0;
      // opacity - .splat stores sigmoid(alpha) ∈ [0,1]
      // but SplatSet expects logit-space (renderer applies sigmoid in all pipelines)
      // convert: logit(x) = log(x/(1-x))
      float clampedA = std::max(std::min(a, 1.0f - 1e-6f), 1e-6f);
      output.opacity[i] = std::log(clampedA / (1.0f - clampedA));
      // rotation (4 xu8, 128-centered, -1..1) quaternion w,x,y,z
      for(int c = 0; c < 4; c++)
      {
        output.rotation[i * 4 + c] = (p[28 + c] - 128.0f) / 128.0f;
      }
    }
    LOGI("Loaded .splat: %zu gaussians\n", n_gaussians);
    auto      endTime  = std::chrono::high_resolution_clock::now();
    long long loadTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    LOGI(".splat file loaded in %lldms\n", loadTime);
    return true;
  }

  // NPZ Loader (SplatAD format)
  if(hasExtension(filename, ".npz") && NpzLoader::canLoad(filename))
  {
    bool success = NpzLoader::load(filename, output, [this](float progress) { setProgress(progress); });
    if(success)
    {
      size_t timestepCount = NpzLoader::getTimestepCount(filename);
      bool   hasTemporal   = timestepCount > 1;
      LOGI("NPZ file loaded: %zu splats, %zu timesteps%s\n", output.size(), timestepCount, hasTemporal ? " [temporal]" : "");
      output.convertCoordinates(spz::CoordinateSystem::RDF, spz::CoordinateSystem::RUB);
      auto      endTime  = std::chrono::high_resolution_clock::now();
      long long loadTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
      LOGI("NPZ file loaded in %lldms\n", loadTime);
      return true;
    }
    LOGE("Error: NPZ loader failed for file: %s\n", filename.string().c_str());
    return false;
  }

  // LCC Loader (Lixel CyberColor format)
  if(LccLoader::canLoad(filename))
  {
    bool success;
    if(m_lodReloadPending)
    {
      // Load at specific LOD level (for LOD slider changes)
      success            = LccLoader::loadWithLod(filename, output, m_targetLod, nullptr, nullptr,
                                                  [this](float progress) { setProgress(progress); });
      m_lodReloadPending = false;
    }
    else
    {
      // Default to LOD 1 for faster initial load (per LCC spec)
      // Only if the scene has multiple LOD levels
      uint32_t lodCount = LccLoader::getLodCount(filename);
      if(lodCount > 1)
      {
        // Multi-LOD scene: load at LOD 1 for faster initial load
        int targetLod = 1;
        success       = LccLoader::loadWithLod(filename, output, targetLod, nullptr, nullptr,
                                               [this](float progress) { setProgress(progress); });
      }
      else
      {
        // Single-LOD scene: load all data directly
        success = LccLoader::load(filename, output, [this](float progress) { setProgress(progress); });
      }
    }

    if(success)
    {
      auto      endTime  = std::chrono::high_resolution_clock::now();
      long long loadTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
      LOGI("LCC file loaded in %lldms\n", loadTime);
      return true;
    }
    LOGE("Error: LCC loader failed for file: %s\n", filename.string().c_str());
    return false;
  }

  if(hasExtension(filename, ".ply") && SplatLoaderFast::canLoad(filename))
  {
    bool success = SplatLoaderFast::load(filename, output, [this](float progress) { setProgress(progress); });
    if(success)
    {
      if(output.has_time_data)
      {
        LOGI("FreeTimeGS temporal data detected: %zu splats with motion vectors\n", output.size());
        size_t sampleCount = output.size() < 5 ? output.size() : 5;
        for(size_t i = 0; i < sampleCount; ++i)
        {
          LOGD("  Splat %zu: motion=(%.3f, %.3f, %.3f) t=%.3f t_scale=%.3f\n", i, output.motion[i * 3],
               output.motion[i * 3 + 1], output.motion[i * 3 + 2], output.time[i], output.time_scale[i]);
        }
      }
      else
      {
        output.convertCoordinates(spz::CoordinateSystem::RDF, spz::CoordinateSystem::RUB);
      }
      auto      endTime  = std::chrono::high_resolution_clock::now();
      long long loadTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
      LOGI("PLY file loaded in %lldms (fast loader)%s\n", loadTime, output.has_time_data ? " [4D temporal]" : "");
      return true;
    }
    LOGE("Error: fast PLY loader failed for binary file: %s\n", filename.string().c_str());
    return false;
  }

  LOGE("Error: unsupported file format: %s\n", filename.string().c_str());
  return false;
}
