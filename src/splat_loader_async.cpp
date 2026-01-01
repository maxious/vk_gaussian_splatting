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
#include <fstream>
#include <array>
#include <chrono>
#include <filesystem>

#include <nvutils/logger.hpp>

#include "splat_loader_fast.h"
// 3rd party spz library
#include "load-spz.h"

//
#include "splat_loader_async.h"
#include "sog_loader.h"
#include "fourdv_loader.h"
#include "utilities.h"

#ifdef _WIN32
#include <windows.h>
#include <winhttp.h>
#endif

using namespace vk_gaussian_splatting;

// Type alias for SogMeta used in SuperSplat download
using SogMeta = SogLoader::SogMeta;

// Helper to extract ID from SuperSplat URL
std::string extractSuperSplatId(const std::string& url)
{
  // Support formats:
  // https://superspl.at/view?id=bd964899
  // https://superspl.at/s?id=bd964899
  
  std::string idKey = "?id=";
  size_t pos = url.find(idKey);
  if (pos == std::string::npos) return "";
  
  std::string id = url.substr(pos + idKey.length());
  // Truncate at next parameter if any
  size_t endPos = id.find('&');
  if (endPos != std::string::npos)
  {
    id = id.substr(0, endPos);
  }
  return id;
}

// Helper to read file into vector
std::vector<uint8_t> readFileLocal(const std::filesystem::path& path)
{
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if(!file) return {};
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<uint8_t> buffer(size);
  if(!file.read(reinterpret_cast<char*>(buffer.data()), size)) return {};
  return buffer;
}

#ifdef _WIN32
namespace {
bool downloadFile(const std::string& url, const std::filesystem::path& destPath)
{
  URL_COMPONENTS urlComp;
  ZeroMemory(&urlComp, sizeof(urlComp));
  urlComp.dwStructSize = sizeof(urlComp);

  wchar_t hostName[256] = {0};
  wchar_t urlPath[2048] = {0};
  urlComp.lpszHostName = hostName;
  urlComp.dwHostNameLength = sizeof(hostName) / sizeof(wchar_t);
  urlComp.lpszUrlPath = urlPath;
  urlComp.dwUrlPathLength = sizeof(urlPath) / sizeof(wchar_t);

  std::wstring wideUrl(url.begin(), url.end());

  if (!WinHttpCrackUrl(wideUrl.c_str(), static_cast<DWORD>(wideUrl.length()), 0, &urlComp))
  {
    LOGE("Failed to parse URL: %s (error %lu)\n", url.c_str(), GetLastError());
    return false;
  }

  HINTERNET hSession = WinHttpOpen(L"VkGaussianSplatting/1.0",
                                   WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                                   WINHTTP_NO_PROXY_NAME,
                                   WINHTTP_NO_PROXY_BYPASS, 0);
  if (!hSession)
  {
    LOGE("WinHttpOpen failed (error %lu)\n", GetLastError());
    return false;
  }

  HINTERNET hConnect = WinHttpConnect(hSession, hostName, urlComp.nPort, 0);
  if (!hConnect)
  {
    LOGE("WinHttpConnect failed (error %lu)\n", GetLastError());
    WinHttpCloseHandle(hSession);
    return false;
  }

  DWORD dwFlags = (urlComp.nScheme == INTERNET_SCHEME_HTTPS) ? WINHTTP_FLAG_SECURE : 0;
  HINTERNET hRequest = WinHttpOpenRequest(hConnect, L"GET", urlPath,
                                          NULL, WINHTTP_NO_REFERER,
                                          WINHTTP_DEFAULT_ACCEPT_TYPES,
                                          dwFlags);
  if (!hRequest)
  {
    LOGE("WinHttpOpenRequest failed (error %lu)\n", GetLastError());
    WinHttpCloseHandle(hConnect);
    WinHttpCloseHandle(hSession);
    return false;
  }

  if (!WinHttpSendRequest(hRequest, WINHTTP_NO_ADDITIONAL_HEADERS, 0,
                          WINHTTP_NO_REQUEST_DATA, 0, 0, 0))
  {
    LOGE("WinHttpSendRequest failed (error %lu)\n", GetLastError());
    WinHttpCloseHandle(hRequest);
    WinHttpCloseHandle(hConnect);
    WinHttpCloseHandle(hSession);
    return false;
  }

  if (!WinHttpReceiveResponse(hRequest, NULL))
  {
    LOGE("WinHttpReceiveResponse failed (error %lu)\n", GetLastError());
    WinHttpCloseHandle(hRequest);
    WinHttpCloseHandle(hConnect);
    WinHttpCloseHandle(hSession);
    return false;
  }

  DWORD dwStatusCode = 0;
  DWORD dwSize = sizeof(dwStatusCode);
  if (!WinHttpQueryHeaders(hRequest, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                           WINHTTP_HEADER_NAME_BY_INDEX, &dwStatusCode, &dwSize, WINHTTP_NO_HEADER_INDEX))
  {
    LOGE("WinHttpQueryHeaders failed (error %lu)\n", GetLastError());
    WinHttpCloseHandle(hRequest);
    WinHttpCloseHandle(hConnect);
    WinHttpCloseHandle(hSession);
    return false;
  }

  if (dwStatusCode != 200)
  {
    LOGE("Download failed: HTTP %d\n", dwStatusCode);
    WinHttpCloseHandle(hRequest);
    WinHttpCloseHandle(hConnect);
    WinHttpCloseHandle(hSession);
    return false;
  }

  std::ofstream outFile(destPath, std::ios::binary);
  if (!outFile)
  {
    LOGE("Failed to create file: %s\n", destPath.string().c_str());
    WinHttpCloseHandle(hRequest);
    WinHttpCloseHandle(hConnect);
    WinHttpCloseHandle(hSession);
    return false;
  }

  DWORD dwSizeAvail = 0;
  DWORD dwDownloaded = 0;
  std::vector<char> buffer(8192);

  do
  {
    dwSizeAvail = 0;
    if (!WinHttpQueryDataAvailable(hRequest, &dwSizeAvail))
    {
      LOGE("WinHttpQueryDataAvailable failed (error %lu)\n", GetLastError());
      break;
    }

    if (dwSizeAvail > 0)
    {
      if (dwSizeAvail > buffer.size()) buffer.resize(dwSizeAvail);

      if (WinHttpReadData(hRequest, buffer.data(), dwSizeAvail, &dwDownloaded))
      {
        outFile.write(buffer.data(), dwDownloaded);
      }
      else
      {
        LOGE("WinHttpReadData failed (error %lu)\n", GetLastError());
        break;
      }
    }
  } while (dwSizeAvail > 0);

  outFile.close();
  WinHttpCloseHandle(hRequest);
  WinHttpCloseHandle(hConnect);
  WinHttpCloseHandle(hSession);

  return true;
}
}
#endif



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

bool SplatLoaderAsync::initialize()
{
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
  if (pathStr.find("http://") == 0 || pathStr.find("https://") == 0)
  {
#ifdef _WIN32
    std::string superSplatId = extractSuperSplatId(pathStr);
    std::filesystem::path cacheDir = std::filesystem::temp_directory_path() / "vk_gaussian_splatting_cache";
    
    if (!std::filesystem::exists(cacheDir)) {
        std::filesystem::create_directories(cacheDir);
    }

    if (!superSplatId.empty())
    {
        LOGI("Detected SuperSplat ID: %s\n", superSplatId.c_str());
        std::filesystem::path sceneDir = cacheDir / superSplatId;
        if (!std::filesystem::exists(sceneDir)) {
            std::filesystem::create_directories(sceneDir);
        }

        std::filesystem::path metaPath = sceneDir / "meta.json";
        
        // Try v3, v2, v1 in order - this is the versioned content path for SuperSplat
        std::vector<std::string> versions = {"v3", "v2", "v1"};
        std::string successVersion;
        
        for (const auto& version : versions)
        {
            std::string metaUrl = "https://d28zzqy0iyovbz.cloudfront.net/" + superSplatId + "/" + version + "/meta.json";
            LOGI("Trying %s meta.json from %s...\n", version.c_str(), metaUrl.c_str());
            
            if (downloadFile(metaUrl, metaPath))
            {
                successVersion = version;
                LOGI("Successfully downloaded meta.json using %s format.\n", version.c_str());
                break;
            }
        }
        
        if (successVersion.empty())
        {
            LOGE("Failed to download meta.json from SuperSplat (tried v3, v2, v1).\n");
            return false;
        }

        std::vector<uint8_t> metaData = readFileLocal(metaPath);
        SogMeta meta;
        if (SogLoader::parseMeta(metaData, meta))
        {
            std::vector<std::string> filesToDownload;
            auto addFiles = [&](const std::vector<std::string>& files) {
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

            int total = static_cast<int>(filesToDownload.size());
            int current = 0;
            for (const auto& file : filesToDownload)
            {
                current++;
                std::filesystem::path localFilePath = sceneDir / file;
                
                if (std::filesystem::exists(localFilePath)) {
                    continue;
                }

                std::string fileUrl = "https://d28zzqy0iyovbz.cloudfront.net/" + superSplatId + "/" + successVersion + "/" + file;
                LOGI("Downloading %s (%d/%d)...\n", file.c_str(), current, total);
                
                setProgress(static_cast<float>(current) / static_cast<float>(total));

                if (!downloadFile(fileUrl, localFilePath))
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
        if (lastDot != std::string::npos && lastDot < pathStr.length() - 1) {
            std::string ext = pathStr.substr(lastDot);
            if (ext.length() <= 5) {
                tempFileName += ext;
            } else {
                 tempFileName += ".sog";
            }
        } else {
            tempFileName += ".sog";
        }
        
        std::filesystem::path destPath = cacheDir / tempFileName;
        
        LOGI("Downloading %s to %s...\n", pathStr.c_str(), destPath.string().c_str());
        
        if (downloadFile(pathStr, destPath))
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
#else
    LOGE("URL loading is currently only supported on Windows.\n");
    return false;
#endif
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
          LOGD("  Splat %zu: motion=(%.3f, %.3f, %.3f) t=%.3f t_scale=%.3f\n",
               i, output.motion[i*3], output.motion[i*3+1], output.motion[i*3+2],
               output.time[i], output.time_scale[i]);
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
