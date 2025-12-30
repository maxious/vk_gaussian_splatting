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

using namespace vk_gaussian_splatting;

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
