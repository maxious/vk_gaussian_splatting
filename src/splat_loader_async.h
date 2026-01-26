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

#ifndef _SPLAT_ASYNC_LOADER_H_
#define _SPLAT_ASYNC_LOADER_H_

#include <string>
// threading
#include <thread>
#include <condition_variable>
#include <mutex>
#include <filesystem>

#include "splat_set.h"

namespace vk_viewer {

// Async loader for Gaussian splat formats (PLY, SPZ, SOG)
class SplatLoaderAsync
{
public:
  enum State
  {
    STATE_SHUTDOWN,  // loader must be initialized (loading thread is not started)
    STATE_READY,     // loader ready to load a new model
    STATE_LOADING,   // loader is currently loading
    STATE_LOADED,    // loader has finished loading, model is available. call reset before another load.
    STATE_FAILURE    // an error occurred. call reset before another load.
  };

public:
  // starts the loader thread
  bool initialize();
  // stops the loader thread, cannot be re-used afterward
  inline void shutdown()
  {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_shutdownRequested = true;
    m_loadCV.notify_all();
    lock.unlock();
    // wait for thread termination
    m_loader.join();
  }
  // triggers the load of a new scene
  // return false if loader not in idled state
  // output must not be accessed if status is not LOADED or READY (after reset)
  bool loadScene(std::filesystem::path filename, SplatSet& output);

  // Load scene at specific LOD level (for LCC format)
  // Returns false if loader not in idled state or file is not LCC
  bool loadSceneAtLod(std::filesystem::path filename, SplatSet& output, int lodLevel);

  // cancel scene loading if possible
  // non blocking, may have no effect
  void cancel();
  // return loader status
  State getStatus();
  // Resets the loader to READY after LOADED or FAILURE
  // used to ack that the consumer has consumed the loaded model
  // loader must be reset to be able to launch a new load
  // thread safe
  bool reset();
  // return the filename currently beeing loaded, "" otherwise
  [[nodiscard]] inline std::filesystem::path getFilename()
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_filename;
  }
  // return percentage in {0,1} of progress
  // do not rely on progress to find loader status
  // use getStatus(). progress is just an indication
  // for UI display.
  [[nodiscard]] inline float getProgress()
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_progress;
  }

  // New getters for download details
  [[nodiscard]] inline std::string getCurrentDownloadingFilename()
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_currentDownloadingFile;
  }

  [[nodiscard]] inline size_t getCurrentDownloadSize()
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_currentDownloadSize;
  }

  [[nodiscard]] inline size_t getCurrentDownloadProgress()
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_currentDownloadProgress;
  }

  inline void getDownloadFileCounts(int& current, int& total)
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    current = m_downloadFileIndex;
    total = m_downloadFileCount;
  }

  // Set/get LOD level for reload
  inline void setTargetLod(int lod) { m_targetLod = lod; }
  inline int  getTargetLod() const { return m_targetLod; }
  inline void setLodReloadPending(bool pending) { m_lodReloadPending = pending; }
  inline bool isLodReloadPending() const { return m_lodReloadPending; }

private:
  // actually loads the scene
  bool innerLoad(std::filesystem::path filename, SplatSet& output);

  // in {0.0,1.0}
  void setProgress(float progress)
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_progress = progress;
  }

private:
  // loading thread
  std::thread m_loader;
  // loader status
  State m_status = STATE_SHUTDOWN;
  // ask to cancel a load
  bool m_cancelRequested = false;
  // ask for loader shutdown before destruction
  bool m_shutdownRequested = false;
  // protects the condition variables and other attributes
  mutable std::mutex m_mutex;
  // loader wakeup condition
  mutable std::condition_variable m_loadCV;

  // the splat file pathname
  std::filesystem::path m_filename = "";
  // the output data storage
  SplatSet* m_output = nullptr;
  // the loading percentage
  float m_progress = 0.0f;
  
  // Download details
  std::string m_currentDownloadingFile = "";
  size_t m_currentDownloadSize = 0;
  size_t m_currentDownloadProgress = 0;
  int m_downloadFileIndex = 0;
  int m_downloadFileCount = 0;

  // LOD reload state
  int  m_targetLod = 0;
  bool m_lodReloadPending = false;
};

}  // namespace vk_viewer

#endif
