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

#pragma once

#ifdef WITH_FREE_SPLATTER

#include <atomic>
#include <string>
#include <thread>

#include <sys/types.h>

namespace vk_viewer {

// Manages the local depth_server subprocess running in --mode splat
// (free-splatter.cpp image-to-3DGS generation).
// Mirrors LocalDepthServerManager but spawns the splat server with
// auto-detected port (9001..9010) and per-process log file.
class LocalSplatServerManager {
public:
    LocalSplatServerManager();
    ~LocalSplatServerManager();

    // Start the splat server subprocess.
    // Auto-detects a free port in the range [9001, 9010].
    // Returns the port number on success, -1 on failure.
    int start(const std::string& modelIdentifier = "");

    // Stop the splat server subprocess.
    void stop();

    // Check if the server is currently running.
    bool isRunning() const { return m_running.load(); }

    // Get the port the server is running on (-1 if not started).
    int getPort() const { return m_port; }

    // Get the model identifier passed to --splat-model.
    const std::string& getModelIdentifier() const { return m_modelIdentifier; }

private:
    void monitorProcess();

    std::atomic<bool> m_running{false};
    std::string       m_modelIdentifier;
    int               m_port = -1;
    pid_t             m_pid = -1;
    std::thread       m_monitorThread;
};

}  // namespace vk_viewer

#endif  // WITH_FREE_SPLATTER