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

#include <string>
#include <atomic>
#include <thread>

namespace vk_viewer {

// Manages the Python backend process for depth streaming
class BackendProcessManager {
public:
    BackendProcessManager();
    ~BackendProcessManager();

    // Start the backend process
    // Returns true if started successfully or already running
    bool start(const std::string& pythonDir = "");

    // Stop the backend process
    void stop();

    // Check if the backend is currently running
    bool isRunning() const;

    // Check if the backend was started by us (vs external)
    bool isManaged() const { return m_managed; }

    // Get the port the backend is running on
    int getPort() const { return m_port; }

    // Get the host address
    std::string getHost() const { return m_host; }

    // Check if the backend is responding
    bool ping() const;

    // Wait for backend to become ready (with timeout)
    bool waitForReady(int timeoutMs = 10000) const;

private:
    void monitorProcess();

    std::atomic<bool> m_running{false};
    std::atomic<bool> m_managed{false};  // true if we started the process
    std::string m_host = "127.0.0.1";
    int m_port = 8000;
    int m_processId = -1;
    std::thread m_monitorThread;
};

}  // namespace vk_viewer
