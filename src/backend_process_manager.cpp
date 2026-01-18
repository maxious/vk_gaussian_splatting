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

#include "backend_process_manager.h"
#include <nvutils/logger.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifndef _WIN32
#include <unistd.h>
#include <sys/wait.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <errno.h>
#endif

namespace vk_viewer {

#ifndef _WIN32

BackendProcessManager::BackendProcessManager() = default;

BackendProcessManager::~BackendProcessManager() {
    stop();
}

bool BackendProcessManager::start(const std::string& pythonDir) {
    if (m_running.load()) {
        return true;  // Already running
    }

    // Find the project root directory
    std::string projectRoot;

    if (!pythonDir.empty()) {
        projectRoot = pythonDir;
    } else {
        // Try to find the project root relative to the executable
        const char* envPath = getenv("VK_GS_PROJECT_ROOT");
        if (envPath) {
            projectRoot = envPath;
        } else {
            // Default path - assume we're running from _bin/[Debug|Release]
            projectRoot = "../..";
        }
    }

    // Build the command to start the backend
    // We need to: cd to python dir, source venv, and run uvicorn
    std::string pythonProjectDir = projectRoot + "/python";

    // Check if the python directory exists
    FILE* testFp = fopen((pythonProjectDir + "/pyproject.toml").c_str(), "r");
    if (!testFp) {
        LOGW("BackendProcessManager: Python project not found at %s\n", pythonProjectDir.c_str());
        return false;
    }
    fclose(testFp);

    // Check if virtual environment exists
    std::string venvActivate = pythonProjectDir + "/.venv/bin/activate";
    FILE* venvFp = fopen(venvActivate.c_str(), "r");
    if (!venvFp) {
        LOGW("BackendProcessManager: Virtual environment not found at %s\n", venvActivate.c_str());
        return false;
    }
    fclose(venvFp);

    // Fork and execute the backend
    pid_t pid = fork();

    if (pid < 0) {
        LOGE("BackendProcessManager: fork() failed: %s\n", strerror(errno));
        return false;
    }

    if (pid == 0) {
        // Child process - set up the environment and exec

        // Change to the python directory
        if (chdir(pythonProjectDir.c_str()) != 0) {
            LOGE("BackendProcessManager: chdir to %s failed: %s\n", pythonProjectDir.c_str(), strerror(errno));
            _exit(1);
        }

        // Set up environment variables for the backend
        // Use XPU backend with multi-device support
        setenv("VIDEO_DEPTH_MULTI_DEVICE", "1", 1);
        setenv("VIDEO_DEPTH_DEVICE_SPEC", "xpu:0,1", 0);

        // Execute the backend using bash to source the venv and run uvicorn
        // The command: source .venv/bin/activate && uv run --extra backend uvicorn backend.main:app --host 0.0.0.0 --port 8000
        execlp("bash", "bash", "-c",
               "source .venv/bin/activate && uv run --extra backend uvicorn backend.main:app --host 0.0.0.0 --port 8000",
               nullptr);

        // If we get here, exec failed
        LOGE("BackendProcessManager: execlp failed: %s\n", strerror(errno));
        _exit(1);
    }

    // Parent process
    m_processId = pid;
    m_running.store(true);
    m_managed.store(true);

    LOGI("BackendProcessManager: Started backend with PID %d\n", pid);

    // Start a monitor thread to track process status
    m_monitorThread = std::thread([this]() {
        int status;
        while (m_running.load()) {
            pid_t result = waitpid(m_processId, &status, WNOHANG);
            if (result != 0) {
                // Process exited
                m_running.store(false);
                m_processId = -1;
                if (result > 0 && WIFEXITED(status)) {
                    LOGI("BackendProcessManager: Backend exited with code %d\n", WEXITSTATUS(status));
                } else if (result > 0 && WIFSIGNALED(status)) {
                    LOGI("BackendProcessManager: Backend killed by signal %d\n", WTERMSIG(status));
                } else if (result < 0) {
                    LOGI("BackendProcessManager: waitpid error: %s\n", strerror(errno));
                }
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    });

    // Wait for the backend to become ready
    if (!waitForReady(15000)) {
        LOGW("BackendProcessManager: Backend did not become ready within timeout\n");
        // Don't fail - the backend might still be starting
    }

    return true;
}

void BackendProcessManager::stop() {
    if (!m_running.load()) {
        return;
    }

    if (m_processId > 0) {
        // Send SIGTERM first for graceful shutdown
        LOGI("BackendProcessManager: Sending SIGTERM to PID %d\n", m_processId);
        kill(m_processId, SIGTERM);

        // Wait up to 5 seconds for graceful shutdown
        for (int i = 0; i < 50; i++) {
            int status;
            pid_t result = waitpid(m_processId, &status, WNOHANG);
            if (result != 0) {
                m_running.store(false);
                m_processId = -1;
                LOGI("BackendProcessManager: Backend terminated gracefully\n");
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        // If still running, force kill
        if (m_running.load()) {
            LOGI("BackendProcessManager: Sending SIGKILL to PID %d\n", m_processId);
            kill(m_processId, SIGKILL);
            waitpid(m_processId, nullptr, 0);
            m_running.store(false);
            m_processId = -1;
        }
    }

    // Join monitor thread
    if (m_monitorThread.joinable()) {
        m_monitorThread.join();
    }

    m_managed.store(false);
}

bool BackendProcessManager::isRunning() const {
    return m_running.load();
}

bool BackendProcessManager::ping() const {
    if (!m_running.load()) {
        return false;
    }

    // Try to connect to the backend port
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        return false;
    }

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(m_port));
    inet_pton(AF_INET, m_host.c_str(), &addr.sin_addr);

    bool success = (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) == 0);
    close(sock);

    return success;
}

bool BackendProcessManager::waitForReady(int timeoutMs) const {
    const int pollIntervalMs = 200;
    int elapsed = 0;

    while (elapsed < timeoutMs) {
        if (ping()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(pollIntervalMs));
        elapsed += pollIntervalMs;
    }

    return false;
}

#else

BackendProcessManager::BackendProcessManager() = default;

BackendProcessManager::~BackendProcessManager() {}

bool BackendProcessManager::start(const std::string&) {
    LOGW("BackendProcessManager: Not supported on Windows\\n");
    return false;
}

void BackendProcessManager::stop() {}

bool BackendProcessManager::isRunning() const {
    return false;
}

bool BackendProcessManager::ping() const {
    return false;
}

bool BackendProcessManager::waitForReady(int) const {
    return false;
}

#endif

}  // namespace vk_viewer
