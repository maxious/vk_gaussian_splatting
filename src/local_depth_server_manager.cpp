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

#include "local_depth_server_manager.h"

#ifdef WITH_TCP_DEPTH

#include <nvutils/logger.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <thread>

namespace vk_viewer {

namespace {

bool isPortListening(int port)
{
    int sock = ::socket(AF_INET, SOCK_STREAM, 0);
    if(sock < 0)
    {
        return false;
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

    const bool connected = (::connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0);
    ::close(sock);
    return connected;
}

std::filesystem::path locateDepthServerBinary()
{
    namespace fs = std::filesystem;

    const char* env_path = std::getenv("VK_GS_DEPTH_SERVER");
    if(env_path != nullptr && env_path[0] != '\0')
    {
        return fs::path(env_path);
    }

    fs::path candidate = fs::current_path() / "depth_server";
    if(fs::exists(candidate))
    {
        return candidate;
    }

    for(const char* prefix : {"../..", "../depth_server/build", "../../depth_server/build", "depth_server/build"})
    {
        fs::path p = prefix;
        p /= "depth_server";
        if(fs::exists(p))
        {
            return fs::absolute(p);
        }
    }

    return {};
}

}  // namespace

LocalDepthServerManager::LocalDepthServerManager() = default;

LocalDepthServerManager::~LocalDepthServerManager()
{
    stop();
}

bool LocalDepthServerManager::start(const std::string& modelPath, int port, int workers, const std::string& backend)
{
    if(m_running.load())
    {
        LOGW("LocalDepthServerManager: already running\n");
        return true;
    }

    if(modelPath.empty())
    {
        LOGE("LocalDepthServerManager: model path is empty\n");
        return false;
    }

    if(!std::filesystem::exists(std::filesystem::path(modelPath)))
    {
        LOGE("LocalDepthServerManager: model file not found: %s\n", modelPath.c_str());
        return false;
    }

    const std::filesystem::path binary = locateDepthServerBinary();
    if(binary.empty() || !std::filesystem::exists(binary))
    {
        LOGE("LocalDepthServerManager: depth_server binary not found in PATH or build dir\n");
        return false;
    }

    m_modelPath = modelPath;
    m_port = port;
    m_workers = workers;
    m_backend = backend;

    const std::string port_str = std::to_string(port);
    const std::string workers_str = std::to_string(workers);

    const pid_t pid = ::fork();
    if(pid < 0)
    {
        LOGE("LocalDepthServerManager: fork failed: %s\n", std::strerror(errno));
        return false;
    }

    if(pid == 0)
    {
        const std::string binary_str = binary.string();
        const char* const model_arg   = modelPath.c_str();
        const char* const port_arg    = port_str.c_str();
        const char* const workers_arg = workers_str.c_str();
        const char* const backend_arg = backend.c_str();
        const char* const binary_arg  = binary_str.c_str();

        ::execlp(binary_arg, binary_arg,
                 "--model", model_arg,
                 "--port", port_arg,
                 "--workers", workers_arg,
                 "--backend", backend_arg,
                 static_cast<char*>(nullptr));

        LOGE("LocalDepthServerManager: execlp failed: %s\n", std::strerror(errno));
        std::_Exit(1);
    }

    m_pid = pid;
    m_running.store(true);

    LOGI("LocalDepthServerManager: started depth_server (PID %d, model %s, port %d, backend %s)\n",
         static_cast<int>(pid), modelPath.c_str(), port, backend.c_str());

    m_monitorThread = std::thread(&LocalDepthServerManager::monitorProcess, this);

    for(int attempt = 0; attempt < 50; ++attempt)
    {
        if(isPortListening(port))
        {
            LOGI("LocalDepthServerManager: depth_server is ready on port %d\n", port);
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    LOGW("LocalDepthServerManager: depth_server may not be ready on port %d (timeout)\n", port);
    return true;
}

void LocalDepthServerManager::stop()
{
    if(!m_running.load())
    {
        return;
    }

    if(m_pid > 0)
    {
        LOGI("LocalDepthServerManager: sending SIGTERM to PID %d\n", static_cast<int>(m_pid));
        ::kill(m_pid, SIGTERM);

        for(int i = 0; i < 50; ++i)
        {
            int status = 0;
            const pid_t result = ::waitpid(m_pid, &status, WNOHANG);
            if(result == m_pid)
            {
                m_pid = -1;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        if(m_pid > 0)
        {
            LOGW("LocalDepthServerManager: sending SIGKILL to PID %d\n", static_cast<int>(m_pid));
            ::kill(m_pid, SIGKILL);
            ::waitpid(m_pid, nullptr, 0);
            m_pid = -1;
        }
    }

    m_running.store(false);

    if(m_monitorThread.joinable())
    {
        m_monitorThread.join();
    }
}

void LocalDepthServerManager::monitorProcess()
{
    while(m_running.load())
    {
        int status = 0;
        const pid_t result = ::waitpid(m_pid, &status, WNOHANG);
        if(result == m_pid)
        {
            if(WIFEXITED(status))
            {
                LOGW("LocalDepthServerManager: depth_server exited with code %d\n", WEXITSTATUS(status));
            }
            else if(WIFSIGNALED(status))
            {
                LOGW("LocalDepthServerManager: depth_server killed by signal %d\n", WTERMSIG(status));
            }
            m_running.store(false);
            m_pid = -1;
            break;
        }
        else if(result < 0 && errno != ECHILD)
        {
            LOGW("LocalDepthServerManager: waitpid error: %s\n", std::strerror(errno));
            m_running.store(false);
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

}  // namespace vk_viewer

#endif
