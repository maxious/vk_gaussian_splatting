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

#include "local_splat_server_manager.h"

#ifdef WITH_FREE_SPLATTER

#include <nvutils/logger.hpp>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <thread>
#include <vector>

namespace vk_viewer {

namespace {

constexpr int kSplatPortStart = 9001;
constexpr int kSplatPortEnd   = 9010;

bool isPortListening(int port)
{
    int sock = ::socket(AF_INET, SOCK_STREAM, 0);
    if(sock < 0)
    {
        return false;
    }

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_port        = htons(static_cast<uint16_t>(port));
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

    const bool connected = (::connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0);
    ::close(sock);
    return connected;
}

int findFreePort()
{
    for(int port = kSplatPortStart; port <= kSplatPortEnd; ++port)
    {
        if(!isPortListening(port))
        {
            return port;
        }
    }
    return -1;
}

std::filesystem::path locateDepthServerBinary()
{
    namespace fs = std::filesystem;

    auto isExecutable = [](const fs::path& p) -> bool {
        if(::access(p.c_str(), X_OK) != 0)
        {
            if(errno == EACCES)
            {
                LOGW("LocalSplatServerManager: depth_server at '%s' exists but is not executable (missing +x?)\n",
                     p.c_str());
            }
            return false;
        }
        return true;
    };

    const char* env_path = std::getenv("VK_GS_DEPTH_SERVER");
    if(env_path != nullptr && env_path[0] != '\0')
    {
        fs::path p(env_path);
        if(isExecutable(p))
            return p;
        return {};
    }

    // CWD-relative (e.g., _bin/Debug/depth_server)
    fs::path candidate = fs::current_path() / "depth_server";
    if(isExecutable(candidate))
    {
        return candidate;
    }

    // Build-tree relative paths
    for(const char* prefix : {"../..", "../depth_server/build", "../../depth_server/build", "depth_server/build"})
    {
        fs::path p = fs::absolute(fs::path(prefix) / "depth_server");
        if(isExecutable(p))
        {
            return p;
        }
    }

    return {};
}

std::filesystem::path makeLogPath(pid_t pid)
{
    namespace fs = std::filesystem;
    const char* tmpdir = std::getenv("TMPDIR");
    if(tmpdir == nullptr || tmpdir[0] == '\0')
    {
        tmpdir = "/tmp";
    }
    return fs::path(tmpdir) / ("splat_server_" + std::to_string(static_cast<int>(pid)) + ".log");
}

}  // namespace

LocalSplatServerManager::LocalSplatServerManager() = default;

LocalSplatServerManager::~LocalSplatServerManager()
{
    stop();
}

int LocalSplatServerManager::start(const std::string& modelIdentifier)
{
    if(m_running.load())
    {
        LOGW("LocalSplatServerManager: already running on port %d\n", m_port);
        return m_port;
    }

    const std::filesystem::path binary = locateDepthServerBinary();
    if(binary.empty() || !std::filesystem::exists(binary))
    {
        LOGE("LocalSplatServerManager: depth_server binary not found in PATH or build dir\n");
        return -1;
    }

    const int port = findFreePort();
    if(port < 0)
    {
        LOGE("LocalSplatServerManager: no free port in range [%d, %d]\n", kSplatPortStart, kSplatPortEnd);
        return -1;
    }

    m_modelIdentifier = modelIdentifier;
    m_port             = port;

    const std::string port_str = std::to_string(port);

    const pid_t pid = ::fork();
    if(pid < 0)
    {
        LOGE("LocalSplatServerManager: fork failed: %s\n", std::strerror(errno));
        m_port = -1;
        return -1;
    }

    if(pid == 0)
    {
        // Child: redirect stdout/stderr to log file, then exec.
        const std::filesystem::path logPath = makeLogPath(::getpid());
        const int fd = ::open(logPath.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if(fd >= 0)
        {
            ::dup2(fd, STDOUT_FILENO);
            ::dup2(fd, STDERR_FILENO);
            ::close(fd);
        }

        std::vector<const char*> argv;
        argv.push_back(binary.c_str());
        argv.push_back("--mode");
        argv.push_back("splat");
        argv.push_back("--port");
        argv.push_back(port_str.c_str());
        argv.push_back("--splat-model");
        argv.push_back(modelIdentifier.c_str());
        argv.push_back("--splat-workers");
        argv.push_back("1");
        argv.push_back("--splat-backend");
        argv.push_back("cpu");
        argv.push_back(nullptr);

        ::execv(binary.c_str(), const_cast<char* const*>(argv.data()));

        // execv only returns on failure
        LOGE("LocalSplatServerManager: execv('%s') failed: %s\n", binary.c_str(), std::strerror(errno));
        std::_Exit(1);
    }

    m_pid = pid;
    m_running.store(true);

    LOGI("LocalSplatServerManager: started depth_server --mode splat (PID %d, model '%s', port %d, log %s)\n",
         static_cast<int>(pid), modelIdentifier.c_str(), port, makeLogPath(pid).c_str());

    m_monitorThread = std::thread(&LocalSplatServerManager::monitorProcess, this);

    for(int attempt = 0; attempt < 50; ++attempt)
    {
        if(isPortListening(port))
        {
            LOGI("LocalSplatServerManager: splat server is ready on port %d\n", port);
            return port;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    LOGW("LocalSplatServerManager: splat server may not be ready on port %d (timeout)\n", port);
    return port;
}

void LocalSplatServerManager::stop()
{
    if(!m_running.load())
    {
        return;
    }

    if(m_pid > 0)
    {
        LOGI("LocalSplatServerManager: sending SIGTERM to PID %d\n", static_cast<int>(m_pid));
        ::kill(m_pid, SIGTERM);

        // Wait up to 5 seconds (50 * 100ms) for graceful exit.
        for(int i = 0; i < 50; ++i)
        {
            int   status = 0;
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
            LOGW("LocalSplatServerManager: sending SIGKILL to PID %d\n", static_cast<int>(m_pid));
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

void LocalSplatServerManager::monitorProcess()
{
    while(m_running.load())
    {
        int   status = 0;
        const pid_t result = ::waitpid(m_pid, &status, WNOHANG);
        if(result == m_pid)
        {
            if(WIFEXITED(status))
            {
                LOGW("LocalSplatServerManager: splat server exited with code %d\n", WEXITSTATUS(status));
            }
            else if(WIFSIGNALED(status))
            {
                LOGW("LocalSplatServerManager: splat server killed by signal %d\n", WTERMSIG(status));
            }
            m_running.store(false);
            m_pid = -1;
            break;
        }
        else if(result < 0 && errno != ECHILD)
        {
            LOGW("LocalSplatServerManager: waitpid error: %s\n", std::strerror(errno));
            m_running.store(false);
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

}  // namespace vk_viewer

#endif  // WITH_FREE_SPLATTER