#include "local_cloud_server_manager.h"

#ifdef WITH_TCP_DEPTH

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

constexpr int kCloudPortStart = 9002;
constexpr int kCloudPortEnd   = 9011;

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
    for(int port = kCloudPortStart; port <= kCloudPortEnd; ++port)
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
                LOGW("LocalCloudServerManager: depth_server at '%s' exists but is not executable\n", p.c_str());
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

    fs::path candidate = fs::current_path() / "depth_server";
    if(isExecutable(candidate))
    {
        return candidate;
    }

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
    return fs::path(tmpdir) / ("cloud_server_" + std::to_string(static_cast<int>(pid)) + ".log");
}

}  // namespace

LocalCloudServerManager::LocalCloudServerManager() = default;

LocalCloudServerManager::~LocalCloudServerManager()
{
    stop();
}

bool LocalCloudServerManager::start(const std::string& modelPath, int port, int numWorkers)
{
    if(m_running.load())
    {
        LOGW("LocalCloudServerManager: already running on port %d\n", m_port);
        return true;
    }

    const std::filesystem::path binary = locateDepthServerBinary();
    if(binary.empty() || !std::filesystem::exists(binary))
    {
        LOGE("LocalCloudServerManager: depth_server binary not found\n");
        return false;
    }

    int actualPort = port;
    if(actualPort <= 0)
    {
        actualPort = findFreePort();
    }
    if(actualPort < 0)
    {
        LOGE("LocalCloudServerManager: no free port in range [%d, %d]\n", kCloudPortStart, kCloudPortEnd);
        return false;
    }

    m_modelPath = modelPath;
    m_port      = actualPort;

    const std::string port_str = std::to_string(actualPort);
    const std::string workers_str = std::to_string(numWorkers);

    const pid_t pid = ::fork();
    if(pid < 0)
    {
        LOGE("LocalCloudServerManager: fork failed: %s\n", std::strerror(errno));
        m_port = -1;
        return false;
    }

    if(pid == 0)
    {
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
        argv.push_back("cloud");
        argv.push_back("--port");
        argv.push_back(port_str.c_str());
        argv.push_back("--cloud-model");
        argv.push_back(modelPath.c_str());
        argv.push_back("--cloud-workers");
        argv.push_back(workers_str.c_str());
        argv.push_back(nullptr);

        ::execvp(binary.c_str(), const_cast<char* const*>(argv.data()));

        LOGE("LocalCloudServerManager: execvp('%s') failed: %s\n", binary.c_str(), std::strerror(errno));
        std::_Exit(1);
    }

    m_pid = pid;
    m_running.store(true);

    LOGI("LocalCloudServerManager: started depth_server --mode cloud (PID %d, model '%s', port %d, log %s)\n",
         static_cast<int>(pid), modelPath.c_str(), actualPort, makeLogPath(pid).c_str());

    m_monitorThread = std::thread(&LocalCloudServerManager::monitorProcess, this);

    for(int attempt = 0; attempt < 50; ++attempt)
    {
        if(isPortListening(actualPort))
        {
            LOGI("LocalCloudServerManager: cloud server is ready on port %d\n", actualPort);
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    LOGW("LocalCloudServerManager: cloud server may not be ready on port %d (timeout)\n", actualPort);
    return true;
}

void LocalCloudServerManager::stop()
{
    if(!m_running.load())
    {
        return;
    }

    if(m_pid > 0)
    {
        LOGI("LocalCloudServerManager: sending SIGTERM to PID %d\n", static_cast<int>(m_pid));
        ::kill(m_pid, SIGTERM);

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
            LOGW("LocalCloudServerManager: sending SIGKILL to PID %d\n", static_cast<int>(m_pid));
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

void LocalCloudServerManager::monitorProcess()
{
    while(m_running.load())
    {
        int   status = 0;
        const pid_t result = ::waitpid(m_pid, &status, WNOHANG);
        if(result == m_pid)
        {
            if(WIFEXITED(status))
            {
                LOGW("LocalCloudServerManager: cloud server exited with code %d\n", WEXITSTATUS(status));
            }
            else if(WIFSIGNALED(status))
            {
                LOGW("LocalCloudServerManager: cloud server killed by signal %d\n", WTERMSIG(status));
            }
            m_running.store(false);
            m_pid = -1;
            break;
        }
        else if(result < 0 && errno != ECHILD)
        {
            LOGW("LocalCloudServerManager: waitpid error: %s\n", std::strerror(errno));
            m_running.store(false);
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

}  // namespace vk_viewer

#endif  // WITH_TCP_DEPTH
