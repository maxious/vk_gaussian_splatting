#pragma once

#ifdef WITH_TCP_DEPTH

#include <atomic>
#include <string>
#include <thread>

#include <sys/types.h>

namespace vk_viewer {

class LocalCloudServerManager {
public:
    LocalCloudServerManager();
    ~LocalCloudServerManager();

    LocalCloudServerManager(const LocalCloudServerManager&)            = delete;
    LocalCloudServerManager& operator=(const LocalCloudServerManager&) = delete;

    bool start(const std::string& modelPath, int port, int numWorkers = 1);
    void stop();

    bool isRunning() const { return m_running.load(); }
    int  port() const { return m_port; }
    const std::string& modelPath() const { return m_modelPath; }

private:
    void monitorProcess();

    std::atomic<bool> m_running{false};
    std::string       m_modelPath;
    int               m_port = -1;
    pid_t             m_pid = -1;
    std::thread       m_monitorThread;
};

}  // namespace vk_viewer

#endif  // WITH_TCP_DEPTH
