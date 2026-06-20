#pragma once

#include "worker_pool.h"

#include <atomic>
#include <cstdint>
#include <deque>
#include <string>
#include <thread>
#include <vector>

class WorkerMonitor {
public:
    WorkerMonitor(WorkerPool& pool);
    ~WorkerMonitor();

    bool start(const std::string& model_path, const std::string& backend);
    void stop();
    bool isRunning() const;

private:
    void monitorLoop();

    WorkerPool& m_pool;
    std::string m_modelPath;
    std::string m_backend;
    std::thread m_monitorThread;
    std::atomic<bool> m_running{false};
    std::vector<int> m_restartCounts;
    std::vector<int64_t> m_restartTimes;
    std::vector<std::deque<int64_t>> m_restartHistory;
    static constexpr int MAX_RESTARTS = 5;
    static constexpr int RESTART_WINDOW_MS = 60000;
};
