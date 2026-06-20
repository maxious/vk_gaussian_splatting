#include "worker_monitor.h"

#include <nvutils/logger.hpp>

#include <sys/wait.h>

#include <chrono>
#include <cerrno>
#include <cstring>

namespace {

std::int64_t nowMs()
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

}  // namespace

WorkerMonitor::WorkerMonitor(WorkerPool& pool) : m_pool(pool) {}

WorkerMonitor::~WorkerMonitor()
{
    stop();
}

bool WorkerMonitor::start(const std::string& model_path, const std::string& backend)
{
    if(m_running.load())
    {
        return true;
    }

    m_modelPath = model_path;
    m_backend = backend;

    const size_t worker_count = m_pool.workerCount();
    const size_t slots = worker_count > 0 ? worker_count : static_cast<size_t>(m_pool.activeWorkers());
    m_restartCounts.assign(slots, 0);
    m_restartTimes.assign(slots, 0);
    m_restartHistory.assign(slots, std::deque<std::int64_t>{});

    m_running.store(true);
    m_monitorThread = std::thread(&WorkerMonitor::monitorLoop, this);
    LOGI("WorkerMonitor: started\n");
    return true;
}

void WorkerMonitor::stop()
{
    if(!m_running.exchange(false))
    {
        return;
    }

    if(m_monitorThread.joinable())
    {
        m_monitorThread.join();
    }
    LOGI("WorkerMonitor: stopped\n");
}

bool WorkerMonitor::isRunning() const
{
    return m_running.load();
}

void WorkerMonitor::monitorLoop()
{
    while(m_running.load())
    {
        const size_t worker_count = m_pool.workerCount();
        if(m_restartHistory.size() < worker_count)
        {
            m_restartCounts.resize(worker_count, 0);
            m_restartTimes.resize(worker_count, 0);
            m_restartHistory.resize(worker_count);
        }

        const std::int64_t now = nowMs();
        for(size_t i = 0; i < worker_count && m_running.load(); ++i)
        {
            const pid_t pid = m_pool.workerPid(i);
            if(pid <= 0)
            {
                continue;
            }

            int status = 0;
            const pid_t rc = waitpid(pid, &status, WNOHANG);
            if(rc == 0)
            {
                continue;
            }

            if(rc < 0 && errno != ECHILD)
            {
                LOGE("Worker %d: waitpid failed: %s\n", static_cast<int>(i), std::strerror(errno));
                continue;
            }

            if(rc > 0 && WIFEXITED(status))
            {
                LOGE("Worker %d: exited with code %d\n", static_cast<int>(i), WEXITSTATUS(status));
            }
            else if(rc > 0 && WIFSIGNALED(status))
            {
                LOGE("Worker %d: terminated by signal %d\n", static_cast<int>(i), WTERMSIG(status));
            }
            else
            {
                LOGE("Worker %d: crashed or stopped\n", static_cast<int>(i));
            }

            auto& history = m_restartHistory[i];
            while(!history.empty() && now - history.front() > RESTART_WINDOW_MS)
            {
                history.pop_front();
            }

            if(history.size() >= static_cast<size_t>(MAX_RESTARTS))
            {
                LOGE("Worker %d: too many restarts\n", static_cast<int>(i));
                m_pool.retireWorker(i);
                continue;
            }

            history.push_back(now);
            m_restartCounts[i] = static_cast<int>(history.size());
            m_restartTimes[i] = now;

            if(!m_pool.restartWorker(i))
            {
                LOGE("Worker %d: restart failed\n", static_cast<int>(i));
                m_pool.retireWorker(i);
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}
