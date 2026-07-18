#include "cloud_worker_pool.h"

#include "cloud_worker.h"

#include <nvutils/logger.hpp>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

namespace {

static constexpr uint32_t MAX_PENDING_CLOUD_JOBS = 32;

bool readAll(int fd, void* buffer, size_t size)
{
    auto* bytes = static_cast<std::uint8_t*>(buffer);
    size_t offset = 0;
    while (offset < size)
    {
        const ssize_t n = ::read(fd, bytes + offset, size - offset);
        if (n == 0)
            return false;
        if (n < 0)
        {
            if (errno == EINTR)
                continue;
            return false;
        }
        offset += static_cast<size_t>(n);
    }
    return true;
}

bool writeAll(int fd, const void* buffer, size_t size)
{
    const auto* bytes = static_cast<const std::uint8_t*>(buffer);
    size_t offset = 0;
    while (offset < size)
    {
        const ssize_t n = ::send(fd, bytes + offset, size - offset, MSG_NOSIGNAL);
        if (n < 0)
        {
            if (errno == EINTR)
                continue;
            return false;
        }
        offset += static_cast<size_t>(n);
    }
    return true;
}

std::string homeCacheDir()
{
    const char* home = std::getenv("HOME");
    if (home == nullptr || home[0] == '\0')
        return "/tmp/depth_server/clouds";
    return std::string(home) + "/.cache/depth_server/clouds";
}

enum CloudMsgType : std::uint8_t {
    CLOUD_MSG_PROGRESS = 0,
    CLOUD_MSG_RESULT   = 1,
    CLOUD_MSG_ERROR    = 2,
};

}  // namespace

CloudWorkerPool::~CloudWorkerPool()
{
    shutdown();
}

void CloudWorkerPool::setWorkerFunc(CloudWorkerFunc func)
{
    m_workerFunc = func;
}

bool CloudWorkerPool::initialize(const std::string& model_path, int num_workers,
                                  const std::string& backend)
{
    shutdown();

    if (model_path.empty())
    {
        LOGE("CloudWorkerPool: model path is empty\n");
        return false;
    }

    m_modelPath     = model_path;
    m_backend       = backend;
    m_cloudCacheDir = homeCacheDir();
    m_workDir       = "/tmp/depth_server/cloud_work";

    std::error_code ec;
    std::filesystem::create_directories(m_cloudCacheDir, ec);
    if (ec)
    {
        LOGE("CloudWorkerPool: cannot create cache dir %s: %s\n",
             m_cloudCacheDir.c_str(), ec.message().c_str());
        return false;
    }
    std::filesystem::create_directories(m_workDir, ec);

    const unsigned int hw_threads = std::thread::hardware_concurrency();
    m_numWorkers = (num_workers > 0) ? num_workers
                                     : static_cast<int>(std::max(1u, hw_threads));
    m_nextJobId.store(1);
    m_pendingQueue.clear();
    m_results.clear();
    m_inFlight.clear();
    m_workers.clear();
    m_workers.reserve(static_cast<size_t>(m_numWorkers));

    for (int worker_id = 0; worker_id < m_numWorkers; ++worker_id)
    {
        CloudWorkerProcess worker{};
        if (!spawnWorker(worker_id, worker))
        {
            LOGE("CloudWorkerPool: failed to spawn worker %d\n", worker_id);
            shutdown();
            return false;
        }
        m_workers.push_back(std::move(worker));
    }

    LOGI("CloudWorkerPool: initialized %d workers (backend=%s, cache=%s)\n",
         m_numWorkers, backend.c_str(), m_cloudCacheDir.c_str());
    return true;
}

bool CloudWorkerPool::spawnWorker(int id, CloudWorkerProcess& worker_out)
{
    int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == -1)
    {
        LOGE("CloudWorkerPool: socketpair failed: %s\n", std::strerror(errno));
        return false;
    }

    const pid_t pid = fork();
    if (pid == -1)
    {
        LOGE("CloudWorkerPool: fork failed: %s\n", std::strerror(errno));
        close(sv[0]);
        close(sv[1]);
        return false;
    }

    if (pid == 0)
    {
        close(sv[0]);
        if (m_workerFunc)
            m_workerFunc(sv[1], m_modelPath, m_backend, m_cloudCacheDir, m_workDir);
        else
            CloudWorker::run(sv[1], m_modelPath, m_backend, m_cloudCacheDir, m_workDir);
        ::_exit(0);
    }

    close(sv[1]);

    worker_out = CloudWorkerProcess{};
    worker_out.id        = id;
    worker_out.pid       = pid;
    worker_out.socket_fd = sv[0];
    worker_out.busy      = false;
    worker_out.current_job_id = 0;

    LOGI("CloudWorkerPool: worker %d started (pid=%d)\n", id, static_cast<int>(pid));
    return true;
}

uint32_t CloudWorkerPool::submitJob(const std::vector<std::string>& frame_paths,
                                     const CloudRequestOptions& options)
{
    std::scoped_lock lock(m_mutex);
    if (frame_paths.empty())
    {
        LOGE("CloudWorkerPool: submitJob with no frame paths\n");
        return 0;
    }
    if (m_pendingQueue.size() >= MAX_PENDING_CLOUD_JOBS)
    {
        LOGW("CloudWorkerPool: queue full (%u)\n", MAX_PENDING_CLOUD_JOBS);
        return 0;
    }

    const uint32_t job_id = m_nextJobId.fetch_add(1);

    PendingCloudJob job{};
    job.job_id      = job_id;
    job.frame_paths = frame_paths;
    job.options     = options;
    m_pendingQueue.push_back(std::move(job));

    for (size_t i = 0; i < m_workers.size(); ++i)
    {
        const size_t idx = i % m_workers.size();
        CloudWorkerProcess& w = m_workers[idx];
        if (!w.busy && w.socket_fd >= 0)
        {
            PendingCloudJob next_job = std::move(m_pendingQueue.front());
            m_pendingQueue.pop_front();
            if (dispatchToWorker(static_cast<int>(idx), next_job))
            {
                m_inFlight.emplace_back(next_job.job_id, static_cast<uint32_t>(idx));
            }
            else
            {
                m_pendingQueue.push_front(std::move(next_job));
            }
            break;
        }
    }

    LOGD("CloudWorkerPool: submitted job %u (%zu frames, queue=%zu)\n",
         job_id, frame_paths.size(), m_pendingQueue.size());
    return job_id;
}

bool CloudWorkerPool::dispatchToWorker(int worker_idx, const PendingCloudJob& job)
{
    if (worker_idx < 0 || static_cast<size_t>(worker_idx) >= m_workers.size())
        return false;

    CloudWorkerProcess& w = m_workers[static_cast<size_t>(worker_idx)];
    if (w.socket_fd < 0)
        return false;

    // Wire format: job_id(4) + CloudRequestOptions(64) + [path_len(4) + path_bytes]*n_frames
    if (!writeAll(w.socket_fd, &job.job_id, sizeof(job.job_id))
        || !writeAll(w.socket_fd, &job.options, sizeof(job.options)))
    {
        LOGE("CloudWorkerPool: failed dispatch header to worker %d: %s\n",
             w.id, std::strerror(errno));
        close(w.socket_fd);
        w.socket_fd = -1;
        return false;
    }

    for (const auto& path : job.frame_paths)
    {
        const uint32_t len = static_cast<uint32_t>(path.size());
        if (!writeAll(w.socket_fd, &len, sizeof(len))
            || (len != 0 && !writeAll(w.socket_fd, path.data(), path.size())))
        {
            LOGE("CloudWorkerPool: failed dispatch path to worker %d: %s\n",
                 w.id, std::strerror(errno));
            close(w.socket_fd);
            w.socket_fd = -1;
            return false;
        }
    }

    w.busy = true;
    w.current_job_id = job.job_id;
    LOGD("CloudWorkerPool: dispatched job %u to worker %d\n", job.job_id, w.id);
    return true;
}

bool CloudWorkerPool::pollResult(CloudJobResult& out_result)
{
    std::scoped_lock lock(m_mutex);
    for (size_t worker_idx = 0; worker_idx < m_workers.size(); ++worker_idx)
    {
        CloudWorkerProcess& w = m_workers[worker_idx];
        if (w.socket_fd < 0)
        {
            respawnWorker(worker_idx);
            continue;
        }
        if (!w.busy)
            continue;

        pollfd pfd{};
        pfd.fd = w.socket_fd;
        pfd.events = POLLIN;
        const int prc = poll(&pfd, 1, 0);
        if (prc < 0)
        {
            if (errno == EINTR)
                continue;
            LOGE("CloudWorkerPool: poll failed for worker %d: %s\n",
                 w.id, std::strerror(errno));
            close(w.socket_fd);
            w.socket_fd = -1;
            w.busy = false;
            w.current_job_id = 0;
            respawnWorker(worker_idx);
            continue;
        }
        if (prc == 0 || (pfd.revents & POLLIN) == 0)
        {
            if ((pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0)
            {
                // Worker closed its write end. Drain any buffered data
                // (e.g. an error message) before discarding the socket.
                CloudJobResult drain{};
                if (readFromWorker(static_cast<int>(worker_idx), drain))
                {
                    w.busy = false;
                    w.current_job_id = 0;
                    m_results[drain.job_id] = drain;
                    out_result = drain;
                    close(w.socket_fd);
                    w.socket_fd = -1;
                    respawnWorker(worker_idx);
                    return true;
                }
                close(w.socket_fd);
                w.socket_fd = -1;
                w.busy = false;
                w.current_job_id = 0;
                respawnWorker(worker_idx);
            }
            continue;
        }

        CloudJobResult result{};
        if (!readFromWorker(static_cast<int>(worker_idx), result))
        {
            w.busy = false;
            w.current_job_id = 0;
            continue;
        }

        if (result.error == "__progress__")
        {
            LOGD("CloudWorkerPool: job %u progress update consumed\n", result.job_id);
            continue;
        }

        w.busy = false;
        w.current_job_id = 0;
        for (auto it = m_inFlight.begin(); it != m_inFlight.end(); ++it)
        {
            if (it->first == result.job_id)
            {
                m_inFlight.erase(it);
                break;
            }
        }

        if (!m_pendingQueue.empty())
        {
            PendingCloudJob next_job = std::move(m_pendingQueue.front());
            m_pendingQueue.pop_front();
            if (!dispatchToWorker(static_cast<int>(worker_idx), next_job))
            {
                m_pendingQueue.push_front(std::move(next_job));
            }
            else
            {
                m_inFlight.emplace_back(next_job.job_id,
                                        static_cast<uint32_t>(worker_idx));
            }
        }

        m_results[result.job_id] = result;
        out_result = m_results[result.job_id];
        return true;
    }

    // Also check m_results for pre-stored results (e.g. from cancelJob)
    for (auto it = m_results.begin(); it != m_results.end(); ++it)
    {
        out_result = it->second;
        m_results.erase(it);
        return true;
    }

    return false;
}

bool CloudWorkerPool::readFromWorker(int worker_idx, CloudJobResult& out)
{
    if (worker_idx < 0 || static_cast<size_t>(worker_idx) >= m_workers.size())
        return false;
    CloudWorkerProcess& w = m_workers[static_cast<size_t>(worker_idx)];

    uint32_t job_id = 0;
    if (!readAll(w.socket_fd, &job_id, sizeof(job_id)))
    {
        LOGE("CloudWorkerPool: failed reading job_id from worker %d\n", w.id);
        return false;
    }

    uint8_t type = 0;
    if (!readAll(w.socket_fd, &type, sizeof(type)))
    {
        LOGE("CloudWorkerPool: failed reading msg type from worker %d\n", w.id);
        return false;
    }

    out.job_id = job_id;

    if (type == CLOUD_MSG_PROGRESS)
    {
        uint8_t stage = 0;
        uint8_t percent = 0;
        if (!readAll(w.socket_fd, &stage, sizeof(stage))
            || !readAll(w.socket_fd, &percent, sizeof(percent)))
        {
            LOGE("CloudWorkerPool: failed reading progress from worker %d\n", w.id);
            return false;
        }
        LOGD("CloudWorkerPool: job %u progress stage=%u percent=%u\n",
             job_id, stage, percent);
        out.error = "__progress__";
        return true;
    }

    if (type == CLOUD_MSG_RESULT)
    {
        uint32_t n_points = 0;
        uint32_t path_len = 0;
        if (!readAll(w.socket_fd, &n_points, sizeof(n_points))
            || !readAll(w.socket_fd, &path_len, sizeof(path_len)))
        {
            LOGE("CloudWorkerPool: failed reading result header from worker %d\n", w.id);
            return false;
        }
        if (path_len > 4096)
        {
            LOGE("CloudWorkerPool: path_len too large (%u) from worker %d\n",
                 path_len, w.id);
            return false;
        }
        std::string path(path_len, '\0');
        if (path_len != 0 && !readAll(w.socket_fd, path.data(), path_len))
        {
            LOGE("CloudWorkerPool: failed reading path from worker %d\n", w.id);
            return false;
        }
        out.output_path = std::move(path);
        out.n_points = n_points;
        out.error.clear();
        return true;
    }

    if (type == CLOUD_MSG_ERROR)
    {
        uint32_t error_code = 0;
        uint32_t err_len = 0;
        if (!readAll(w.socket_fd, &error_code, sizeof(error_code))
            || !readAll(w.socket_fd, &err_len, sizeof(err_len)))
        {
            LOGE("CloudWorkerPool: failed reading error from worker %d\n", w.id);
            return false;
        }
        if (err_len > 4096)
        {
            LOGE("CloudWorkerPool: err_len too large (%u) from worker %d\n",
                 err_len, w.id);
            return false;
        }
        std::string err(err_len, '\0');
        if (err_len != 0 && !readAll(w.socket_fd, err.data(), err_len))
        {
            LOGE("CloudWorkerPool: failed reading error body from worker %d\n", w.id);
            return false;
        }
        out.error = std::move(err);
        out.output_path.clear();
        out.n_points = 0;
        return true;
    }

    LOGE("CloudWorkerPool: unknown msg type %u from worker %d\n", type, w.id);
    return false;
}

bool CloudWorkerPool::cancelJob(uint32_t job_id)
{
    std::scoped_lock lock(m_mutex);

    for (auto it = m_pendingQueue.begin(); it != m_pendingQueue.end(); ++it)
    {
        if (it->job_id == job_id)
        {
            m_pendingQueue.erase(it);
            LOGI("CloudWorkerPool: cancelled queued job %u\n", job_id);
            CloudJobResult cancelled{};
            cancelled.job_id = job_id;
            cancelled.error  = "cancelled";
            m_results[job_id] = std::move(cancelled);
            return true;
        }
    }

    for (const auto& entry : m_inFlight)
    {
        if (entry.first == job_id)
        {
            const uint32_t worker_idx = entry.second;
            if (worker_idx < m_workers.size())
            {
                CloudWorkerProcess& w = m_workers[worker_idx];
                if (w.pid > 0)
                    ::kill(w.pid, SIGTERM);
                if (w.socket_fd >= 0)
                {
                    close(w.socket_fd);
                    w.socket_fd = -1;
                }
                w.busy = false;
                w.current_job_id = 0;
            }
            break;
        }
    }

    for (auto it = m_inFlight.begin(); it != m_inFlight.end(); ++it)
    {
        if (it->first == job_id)
        {
            m_inFlight.erase(it);
            break;
        }
    }

    CloudJobResult cancelled{};
    cancelled.job_id = job_id;
    cancelled.error  = "cancelled";
    m_results[job_id] = std::move(cancelled);
    LOGI("CloudWorkerPool: cancelled in-flight job %u\n", job_id);
    return true;
}

void CloudWorkerPool::respawnWorker(size_t index)
{
    if (index >= m_workers.size())
        return;

    CloudWorkerProcess& old = m_workers[index];
    if (old.pid > 0)
    {
        int status = 0;
        ::waitpid(old.pid, &status, WNOHANG);
        old.pid = -1;
    }
    if (old.socket_fd >= 0)
    {
        close(old.socket_fd);
        old.socket_fd = -1;
    }
    old.busy = false;
    old.current_job_id = 0;

    CloudWorkerProcess replacement{};
    if (!spawnWorker(static_cast<int>(index), replacement))
    {
        LOGE("CloudWorkerPool: failed to respawn worker %zu\n", index);
        return;
    }
    m_workers[index] = std::move(replacement);
    LOGI("CloudWorkerPool: auto-respawned worker %zu (pid=%d)\n",
         index, static_cast<int>(m_workers[index].pid));
}

bool CloudWorkerPool::restartWorker(size_t index)
{
    std::scoped_lock lock(m_mutex);
    if (index >= m_workers.size())
        return false;

    CloudWorkerProcess& w = m_workers[index];
    const int old_pid = w.pid;

    if (w.socket_fd >= 0)
    {
        close(w.socket_fd);
        w.socket_fd = -1;
    }
    if (w.pid > 0)
    {
        ::kill(w.pid, SIGTERM);
        int status = 0;
        for (int attempt = 0; attempt < 20; ++attempt)
        {
            if (waitpid(w.pid, &status, WNOHANG) != 0)
                break;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    w.busy = false;
    w.current_job_id = 0;

    CloudWorkerProcess replacement{};
    if (!spawnWorker(w.id, replacement))
    {
        w.pid = -1;
        return false;
    }
    m_workers[index] = std::move(replacement);
    LOGI("CloudWorkerPool: restarted worker %d (old pid=%d, new pid=%d)\n",
         m_workers[index].id, old_pid, static_cast<int>(m_workers[index].pid));
    return true;
}

int CloudWorkerPool::activeWorkers() const
{
    std::scoped_lock lock(m_mutex);
    int active = 0;
    for (const CloudWorkerProcess& w : m_workers)
    {
        if (w.socket_fd >= 0)
            ++active;
    }
    return active;
}

size_t CloudWorkerPool::workerCount() const
{
    std::scoped_lock lock(m_mutex);
    return m_workers.size();
}

pid_t CloudWorkerPool::workerPid(size_t index) const
{
    std::scoped_lock lock(m_mutex);
    if (index >= m_workers.size())
        return -1;
    return m_workers[index].pid;
}

int CloudWorkerPool::queueDepth() const
{
    std::scoped_lock lock(m_mutex);
    return static_cast<int>(m_pendingQueue.size());
}

bool CloudWorkerPool::isJobComplete(uint32_t job_id) const
{
    std::scoped_lock lock(m_mutex);
    return m_results.find(job_id) != m_results.end();
}

void CloudWorkerPool::shutdown()
{
    std::scoped_lock lock(m_mutex);
    for (CloudWorkerProcess& w : m_workers)
    {
        if (w.pid > 0)
            ::kill(w.pid, SIGTERM);
    }

    for (CloudWorkerProcess& w : m_workers)
    {
        if (w.pid > 0)
        {
            int status = 0;
            for (int attempt = 0; attempt < 20; ++attempt)
            {
                if (waitpid(w.pid, &status, WNOHANG) != 0)
                {
                    w.pid = -1;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            if (w.pid > 0)
            {
                ::kill(w.pid, SIGKILL);
                waitpid(w.pid, &status, 0);
                w.pid = -1;
            }
        }
        if (w.socket_fd >= 0)
        {
            close(w.socket_fd);
            w.socket_fd = -1;
        }
    }

    m_workers.clear();
    m_pendingQueue.clear();
    m_results.clear();
    m_inFlight.clear();
    m_numWorkers = 0;
}
