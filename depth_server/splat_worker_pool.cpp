#include "splat_worker_pool.h"

#include "splat_worker.h"

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

bool readAll(int fd, void* buffer, size_t size)
{
    auto* bytes = static_cast<std::uint8_t*>(buffer);
    size_t offset = 0;
    while(offset < size)
    {
        const ssize_t n = ::read(fd, bytes + offset, size - offset);
        if(n == 0)
        {
            return false;
        }
        if(n < 0)
        {
            if(errno == EINTR)
            {
                continue;
            }
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
    while(offset < size)
    {
        const ssize_t n = ::write(fd, bytes + offset, size - offset);
        if(n < 0)
        {
            if(errno == EINTR)
            {
                continue;
            }
            return false;
        }
        offset += static_cast<size_t>(n);
    }
    return true;
}

std::string homeCacheDir()
{
    const char* home = std::getenv("HOME");
    if(home == nullptr || home[0] == '\0')
    {
        return "/tmp/depth_server/splats";
    }
    return std::string(home) + "/.cache/depth_server/splats";
}

}  // namespace

SplatWorkerPool::~SplatWorkerPool()
{
    shutdown();
}

bool SplatWorkerPool::initialize(const std::string& model_path, int num_workers, const std::string& backend)
{
    shutdown();

    if(model_path.empty())
    {
        LOGE("SplatWorkerPool: model path is empty\n");
        return false;
    }

    m_modelPath     = model_path;
    m_backend       = backend;
    m_splatCacheDir = homeCacheDir();

    std::error_code ec;
    std::filesystem::create_directories(m_splatCacheDir, ec);
    if(ec)
    {
        LOGE("SplatWorkerPool: cannot create cache dir %s: %s\n", m_splatCacheDir.c_str(), ec.message().c_str());
        return false;
    }

    const unsigned int hw_threads = std::thread::hardware_concurrency();
    m_numWorkers = (num_workers > 0) ? num_workers : static_cast<int>(std::max(1u, hw_threads));
    m_nextJobId.store(1);
    m_pendingQueue.clear();
    m_results.clear();
    m_inFlight.clear();
    m_workers.clear();
    m_workers.reserve(static_cast<size_t>(m_numWorkers));

    for(int worker_id = 0; worker_id < m_numWorkers; ++worker_id)
    {
        SplatWorkerProcess worker{};
        if(!spawnWorker(worker_id, worker))
        {
            LOGE("SplatWorkerPool: failed to spawn worker %d\n", worker_id);
            shutdown();
            return false;
        }
        m_workers.push_back(std::move(worker));
    }

    LOGI("SplatWorkerPool: initialized %d workers (backend=%s, cache=%s)\n",
         m_numWorkers, backend.c_str(), m_splatCacheDir.c_str());
    return true;
}

bool SplatWorkerPool::spawnWorker(int id, SplatWorkerProcess& worker_out)
{
    int sv[2];
    if(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == -1)
    {
        LOGE("SplatWorkerPool: socketpair failed: %s\n", std::strerror(errno));
        return false;
    }

    const pid_t pid = fork();
    if(pid == -1)
    {
        LOGE("SplatWorkerPool: fork failed: %s\n", std::strerror(errno));
        close(sv[0]);
        close(sv[1]);
        return false;
    }

    if(pid == 0)
    {
        close(sv[0]);
        SplatWorker::run(sv[1], m_modelPath, m_backend, m_splatCacheDir);
        _exit(0);
    }

    close(sv[1]);

    worker_out = SplatWorkerProcess{};
    worker_out.id        = id;
    worker_out.pid       = pid;
    worker_out.socket_fd = sv[0];
    worker_out.busy      = false;
    worker_out.current_job_id = 0;

    LOGI("SplatWorkerPool: worker %d started (pid=%d)\n", id, static_cast<int>(pid));
    return true;
}

uint32_t SplatWorkerPool::submitJob(const std::vector<std::vector<uint8_t>>& images,
                                    const SplatRequestOptions& options)
{
    std::scoped_lock lock(m_mutex);
    if(images.empty())
    {
        LOGE("SplatWorkerPool: submitJob with no images\n");
        return 0;
    }
    if(m_pendingQueue.size() >= MAX_PENDING_SPLAT_JOBS)
    {
        LOGW("SplatWorkerPool: queue full (%u)\n", static_cast<uint32_t>(MAX_PENDING_SPLAT_JOBS));
        return 0;
    }

    const uint32_t job_id = m_nextJobId.fetch_add(1);

    PendingSplatJob job{};
    job.job_id  = job_id;
    job.images  = images;
    job.options = options;
    m_pendingQueue.push_back(std::move(job));

    for(size_t i = 0; i < m_workers.size(); ++i)
    {
        const size_t idx = (i + 0) % m_workers.size();
        SplatWorkerProcess& w = m_workers[idx];
        if(!w.busy && w.socket_fd >= 0)
        {
            PendingSplatJob next_job = std::move(m_pendingQueue.front());
            m_pendingQueue.pop_front();
            if(dispatchToWorker(static_cast<int>(idx), next_job))
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

    LOGD("SplatWorkerPool: submitted job %u (%zu images, queue=%zu)\n",
         job_id, images.size(), m_pendingQueue.size());
    return job_id;
}

bool SplatWorkerPool::dispatchToWorker(int worker_idx, const PendingSplatJob& job)
{
    if(worker_idx < 0 || static_cast<size_t>(worker_idx) >= m_workers.size())
    {
        return false;
    }
    SplatWorkerProcess& w = m_workers[static_cast<size_t>(worker_idx)];
    if(w.socket_fd < 0)
    {
        return false;
    }

    const uint32_t n_views = static_cast<uint32_t>(job.images.size());
    if(!writeAll(w.socket_fd, &job.job_id, sizeof(job.job_id))
       || !writeAll(w.socket_fd, &n_views, sizeof(n_views))
       || !writeAll(w.socket_fd, &job.options, sizeof(job.options)))
    {
        LOGE("SplatWorkerPool: failed dispatch header to worker %d: %s\n", w.id, std::strerror(errno));
        close(w.socket_fd);
        w.socket_fd = -1;
        return false;
    }

    for(const auto& img : job.images)
    {
        const uint32_t len = static_cast<uint32_t>(img.size());
        if(!writeAll(w.socket_fd, &len, sizeof(len))
           || (len != 0 && !writeAll(w.socket_fd, img.data(), img.size())))
        {
            LOGE("SplatWorkerPool: failed dispatch image to worker %d: %s\n", w.id, std::strerror(errno));
            close(w.socket_fd);
            w.socket_fd = -1;
            return false;
        }
    }

    w.busy = true;
    w.current_job_id = job.job_id;
    LOGD("SplatWorkerPool: dispatched job %u to worker %d\n", job.job_id, w.id);
    return true;
}

bool SplatWorkerPool::pollResult(SplatJobResult& out_result)
{
    std::scoped_lock lock(m_mutex);
    for(size_t worker_idx = 0; worker_idx < m_workers.size(); ++worker_idx)
    {
        SplatWorkerProcess& w = m_workers[worker_idx];
        if(w.socket_fd < 0 || !w.busy)
        {
            continue;
        }

        pollfd pfd{};
        pfd.fd = w.socket_fd;
        pfd.events = POLLIN;
        const int prc = poll(&pfd, 1, 0);
        if(prc < 0)
        {
            if(errno == EINTR)
            {
                continue;
            }
            LOGE("SplatWorkerPool: poll failed for worker %d: %s\n", w.id, std::strerror(errno));
            continue;
        }
        if(prc == 0 || (pfd.revents & POLLIN) == 0)
        {
            continue;
        }

        SplatJobResult result{};
        if(!readFromWorker(static_cast<int>(worker_idx), result))
        {
            w.busy = false;
            w.current_job_id = 0;
            continue;
        }

        if(result.error == "__progress__")
        {
            LOGD("SplatWorkerPool: job %u progress update consumed\n", result.job_id);
            continue;
        }

        if(result.error.empty())
        {
            w.busy = false;
            w.current_job_id = 0;
            for(auto it = m_inFlight.begin(); it != m_inFlight.end(); ++it)
            {
                if(it->first == result.job_id)
                {
                    m_inFlight.erase(it);
                    break;
                }
            }

            if(!m_pendingQueue.empty())
            {
                PendingSplatJob next_job = std::move(m_pendingQueue.front());
                m_pendingQueue.pop_front();
                if(!dispatchToWorker(static_cast<int>(worker_idx), next_job))
                {
                    m_pendingQueue.push_front(std::move(next_job));
                }
                else
                {
                    m_inFlight.emplace_back(next_job.job_id, static_cast<uint32_t>(worker_idx));
                }
            }
        }
        else
        {
            w.busy = false;
            w.current_job_id = 0;
            for(auto it = m_inFlight.begin(); it != m_inFlight.end(); ++it)
            {
                if(it->first == result.job_id)
                {
                    m_inFlight.erase(it);
                    break;
                }
            }

            if(!m_pendingQueue.empty())
            {
                PendingSplatJob next_job = std::move(m_pendingQueue.front());
                m_pendingQueue.pop_front();
                if(!dispatchToWorker(static_cast<int>(worker_idx), next_job))
                {
                    m_pendingQueue.push_front(std::move(next_job));
                }
                else
                {
                    m_inFlight.emplace_back(next_job.job_id, static_cast<uint32_t>(worker_idx));
                }
            }
        }

        m_results[result.job_id] = std::move(result);
        out_result = m_results[result.job_id];
        return true;
    }

    return false;
}

bool SplatWorkerPool::readFromWorker(int worker_idx, SplatJobResult& out)
{
    if(worker_idx < 0 || static_cast<size_t>(worker_idx) >= m_workers.size())
    {
        return false;
    }
    SplatWorkerProcess& w = m_workers[static_cast<size_t>(worker_idx)];

    uint32_t job_id = 0;
    if(!readAll(w.socket_fd, &job_id, sizeof(job_id)))
    {
        LOGE("SplatWorkerPool: failed reading job_id from worker %d\n", w.id);
        return false;
    }

    uint8_t type = 0;
    if(!readAll(w.socket_fd, &type, sizeof(type)))
    {
        LOGE("SplatWorkerPool: failed reading msg type from worker %d\n", w.id);
        return false;
    }

    out.job_id = job_id;

    if(type == 0)  // progress
    {
        uint8_t stage = 0;
        uint8_t percent = 0;
        if(!readAll(w.socket_fd, &stage, sizeof(stage))
           || !readAll(w.socket_fd, &percent, sizeof(percent)))
        {
            LOGE("SplatWorkerPool: failed reading progress from worker %d\n", w.id);
            return false;
        }
        LOGD("SplatWorkerPool: job %u progress stage=%u percent=%u\n", job_id, stage, percent);
        out.error = "__progress__";
        return true;
    }

    if(type == 1)  // result
    {
        uint32_t n_gaussians = 0;
        uint32_t path_len = 0;
        if(!readAll(w.socket_fd, &n_gaussians, sizeof(n_gaussians))
           || !readAll(w.socket_fd, &path_len, sizeof(path_len)))
        {
            LOGE("SplatWorkerPool: failed reading result header from worker %d\n", w.id);
            return false;
        }
        if(path_len > 4096)
        {
            LOGE("SplatWorkerPool: path_len too large (%u) from worker %d\n", path_len, w.id);
            return false;
        }
        std::string path(path_len, '\0');
        if(path_len != 0 && !readAll(w.socket_fd, path.data(), path_len))
        {
            LOGE("SplatWorkerPool: failed reading path from worker %d\n", w.id);
            return false;
        }
        out.output_path = std::move(path);
        out.n_gaussians = n_gaussians;
        out.error.clear();
        return true;
    }

    if(type == 2)  // error
    {
        uint32_t err_len = 0;
        if(!readAll(w.socket_fd, &err_len, sizeof(err_len)))
        {
            LOGE("SplatWorkerPool: failed reading err_len from worker %d\n", w.id);
            return false;
        }
        if(err_len > 4096)
        {
            LOGE("SplatWorkerPool: err_len too large (%u) from worker %d\n", err_len, w.id);
            return false;
        }
        std::string err(err_len, '\0');
        if(err_len != 0 && !readAll(w.socket_fd, err.data(), err_len))
        {
            LOGE("SplatWorkerPool: failed reading error from worker %d\n", w.id);
            return false;
        }
        out.error = std::move(err);
        out.output_path.clear();
        out.n_gaussians = 0;
        return true;
    }

    LOGE("SplatWorkerPool: unknown msg type %u from worker %d\n", type, w.id);
    return false;
}

bool SplatWorkerPool::cancelJob(uint32_t job_id)
{
    std::scoped_lock lock(m_mutex);

    for(auto it = m_pendingQueue.begin(); it != m_pendingQueue.end(); ++it)
    {
        if(it->job_id == job_id)
        {
            m_pendingQueue.erase(it);
            LOGI("SplatWorkerPool: cancelled queued job %u\n", job_id);
            return true;
        }
    }

    for(const auto& entry : m_inFlight)
    {
        if(entry.first == job_id)
        {
            const uint32_t worker_idx = entry.second;
            if(worker_idx < m_workers.size())
            {
                SplatWorkerProcess& w = m_workers[worker_idx];
                if(w.pid > 0)
                {
                    ::kill(w.pid, SIGTERM);
                }
                if(w.socket_fd >= 0)
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

    for(auto it = m_inFlight.begin(); it != m_inFlight.end(); ++it)
    {
        if(it->first == job_id)
        {
            m_inFlight.erase(it);
            break;
        }
    }

    SplatJobResult cancelled{};
    cancelled.job_id = job_id;
    cancelled.error  = "cancelled";
    m_results[job_id] = std::move(cancelled);
    LOGI("SplatWorkerPool: cancelled in-flight job %u\n", job_id);
    return true;
}

bool SplatWorkerPool::restartWorker(size_t index)
{
    std::scoped_lock lock(m_mutex);
    if(index >= m_workers.size())
    {
        return false;
    }

    SplatWorkerProcess& w = m_workers[index];
    const int old_pid = w.pid;

    if(w.socket_fd >= 0)
    {
        close(w.socket_fd);
        w.socket_fd = -1;
    }
    if(w.pid > 0)
    {
        ::kill(w.pid, SIGTERM);
        int status = 0;
        for(int attempt = 0; attempt < 20; ++attempt)
        {
            if(waitpid(w.pid, &status, WNOHANG) != 0)
            {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    w.busy = false;
    w.current_job_id = 0;

    SplatWorkerProcess replacement{};
    if(!spawnWorker(w.id, replacement))
    {
        w.pid = -1;
        return false;
    }
    m_workers[index] = std::move(replacement);
    LOGI("SplatWorkerPool: restarted worker %d (old pid=%d, new pid=%d)\n",
         m_workers[index].id, old_pid, static_cast<int>(m_workers[index].pid));
    return true;
}

int SplatWorkerPool::activeWorkers() const
{
    std::scoped_lock lock(m_mutex);
    int active = 0;
    for(const SplatWorkerProcess& w : m_workers)
    {
        if(w.socket_fd >= 0)
        {
            ++active;
        }
    }
    return active;
}

size_t SplatWorkerPool::workerCount() const
{
    std::scoped_lock lock(m_mutex);
    return m_workers.size();
}

pid_t SplatWorkerPool::workerPid(size_t index) const
{
    std::scoped_lock lock(m_mutex);
    if(index >= m_workers.size())
    {
        return -1;
    }
    return m_workers[index].pid;
}

int SplatWorkerPool::queueDepth() const
{
    std::scoped_lock lock(m_mutex);
    return static_cast<int>(m_pendingQueue.size());
}

bool SplatWorkerPool::isJobComplete(uint32_t job_id) const
{
    std::scoped_lock lock(m_mutex);
    return m_results.find(job_id) != m_results.end();
}

void SplatWorkerPool::shutdown()
{
    std::scoped_lock lock(m_mutex);
    for(SplatWorkerProcess& w : m_workers)
    {
        if(w.pid > 0)
        {
            ::kill(w.pid, SIGTERM);
        }
    }

    for(SplatWorkerProcess& w : m_workers)
    {
        if(w.pid > 0)
        {
            int status = 0;
            for(int attempt = 0; attempt < 20; ++attempt)
            {
                if(waitpid(w.pid, &status, WNOHANG) != 0)
                {
                    w.pid = -1;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            if(w.pid > 0)
            {
                ::kill(w.pid, SIGKILL);
                waitpid(w.pid, &status, 0);
                w.pid = -1;
            }
        }
        if(w.socket_fd >= 0)
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