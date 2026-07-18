#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <sys/types.h>

#include "protocol.h"

struct CloudWorkerProcess {
    int id;
    pid_t pid;
    int socket_fd;              // socketpair fd for IPC
    bool busy = false;
    uint32_t current_job_id = 0;
};

struct PendingCloudJob {
    uint32_t job_id;
    std::vector<std::string> frame_paths;
    CloudRequestOptions options;
};

struct CloudJobResult {
    uint32_t job_id;
    std::string output_path;    // path to .splat file
    uint32_t n_points = 0;
    std::string error;
};

class CloudWorkerPool {
public:
    /// Signature of the worker entry-point function (CloudWorker::run or mock).
    using CloudWorkerFunc = void (*)(int parent_fd, const std::string& model_path,
                                     const std::string& backend,
                                     const std::string& cloud_cache_dir,
                                     const std::string& work_dir);

    CloudWorkerPool() = default;
    ~CloudWorkerPool();

    /// Set a custom worker function (for testing). nullptr reverts to CloudWorker::run.
    void setWorkerFunc(CloudWorkerFunc func);

    bool initialize(const std::string& model_path, int num_workers, const std::string& backend);
    uint32_t submitJob(const std::vector<std::string>& frame_paths, const CloudRequestOptions& options);
    bool pollResult(CloudJobResult& out_result);
    bool cancelJob(uint32_t job_id);
    bool restartWorker(size_t index);
    int activeWorkers() const;
    size_t workerCount() const;
    pid_t workerPid(size_t index) const;
    int queueDepth() const;
    bool isJobComplete(uint32_t job_id) const;
    void shutdown();

private:
    std::vector<CloudWorkerProcess> m_workers;
    std::deque<PendingCloudJob> m_pendingQueue;
    std::map<uint32_t, CloudJobResult> m_results;  // completed jobs
    std::vector<std::pair<uint32_t, uint32_t>> m_inFlight;  // job_id -> worker index
    std::atomic<uint32_t> m_nextJobId{1};
    std::string m_modelPath;
    std::string m_backend;
    std::string m_cloudCacheDir;
    std::string m_workDir;
    int m_numWorkers = 0;
    mutable std::mutex m_mutex;
    CloudWorkerFunc m_workerFunc = nullptr;  // nullptr = use CloudWorker::run

    bool spawnWorker(int id, CloudWorkerProcess& out_worker);
    bool readFromWorker(int worker_idx, CloudJobResult& out);
    bool dispatchToWorker(int worker_idx, const PendingCloudJob& job);
    void respawnWorker(size_t index);
};
