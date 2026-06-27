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

struct SplatWorkerProcess {
    int id;
    pid_t pid;
    int socket_fd;              // socketpair fd for IPC
    bool busy = false;
    uint32_t current_job_id = 0;
};

struct PendingSplatJob {
    uint32_t job_id;
    std::vector<std::vector<uint8_t>> images;  // JPEG/PNG bytes per view
    SplatRequestOptions options;
};

struct SplatJobResult {
    uint32_t job_id;
    std::string output_path;    // path to .splat file
    uint32_t n_gaussians = 0;
    std::string error;
};

class SplatWorkerPool {
public:
    SplatWorkerPool() = default;
    ~SplatWorkerPool();

    bool initialize(const std::string& model_path, int num_workers, const std::string& backend);
    uint32_t submitJob(const std::vector<std::vector<uint8_t>>& images, const SplatRequestOptions& options);
    bool pollResult(SplatJobResult& out_result);
    bool cancelJob(uint32_t job_id);
    bool restartWorker(size_t index);
    int activeWorkers() const;
    size_t workerCount() const;
    pid_t workerPid(size_t index) const;
    int queueDepth() const;
    bool isJobComplete(uint32_t job_id) const;
    void shutdown();

private:
    std::vector<SplatWorkerProcess> m_workers;
    std::deque<PendingSplatJob> m_pendingQueue;
    std::map<uint32_t, SplatJobResult> m_results;  // completed jobs
    std::vector<std::pair<uint32_t, uint32_t>> m_inFlight;  // job_id -> worker index
    std::atomic<uint32_t> m_nextJobId{1};
    std::string m_modelPath;
    std::string m_backend;
    std::string m_splatCacheDir;
    int m_numWorkers = 0;
    mutable std::mutex m_mutex;

    bool spawnWorker(int id, SplatWorkerProcess& out_worker);
    bool readFromWorker(int worker_idx, SplatJobResult& out);
    bool dispatchToWorker(int worker_idx, const PendingSplatJob& job);
};