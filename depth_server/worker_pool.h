#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include <sys/types.h>

struct WorkerProcess {
    int id;
    pid_t pid;
    std::string device;         // "cpu", "cuda:0", "cuda:1"
    int socket_fd;              // socketpair fd for IPC
    bool busy;
    uint32_t current_frame;     // frame_index being processed
    int pending_depth_fd;       // -1 if not waiting, else fd
};

struct PendingFrame {
    uint32_t frame_index;
    uint32_t timestamp_ms;
    std::vector<uint8_t> rgb;
    uint32_t width;
    uint32_t height;
};

class WorkerPool {
public:
    WorkerPool() = default;
    ~WorkerPool();

    bool initialize(const std::string& model_path, int num_workers, const std::string& backend);
    int submitFrame(uint32_t frame_index, uint32_t timestamp_ms,
                    const uint8_t* rgb_data, uint32_t width, uint32_t height);
    bool pollResult(int& frame_index, std::vector<float>& depth_data,
                    int& out_w, int& out_h, float& scale, float& bias, float& z_max);
    void shutdown();
    int activeWorkers() const;
    int queueDepth() const;
    float avgProcessingMs() const;
    std::vector<int> getWorkerLoads() const;

private:
    std::vector<WorkerProcess> m_workers;
    std::deque<PendingFrame> m_pendingQueue;
    size_t m_nextWorker = 0;
    std::string m_modelPath;
    int m_numWorkers = 0;
    std::vector<std::int64_t> m_dispatchTimesNs;
    double m_totalProcessingMs = 0.0;
    std::uint64_t m_completedFrames = 0;

    int spawnWorker(int id, const std::string& device);
    int dispatchFrame(const PendingFrame& frame);
    bool readResult(int worker_idx, int& frame_index, std::vector<float>& depth_data,
                    int& out_w, int& out_h, float& scale, float& bias, float& z_max);
};
