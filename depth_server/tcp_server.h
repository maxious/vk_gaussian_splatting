#pragma once

#include "cloud_worker_pool.h"
#include "protocol_parser.h"
#include "splat_worker_pool.h"
#include "worker_pool.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <poll.h>
#include <string>
#include <unordered_map>
#include <vector>

struct ClientConnection {
    int fd = -1;
    TcpProtocolParser parser;
    uint64_t frames_received = 0;
    uint64_t frames_processed = 0;
    uint64_t bytes_rx = 0;
    uint64_t bytes_tx = 0;
    int64_t last_activity_ms = 0;   // epoch ms of last recv/send
};

class TcpServer {
public:
    TcpServer(WorkerPool& pool, SplatWorkerPool* splatPool = nullptr);
    ~TcpServer();

    bool start(int port);
    void stop();
    bool isRunning() const;

    // Called each frame from main loop — poll clients, dispatch frames, send responses
    void update();

    int activeConnections() const;
    uint64_t totalFramesProcessed() const;

    // Allow late-binding the splat pool (e.g. when constructed in depth-only mode)
    void setSplatPool(SplatWorkerPool* pool) { m_splatPool = pool; }

    // Allow late-binding the cloud pool
    void setCloudPool(CloudWorkerPool* pool) { m_cloudPool = pool; }

private:
    struct InFlightFrame {
        int client_slot = -1;
        uint64_t client_generation = 0;
        uint32_t timestamp_ms = 0;
    };

    WorkerPool& m_pool;
    SplatWorkerPool* m_splatPool = nullptr;
    CloudWorkerPool* m_cloudPool = nullptr;
    int m_listenFd = -1;
    std::atomic<bool> m_running{false};
    std::vector<ClientConnection> m_clients;
    int m_port = 9000;
    int64_t m_lastCleanupMs = 0;
    uint64_t m_totalFramesProcessed = 0;
    uint32_t m_nextFrameIndex = 1;
    std::vector<uint64_t> m_clientGenerations;
    std::unordered_map<uint32_t, InFlightFrame> m_inFlightFrames;
    std::map<uint32_t, InFlightFrame> m_inFlightSplatJobs;
    std::map<uint32_t, InFlightFrame> m_inFlightCloudJobs;
    std::unordered_map<uint32_t, CloudJobResult> m_pendingCloudResults;

    // Constants
    static constexpr int MAX_CLIENTS = 16;
    static constexpr int IDLE_TIMEOUT_MS = 60000;  // 60 seconds
    static constexpr int CLEANUP_INTERVAL_MS = 5000;  // check every 5s
};
