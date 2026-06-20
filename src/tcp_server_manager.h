#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#ifdef WITH_TCP_DEPTH
#include "tcp_depth_client.h"

struct ServerConnection {
    ServerConnection() = default;
    ServerConnection(ServerConnection&& other) noexcept;
    ServerConnection& operator=(ServerConnection&& other) noexcept;
    ServerConnection(const ServerConnection&) = delete;
    ServerConnection& operator=(const ServerConnection&) = delete;

    std::string host;
    int port;
    TcpDepthClient client;
    uint64_t frames_sent = 0;
    uint64_t frames_completed = 0;
    uint64_t frames_failed = 0;
    double avg_latency_ms = 0.0;
    bool is_connected = false;
    int64_t last_connect_attempt = 0;
    int retry_count = 0;
    static constexpr int MAX_BACKOFF_MS = 30000;

    int getBackoffMs() const;
};

class TcpServerManager {
public:
    TcpServerManager();
    ~TcpServerManager();

    void addServer(const std::string& host, int port);
    void connectAll();
    void disconnectAll();
    void setDepthBuffer(DepthBuffer* buffer);
    void setFrameSkip(int video_fps, int server_fps_estimate);

    // Send frame via round-robin, re-queue on failure
    int sendFrame(uint32_t frame_index,
                   uint32_t timestamp_ms,
                  const uint8_t* rgb_data,
                  uint32_t width,
                  uint32_t height);

    // Set callback for depth frame responses
    void setDepthFrameCallback(DepthFrameCallback cb);

    // Called each frame — check connections, reconnect, process completions
    void update();

    // Stats
    size_t serverCount() const;
    const std::vector<ServerConnection>& getServers() const;

    // Parse server list from string "host:port,host:port"
    void parseServerList(const std::string& list);

private:
    struct PendingFrame {
        uint32_t frame_index;
        uint32_t timestamp_ms;
        std::vector<uint8_t> rgb;
        uint32_t width;
        uint32_t height;
        int retry_count = 0;
        static constexpr int MAX_RETRIES = 3;
    };

    struct InFlightFrame {
        PendingFrame frame;
        int64_t dispatch_time_ms = 0;
    };

    struct CompletedFrame {
        size_t server_idx = 0;
        DepthFrame frame;
    };

    int dispatchToServer(PendingFrame& frame, size_t server_idx);
    void reconnectFailedServers();
    void processCompletedFrames();
    void requeueDisconnectedFrames(size_t server_idx, int64_t now_ms);

    mutable std::mutex m_mutex;
    std::vector<ServerConnection> m_servers;
    std::vector<std::vector<InFlightFrame>> m_inFlightFrames;
    size_t m_nextServer = 0;
    DepthFrameCallback m_callback;
    DepthBuffer* m_depthBuffer = nullptr;
    int m_frameSkip = 1;
    std::vector<PendingFrame> m_pendingFrames;
    std::vector<CompletedFrame> m_completedFrames;
};
#endif
