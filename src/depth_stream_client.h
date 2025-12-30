#pragma once

#ifdef WITH_DEPTH_STREAMING

#include <string>
#include <functional>
#include <thread>
#include <mutex>
#include <atomic>
#include <filesystem>
#include <queue>

#include <ixwebsocket/IXWebSocket.h>
#include <ixwebsocket/IXHttpClient.h>

#include <tinygltf/json.hpp>
#include "depth_parser.h"
#include "depth_buffer.h"

namespace vk_gaussian_splatting {

class DepthStreamClient {
public:
    struct SessionInfo {
        std::string sessionId;
        uint32_t width;
        uint32_t height;
        float fps;
        uint64_t durationMs;
    };

    struct SessionStatus {
        SessionInfo info;
        uint32_t bufferLength;
        uint64_t lastDepthTimeMs;
        std::map<std::string, float> telemetry;
        struct RollingStats {
            float depth_fps;
            float latency_ms;
            float infer_avg_s;
            float queue_avg_s;
            float ws_send_avg_s;
            float drop_count;
        } rollingStats;
        struct Config {
            int inference_workers;
            int process_res;
            int downsample_factor;
        } config;
    };

    using DepthFrameCallback = std::function<void(const DepthFrame&)>;
    using StatusCallback = std::function<void(bool)>;

    DepthStreamClient(const std::string& backendHost = "192.168.1.200", int backendPort = 8000);
    ~DepthStreamClient();

    void setBackendAddress(const std::string& host, int port) { m_backendHost = host; m_backendPort = port; }

    bool testConnection();
    bool uploadVideo(const std::filesystem::path& videoPath, SessionInfo& outSession);
    bool getSessionStatus(const std::string& sessionId, SessionStatus& outStatus);
    bool deleteSession(const std::string& sessionId);

    bool connectWebSocket(const std::string& sessionId);
    void disconnectWebSocket();
    void close();
    bool requestDepth(uint64_t timestampMs);
    bool getFrame(uint64_t targetMs, DepthFrame& outFrame);
    
    void update(double currentTimeMs, float videoFps);

    void setDepthCallback(DepthFrameCallback callback) { m_depthCallback = callback; }
    void setStatusCallback(StatusCallback callback) { m_statusCallback = callback; }

    struct ClientStats {
        float rttMs;
        size_t pendingRequests;
        float fps;
        uint64_t totalFrames;
        uint64_t droppedFrames;
    };
    
    ClientStats getStats() const;
    
    float getFps() const { return m_currentSession.fps; }

private:
    void onDepthFrame(const ix::WebSocketMessagePtr& msg);

    bool sendHttpPostMultipart(const std::string& endpoint, const std::filesystem::path& filePath, std::string& response);
    bool sendHttpGet(const std::string& endpoint, std::string& response);
    bool sendHttpDelete(const std::string& endpoint);

    std::string             m_backendHost;
    int                     m_backendPort;
    SessionInfo             m_currentSession;

    ix::WebSocket           m_webSocket;
    std::atomic<bool>       m_shouldRun{false};
    std::atomic<bool>       m_connected{false};

    DepthBuffer             m_depthBuffer;
    DepthFrameCallback      m_depthCallback;
    StatusCallback          m_statusCallback;
};

} // namespace vk_gaussian_splatting

#endif // WITH_DEPTH_STREAMING
