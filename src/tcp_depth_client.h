#pragma once

#ifdef WITH_TCP_DEPTH

#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "depth_buffer.h"

using DepthFrameCallback = std::function<void(const DepthFrame&)>;

class TcpDepthClient {
public:
    TcpDepthClient();
    ~TcpDepthClient();

    bool connect(const std::string& host, int port, int timeout_ms = 5000);
    void disconnect();

    void sendFrameRequest(uint32_t frame_index,
                          uint32_t timestamp_ms,
                          const uint8_t* rgb_data,
                          uint32_t width,
                          uint32_t height);

    void setDepthFrameCallback(DepthFrameCallback cb);
    void update();

    enum class State { DISCONNECTED, CONNECTING, CONNECTED, DISCONNECTING };

    State getState() const;
    bool isConnected() const;

private:
    void ioThreadFunc();

    std::string m_host;
    int m_port = 9000;
    int m_socketFd = -1;
    int m_epollFd = -1;
    int m_wakeupFd = -1;
    int m_connectTimeoutMs = 5000;
    int m_responseTimeoutMs = 5000;
    std::atomic<State> m_state{State::DISCONNECTED};

    std::thread m_ioThread;
    DepthFrameCallback m_callback;

    std::mutex m_mutex;
    std::deque<std::vector<uint8_t>> m_pendingRequests;
    std::string m_lastError;
    State m_lastLoggedState{State::DISCONNECTED};
    uint64_t m_lastIoActivityMs = 0;
    size_t m_inFlightRequests = 0;
};

#endif  // WITH_TCP_DEPTH
