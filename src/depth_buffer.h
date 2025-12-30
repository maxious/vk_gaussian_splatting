#pragma once

#include "depth_parser.h"
#include <deque>
#include <map>
#include <chrono>

namespace vk_gaussian_splatting {
    class DepthStreamClient;
}

class DepthBuffer {
public:
    DepthBuffer(size_t maxPending = 60) : m_maxPending(maxPending) {}

    void setClient(vk_gaussian_splatting::DepthStreamClient* client) { m_client = client; }
    void addFrame(const DepthFrame& frame);

    // Get best frame for current timestamp
    bool getFrame(uint64_t targetMs, DepthFrame& outFrame);
    
    void ensureFrame(uint64_t targetMs);

    void cleanup(uint64_t oldThresholdMs);

    size_t getPendingCount() const { return m_pendingFrames.size(); }
    float getRTT() const { return m_rtt; }

private:
    struct PendingFrame {
        uint64_t timestampMs;
        double sentTime;
    };
    
    // Helper to get current time in ms
    double getCurrentTimeMs() const {
        using namespace std::chrono;
        return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
    }

    std::deque<PendingFrame> m_pendingFrames;
    std::map<uint64_t, DepthFrame> m_receivedFrames;  // Cache for re-requests
    size_t m_maxPending;

    vk_gaussian_splatting::DepthStreamClient* m_client{nullptr};

    uint64_t m_lastTimestamp{0};
    float m_rtt{0.0f};
};
