#pragma once

#include "depth_parser.h"
#include <deque>
#include <map>
#include <chrono>
#include <mutex>

namespace vk_viewer {
    class DepthStreamClient;
}

class DepthBuffer {
public:
    DepthBuffer(size_t maxPending = 60) : m_maxPending(maxPending) {}

    void setClient(vk_viewer::DepthStreamClient* client) { m_client = client; }
    void addFrame(const DepthFrame& frame);

    // Get best frame for current timestamp. Returns true if frame is available.
    // If not cached, triggers a prefetch request and returns false.
    bool getFrame(uint64_t targetMs, DepthFrame& outFrame);
    
    // Request a frame to be fetched ahead of time (for lookahead buffering).
    // Does nothing if already cached or pending.
    void prefetch(uint64_t targetMs);

    void cleanup(uint64_t oldThresholdMs);

    size_t getPendingCount() const { return m_pendingFrames.size(); }
    size_t getCachedCount() const { return m_receivedFrames.size(); }
    bool hasCachedFrame(uint64_t timestampMs) const {
      return m_receivedFrames.find(timestampMs) != m_receivedFrames.end();
    }
    uint64_t getFirstCachedTimestamp() const {
      if(m_receivedFrames.empty()) return 0;
      return m_receivedFrames.begin()->first;
    }
    float getRTT() const { return m_rtt; }
    float getJitter() const { return m_jitter; }

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
    mutable std::recursive_mutex m_mutex;  // Recursive mutex for prefetch calling from getFrame

    vk_viewer::DepthStreamClient* m_client{nullptr};

    uint64_t m_lastTimestamp{0};
    float m_rtt{0.0f};
    
    // Jitter measurement and EMA filtering for better synchronization
    float m_jitter{0.0f};
    const float RTT_EMA_ALPHA = 0.1f;  // EMA smoothing factor
};
