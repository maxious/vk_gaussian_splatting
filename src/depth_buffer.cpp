#include "depth_buffer.h"
#include "depth_stream_client.h"
#include <nvutils/logger.hpp>
#include <algorithm>
#include <mutex>

void DepthBuffer::addFrame(const DepthFrame& frame) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);

    m_receivedFrames[frame.timestampMs] = frame;

    double now = getCurrentTimeMs();
    auto it = std::find_if(m_pendingFrames.begin(), m_pendingFrames.end(),
                          [&](const PendingFrame& pf) {
                              return pf.timestampMs == frame.timestampMs;
                          });

    if (it != m_pendingFrames.end()) {
        float rtt = static_cast<float>(now - it->sentTime);

        const float alpha = 0.1f;
        m_rtt = m_rtt * (1.0f - alpha) + rtt * alpha;

        m_pendingFrames.erase(it);
    }
}

void DepthBuffer::prefetch(uint64_t targetMs) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);

    if (m_receivedFrames.find(targetMs) != m_receivedFrames.end()) {
        return;
    }

    if (!m_client) return;

    auto it = std::find_if(m_pendingFrames.begin(), m_pendingFrames.end(),
                          [&](const PendingFrame& pf) {
                              return pf.timestampMs == targetMs;
                          });

    if (it != m_pendingFrames.end()) {
        return;
    }

    if (m_pendingFrames.size() >= m_maxPending) {
        m_pendingFrames.pop_front();
    }

    PendingFrame pf;
    pf.timestampMs = targetMs;
    pf.sentTime = getCurrentTimeMs();
    m_pendingFrames.push_back(pf);

    m_client->requestDepth(targetMs);
}

void DepthBuffer::cleanup(uint64_t oldThresholdMs) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);

    for (auto it = m_receivedFrames.begin(); it != m_receivedFrames.end();) {
        if (it->first < oldThresholdMs) {
            it = m_receivedFrames.erase(it);
        } else {
            ++it;
        }
    }
}

bool DepthBuffer::getFrame(uint64_t targetMs, DepthFrame& outFrame) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);

    auto cacheIt = m_receivedFrames.find(targetMs);
    if (cacheIt != m_receivedFrames.end()) {
        outFrame = cacheIt->second;
        return true;
    }

    // Note: prefetch() will acquire its own lock, but that's OK since it's recursive
    prefetch(targetMs);

    return false;
}
