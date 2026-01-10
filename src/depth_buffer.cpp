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

        // Calculate jitter as deviation from current EMA
        float jitter = std::abs(rtt - m_rtt);
        
        // Apply EMA filtering to both RTT and jitter
        const float EMA_ALPHA = 0.1f;
        m_rtt = m_rtt * (1.0f - EMA_ALPHA) + rtt * EMA_ALPHA;
        m_jitter = m_jitter * (1.0f - EMA_ALPHA) + jitter * EMA_ALPHA;

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

    // Try exact match first
    auto cacheIt = m_receivedFrames.find(targetMs);
    if (cacheIt != m_receivedFrames.end()) {
        outFrame = cacheIt->second;
        return true;
    }

    // No exact match - find the closest frame (nearest neighbor interpolation)
    // This handles cases where the exact timestamp doesn't exist
    if (!m_receivedFrames.empty()) {
        // Find the frame with the closest timestamp
        auto closestIt = m_receivedFrames.begin();
        int64_t minDiff = INT64_MAX;

        for (auto it = m_receivedFrames.begin(); it != m_receivedFrames.end(); ++it) {
            int64_t diff = static_cast<int64_t>(it->first) - static_cast<int64_t>(targetMs);
            int64_t absDiff = std::abs(diff);
            if (absDiff < minDiff) {
                minDiff = absDiff;
                closestIt = it;
            }

            // Early exit if we found an exact match
            if (absDiff == 0) break;
        }

        // Only use the closest frame if it's within a reasonable threshold (100ms)
        if (minDiff < 100) {
            LOGD(">>> BUFFER: Using nearest frame: requested=%llu, found=%llu, diff=%lld ms\n",
                 targetMs, closestIt->first, minDiff);
            outFrame = closestIt->second;
            return true;
        }
    }

    // Note: prefetch() will acquire its own lock, but that's OK since it's recursive
    prefetch(targetMs);

    return false;
}
