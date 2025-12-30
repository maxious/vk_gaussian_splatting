#include "depth_buffer.h"
#include "depth_stream_client.h"
#include <nvutils/logger.hpp>
#include <algorithm>

void DepthBuffer::addFrame(const DepthFrame& frame) {
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

bool DepthBuffer::getFrame(uint64_t targetMs, DepthFrame& outFrame) {
    auto cacheIt = m_receivedFrames.find(targetMs);
    if (cacheIt != m_receivedFrames.end()) {
        outFrame = cacheIt->second;
        return true;
    }

    if (!m_client) return false;

    auto it = std::find_if(m_pendingFrames.begin(), m_pendingFrames.end(),
                          [&](const PendingFrame& pf) {
                              return pf.timestampMs == targetMs;
                          });

    if (it != m_pendingFrames.end()) {
        return false;
    }

    if (m_pendingFrames.size() >= m_maxPending) {
        m_pendingFrames.pop_front();
    }

    PendingFrame pf;
    pf.timestampMs = targetMs;
    pf.sentTime = getCurrentTimeMs();
    m_pendingFrames.push_back(pf);

    m_client->requestDepth(targetMs);
    
    return false;
}
