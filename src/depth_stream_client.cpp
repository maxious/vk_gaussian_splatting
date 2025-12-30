#include "depth_stream_client.h"
#include <nvutils/logger.hpp>
#include <fstream>
#include <sstream>
#include <tinygltf/json.hpp>

#include <ixwebsocket/IXNetSystem.h>

namespace vk_gaussian_splatting {



DepthStreamClient::DepthStreamClient(const std::string& backendHost, int backendPort)
    : m_backendHost(backendHost), m_backendPort(backendPort)
{
    ix::initNetSystem();
    m_depthBuffer.setClient(this);

    // IXWebSocket configuration
    m_webSocket.setOnMessageCallback([this](const ix::WebSocketMessagePtr& msg) {
        onDepthFrame(msg);
    });
}

DepthStreamClient::~DepthStreamClient() {
    disconnectWebSocket();
}

bool DepthStreamClient::uploadVideo(const std::filesystem::path& videoPath, SessionInfo& outSession) {
    std::string response;
    if (!sendHttpPostMultipart("/api/sessions", videoPath, response)) {
        return false;
    }

    try {
        auto json = nlohmann::json::parse(response);
        outSession.sessionId = json["session_id"].get<std::string>();
        outSession.width = json["width"].get<uint32_t>();
        outSession.height = json["height"].get<uint32_t>();
        outSession.fps = json["fps"].get<float>();
        outSession.durationMs = json.value("duration_ms", static_cast<uint64_t>(0));

        m_currentSession = outSession;

        LOGI("Session created: %s (%dx%d @ %.1f FPS)\n",
               outSession.sessionId.c_str(), outSession.width, outSession.height, outSession.fps);
        return true;
    } catch (const std::exception& e) {
        LOGE("Failed to parse session response: %s\n", e.what());
        return false;
    }
}

bool DepthStreamClient::getSessionStatus(const std::string& sessionId, SessionStatus& outStatus) {
    std::string response;
    if (!sendHttpGet("/api/sessions/" + sessionId, response)) {
        return false;
    }

    try {
        auto json = nlohmann::json::parse(response);
        return true;
    } catch (...) {
        return false;
    }
}

bool DepthStreamClient::deleteSession(const std::string& sessionId) {
    return sendHttpDelete("/api/sessions/" + sessionId);
}

void DepthStreamClient::close() {
    disconnectWebSocket();
    if (!m_currentSession.sessionId.empty()) {
        deleteSession(m_currentSession.sessionId);
        m_currentSession = SessionInfo{};
    }
}

bool DepthStreamClient::connectWebSocket(const std::string& sessionId) {
    if (m_connected.load()) {
        disconnectWebSocket();
    }

    try {
        std::string uri = "ws://" + m_backendHost + ":" + std::to_string(m_backendPort) + "/api/sessions/" + sessionId + "/stream";
        
        LOGI("Connecting depth WebSocket: %s\n", uri.c_str());
        m_webSocket.setUrl(uri);
        m_webSocket.start();
        return true;
    } catch (const std::exception& e) {
        LOGE("WebSocket connect exception: %s\n", e.what());
        return false;
    }
}

void DepthStreamClient::disconnectWebSocket() {
    m_shouldRun.store(false);

    if (m_connected.load()) {
        m_webSocket.stop();
    }

    m_connected.store(false);
}

void DepthStreamClient::onDepthFrame(const ix::WebSocketMessagePtr& msg) {
    if (msg->type == ix::WebSocketMessageType::Open) {
        m_connected.store(true);
        LOGI("Depth WebSocket connected successfully\n");
        if (m_statusCallback) m_statusCallback(true);
    }
    else if (msg->type == ix::WebSocketMessageType::Close) {
        m_connected.store(false);
        LOGI("Depth WebSocket closed (code: %d, reason: %s)\n", 
               msg->closeInfo.code, msg->closeInfo.reason.c_str());
        if (m_statusCallback) m_statusCallback(false);
    }
    else if (msg->type == ix::WebSocketMessageType::Error) {
        LOGE("Depth WebSocket error: %s (retries: %d)\n", 
               msg->errorInfo.reason.c_str(), msg->errorInfo.retries);
        if (m_statusCallback) m_statusCallback(false);
    }
    else if (msg->type == ix::WebSocketMessageType::Message) {
        try {
            if (msg->binary) {
                // Binary message - depth frame
                // Create vector from string
                std::vector<uint8_t> buffer(msg->str.begin(), msg->str.end());
                DepthFrame frame;

                if (parseDepthFrame(buffer, frame)) {
                    LOGD("Received depth frame: %dx%d, timestamp=%llu, scale=%.4f, bias=%.4f\n",
                           frame.width, frame.height, frame.timestampMs, frame.scale, frame.bias);
                    
                    m_depthBuffer.addFrame(frame);
                    if (m_depthCallback) m_depthCallback(frame);
                }
            } else {
                // Text message - likely JSON error or status
                auto json = nlohmann::json::parse(msg->str);
                if (json.contains("type") && json["type"] == "error") {
                    LOGE("Depth stream error: %s\n", json["message"].get<std::string>().c_str());
                }
            }
        } catch (const std::exception& e) {
            LOGE("Error parsing depth frame: %s\n", e.what());
        }
    }
}

bool DepthStreamClient::requestDepth(uint64_t timestampMs) {
    if (!m_connected.load()) {
        LOGD("Cannot request depth - WebSocket not connected\n");
        return false;
    }

    try {
        nlohmann::json request;
        request["time_ms"] = timestampMs;

        std::string msg = request.dump();
        m_webSocket.send(msg);
        
        LOGD("Sent depth request for timestamp: %llu\n", timestampMs);
        return true;
    } catch (const std::exception& e) {
        LOGE("Failed to send depth request: %s\n", e.what());
        return false;
    }
}

bool DepthStreamClient::getFrame(uint64_t targetMs, DepthFrame& outFrame) {
    return m_depthBuffer.getFrame(targetMs, outFrame);
}

void DepthStreamClient::update(double currentTimeMs, float videoFps) {
    if (!m_connected.load()) return;

    const float fps = videoFps > 0.0f ? videoFps : 30.0f;
    const double stepMs = 1000.0 / fps;
    const double bufferWindowMs = 3000.0;

    float rtt = m_depthBuffer.getRTT();
    double minLeadMs = std::min(3000.0, std::max(100.0, static_cast<double>(rtt) + 100.0));

    double startMs = std::max(0.0, currentTimeMs + minLeadMs);
    uint64_t alignedStartMs = static_cast<uint64_t>(std::ceil(startMs / stepMs) * stepMs);
    uint64_t endMs = static_cast<uint64_t>(alignedStartMs + bufferWindowMs);

    // Iterate through the lookahead window and request missing frames
    for (uint64_t t = alignedStartMs; t < endMs; t += static_cast<uint64_t>(stepMs)) {
        m_depthBuffer.ensureFrame(t);
    }

    // Cleanup old frames (keep 2 seconds history)
    m_depthBuffer.cleanup(static_cast<uint64_t>(std::max(0.0, currentTimeMs - 2000.0)));
}

bool DepthStreamClient::sendHttpPostMultipart(const std::string& endpoint, const std::filesystem::path& videoPath, std::string& response) {
    ix::HttpClient httpClient;
    auto args = std::make_shared<ix::HttpRequestArgs>();
    
    std::ifstream file(videoPath, std::ios::binary);
    if (!file.is_open()) {
        LOGE("Failed to open video file: %s\n", videoPath.string().c_str());
        return false;
    }

    std::vector<uint8_t> videoData((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    // Create multipart boundary
    std::string boundary = "----WebKitFormBoundary" + std::to_string(std::rand());

    // Build multipart body
    std::ostringstream body_stream;
    body_stream << "--" << boundary << "\r\n";
    body_stream << "Content-Disposition: form-data; name=\"file\"; filename=\"" << videoPath.filename().string() << "\"\r\n";
    body_stream << "Content-Type: video/mp4\r\n\r\n";
    body_stream.write(reinterpret_cast<const char*>(videoData.data()), videoData.size());
    body_stream << "\r\n--" << boundary << "--\r\n";

    std::string body = body_stream.str();
    args->extraHeaders["Content-Type"] = "multipart/form-data; boundary=" + boundary;
    
    std::string url = "http://" + m_backendHost + ":" + std::to_string(m_backendPort) + endpoint;
    auto res = httpClient.post(url, body, args);
    
    if (res->errorCode != ix::HttpErrorCode::Ok) {
        LOGE("HTTP error: %s\n", res->errorMsg.c_str());
        return false;
    }
    
    if (res->statusCode != 200) {
        LOGE("HTTP error %d uploading video\n", res->statusCode);
        return false;
    }
    
    response = res->body;
    return true;
}

bool DepthStreamClient::sendHttpGet(const std::string& endpoint, std::string& response)
{
    ix::HttpClient httpClient;
    auto args = std::make_shared<ix::HttpRequestArgs>();
    
    std::string url = "http://" + m_backendHost + ":" + std::to_string(m_backendPort) + endpoint;
    auto res = httpClient.get(url, args);
    
    if (res->errorCode != ix::HttpErrorCode::Ok) {
        LOGE("HTTP error: %s\n", res->errorMsg.c_str());
        return false;
    }
    
    if (res->statusCode != 200) {
        LOGE("HTTP error %d getting session status\n", res->statusCode);
        return false;
    }
    
    response = res->body;
    return true;
}

bool DepthStreamClient::testConnection() {
    ix::HttpClient httpClient;
    auto args = std::make_shared<ix::HttpRequestArgs>();

    // Try to access the sessions endpoint to test connectivity
    std::string url = "http://" + m_backendHost + ":" + std::to_string(m_backendPort) + "/api/sessions";
    auto res = httpClient.get(url, args);

    if (res->errorCode != ix::HttpErrorCode::Ok) {
        LOGE("Backend connection test failed: %s\n", res->errorMsg.c_str());
        return false;
    }

    // Any response (even 405 Method Not Allowed) indicates the backend is reachable
    LOGI("Backend connection test successful (status: %d)\n", res->statusCode);
    return true;
}

bool DepthStreamClient::sendHttpDelete(const std::string& endpoint) {
    ix::HttpClient httpClient;
    auto args = std::make_shared<ix::HttpRequestArgs>();

    std::string url = "http://" + m_backendHost + ":" + std::to_string(m_backendPort) + endpoint;
    auto res = httpClient.Delete(url, args);

    if (res->errorCode != ix::HttpErrorCode::Ok) {
        LOGE("HTTP error: %s\n", res->errorMsg.c_str());
        return false;
    }

    if (res->statusCode != 200 && res->statusCode != 204) {
        LOGE("HTTP error %d deleting session\n", res->statusCode);
        return false;
    }

    return true;
}

DepthStreamClient::ClientStats DepthStreamClient::getStats() const {
    ClientStats stats;
    stats.rttMs = m_depthBuffer.getRTT();
    stats.pendingRequests = m_depthBuffer.getPendingCount();
    // Simple FPS calculation could be added here or in DepthBuffer
    stats.fps = 0.0f; 
    stats.totalFrames = 0; 
    stats.droppedFrames = 0; 
    return stats;
}

} // namespace vk_gaussian_splatting
