#include "depth_stream_client.h"
#include <nvutils/logger.hpp>
#include <fstream>
#include <sstream>
#include <tinygltf/json.hpp>

#include <ixwebsocket/IXNetSystem.h>

// Workaround for potential mutex issues
// See: https://developercommunity.visualstudio.com/t/5e013719/8n9t9qf/
#define _HAS_EXCEPTIONS 0
#include <mutex>

namespace vk_viewer {



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

bool DepthStreamClient::uploadImage(const std::filesystem::path& imagePath, std::vector<uint8_t>& outPlyData) {
    std::string response;
    // Determine mime type based on extension
    std::string ext = imagePath.extension().string();
    std::string contentType = "image/jpeg";
    if (ext == ".png") contentType = "image/png";
    
    if (!sendHttpPostMultipart("/api/image/upload", imagePath, response, contentType)) {
        return false;
    }
    
    // The response body IS the PLY data (binary)
    outPlyData.assign(response.begin(), response.end());
    return true;
}

bool DepthStreamClient::processImagePath(const std::filesystem::path& imagePath, std::vector<uint8_t>& outPlyData) {
    ix::HttpClient httpClient;
    auto args = std::make_shared<ix::HttpRequestArgs>();
    
    nlohmann::json jsonBody;
    jsonBody["path"] = imagePath.string();
    std::string body = jsonBody.dump();
    
    args->extraHeaders["Content-Type"] = "application/json";
    
    std::string url = "http://" + m_backendHost + ":" + std::to_string(m_backendPort) + "/api/image/path";
    auto res = httpClient.post(url, body, args);
    
    if (res->errorCode != ix::HttpErrorCode::Ok) {
        LOGE("HTTP error: %s\n", res->errorMsg.c_str());
        return false;
    }
    
    if (res->statusCode != 200) {
        LOGE("HTTP error %d processing image path\n", res->statusCode);
        return false;
    }
    
    outPlyData.assign(res->body.begin(), res->body.end());
    return true;
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
        LOGD("Already connected, disconnecting first...\n");
        disconnectWebSocket();
    }

    try {
        std::string uri = "ws://" + m_backendHost + ":" + std::to_string(m_backendPort) + "/api/sessions/" + sessionId + "/stream";

        LOGI(">>> DEPTH_CLIENT: Connecting WebSocket to: %s\n", uri.c_str());
        m_webSocket.setUrl(uri);
        m_webSocket.start();
        LOGI(">>> DEPTH_CLIENT: WebSocket connection initiated\n");
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
        LOGI(">>> DEPTH_CLIENT: WebSocket OPEN - connected successfully!\n");
        if (m_statusCallback) m_statusCallback(true);
    }
    else if (msg->type == ix::WebSocketMessageType::Close) {
        m_connected.store(false);
        LOGI(">>> DEPTH_CLIENT: WebSocket CLOSED (code: %d, reason: %s)\n",
               msg->closeInfo.code, msg->closeInfo.reason.c_str());
        if (m_statusCallback) m_statusCallback(false);
    }
    else if (msg->type == ix::WebSocketMessageType::Error) {
        LOGE(">>> DEPTH_CLIENT: WebSocket ERROR: %s (retries: %d)\n",
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
                    static int frameCount = 0;
                    frameCount++;
                    m_stats.totalFrames = frameCount;

                    // Log every frame for debugging (remove after verified)
                    LOGI(">>> DEPTH_CLIENT: Frame #%d received: %dx%d @ %u ms (scale=%.4f, bias=%.4f)\n",
                         frameCount, frame.width, frame.height, frame.timestampMs, frame.scale, frame.bias);

                    m_depthBuffer.addFrame(frame);
                    if (m_depthCallback) m_depthCallback(frame);
                } else {
                    LOGE(">>> DEPTH_CLIENT: Failed to parse depth frame!\n");
                }
            } else {
                // Text message - likely JSON error or status
                auto json = nlohmann::json::parse(msg->str);
                if (json.contains("type") && json["type"] == "error") {
                    LOGE("Depth stream error: %s\n", json["message"].get<std::string>().c_str());
                } else {
                    LOGI(">>> DEPTH_CLIENT: Text message: %s\n", msg->str.c_str());
                }
            }
        } catch (const std::exception& e) {
            LOGE(">>> DEPTH_CLIENT: Exception parsing frame: %s\n", e.what());
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
    float jitter = m_depthBuffer.getJitter();

    // Calculate lead time - start at 0 for initial fill, then add lookahead
    double startMs;
    if (m_stats.totalFrames < 10) {
        // Initial fill phase: request from beginning
        startMs = 0.0;
    } else {
        // Playback phase: request ahead of current time
        // Use smaller lead time once we have initial buffer
        double minLeadMs = std::min(1500.0, std::max(100.0, static_cast<double>(rtt + jitter + 100.0)));
        startMs = currentTimeMs + minLeadMs;
    }

    uint64_t alignedStartMs = static_cast<uint64_t>(std::ceil(startMs / stepMs) * stepMs);
    uint64_t endMs = static_cast<uint64_t>(alignedStartMs + bufferWindowMs);

    // Iterate through the lookahead window and request missing frames
    for (uint64_t t = alignedStartMs; t < endMs; t += static_cast<uint64_t>(stepMs)) {
        m_depthBuffer.prefetch(t);
    }

    // Cleanup old frames (keep 2 seconds history)
    m_depthBuffer.cleanup(static_cast<uint64_t>(std::max(0.0, currentTimeMs - 2000.0)));

    // Periodically fetch backend telemetry (every 2 seconds)
    static double lastTelemetryFetch = 0.0;
    if (currentTimeMs - lastTelemetryFetch > 2000.0) {
        fetchSessionTelemetry();
        lastTelemetryFetch = currentTimeMs;
    }
}

bool DepthStreamClient::sendHttpPostMultipart(const std::string& endpoint, const std::filesystem::path& filePath, std::string& response, const std::string& contentType) {
    ix::HttpClient httpClient;
    auto args = std::make_shared<ix::HttpRequestArgs>();
    
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        LOGE("Failed to open file: %s\n", filePath.string().c_str());
        return false;
    }

    std::vector<uint8_t> fileData((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    // Create multipart boundary
    std::string boundary = "----WebKitFormBoundary" + std::to_string(std::rand());

    // Build multipart body
    std::ostringstream body_stream;
    body_stream << "--" << boundary << "\r\n";
    body_stream << "Content-Disposition: form-data; name=\"file\"; filename=\"" << filePath.filename().string() << "\"\r\n";
    body_stream << "Content-Type: " << contentType << "\r\n\r\n";
    body_stream.write(reinterpret_cast<const char*>(fileData.data()), fileData.size());
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
        LOGE("HTTP error %d uploading file\n", res->statusCode);
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

bool DepthStreamClient::fetchSessionTelemetry() {
    if (!m_connected.load()) {
        return false;
    }
    
    std::string endpoint = "/api/sessions/" + m_currentSession.sessionId + "/telemetry";
    std::string response;
    
    if (!sendHttpGet(endpoint, response)) {
        return false;
    }
    
    try {
        nlohmann::json telemetry = nlohmann::json::parse(response);
        
        // Parse telemetry data (matching web app structure)
        if (telemetry.contains("avg_infer_ms")) {
            m_stats.inferTimeMs = telemetry["avg_infer_ms"];
        }
        if (telemetry.contains("avg_decode_ms")) {
            m_stats.decodeTimeMs = telemetry["avg_decode_ms"];
        }
        if (telemetry.contains("avg_pack_ms")) {
            m_stats.packTimeMs = telemetry["avg_pack_ms"];
        }
        if (telemetry.contains("avg_queue_wait_ms")) {
            m_stats.queueWaitTimeMs = telemetry["avg_queue_wait_ms"];
        }
        
        LOGD("Fetched telemetry: infer=%.1fms, decode=%.1fms, pack=%.1fms, queue=%.1fms\n",
              m_stats.inferTimeMs, m_stats.decodeTimeMs, m_stats.packTimeMs, m_stats.queueWaitTimeMs);
              
        return true;
    } catch (const std::exception& e) {
        LOGE("Failed to parse telemetry: %s\n", e.what());
        return false;
    }
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
    stats.jitterMs = m_depthBuffer.getJitter();
    stats.pendingRequests = m_depthBuffer.getPendingCount();
    stats.fps = 0.0f;
    stats.totalFrames = m_stats.totalFrames;
    stats.droppedFrames = m_stats.droppedFrames;

    // Include telemetry data if available
    stats.inferTimeMs = m_stats.inferTimeMs;
    stats.decodeTimeMs = m_stats.decodeTimeMs;
    stats.packTimeMs = m_stats.packTimeMs;
    stats.queueWaitTimeMs = m_stats.queueWaitTimeMs;

    LOGD(">>> DEPTH_CLIENT: getStats: totalFrames=%zu, rtt=%.1f, jitter=%.1f\n",
         stats.totalFrames, stats.rttMs, stats.jitterMs);

    return stats;
}

bool DepthStreamClient::createHlsSession(const std::filesystem::path& videoPath, SessionInfo& outSession) {
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

        LOGI("HLS session created: %s (%dx%d @ %.1f FPS)\n",
             outSession.sessionId.c_str(), outSession.width, outSession.height, outSession.fps);
        return true;
    } catch (const std::exception& e) {
        LOGE("Failed to parse session response: %s\n", e.what());
        return false;
    }
}

bool DepthStreamClient::startHlsGeneration(const std::string& sessionId, float fps, int processRes) {
    ix::HttpClient httpClient;
    auto args = std::make_shared<ix::HttpRequestArgs>();

    nlohmann::json jsonBody;
    jsonBody["fps"] = fps;
    jsonBody["segment_duration"] = 2.0;
    jsonBody["process_res"] = processRes;
    std::string body = jsonBody.dump();

    args->extraHeaders["Content-Type"] = "application/json";

    std::string url = "http://" + m_backendHost + ":" + std::to_string(m_backendPort)
                      + "/api/sessions/" + sessionId + "/hls";
    auto res = httpClient.post(url, body, args);

    if (res->errorCode != ix::HttpErrorCode::Ok) {
        LOGE("HTTP error starting HLS: %s\n", res->errorMsg.c_str());
        return false;
    }

    if (res->statusCode != 200) {
        LOGE("HTTP error %d starting HLS generation\n", res->statusCode);
        return false;
    }

    LOGI("HLS generation started for session: %s\n", sessionId.c_str());
    return true;
}

bool DepthStreamClient::getHlsStatus(const std::string& sessionId, HlsGenerationStatus& outStatus) {
    std::string response;
    std::string endpoint = "/api/sessions/" + sessionId + "/hls/status";

    if (!sendHttpGet(endpoint, response)) {
        return false;
    }

    try {
        auto json = nlohmann::json::parse(response);

        outStatus.status = json.value("status", "unknown");
        outStatus.progress = json.value("progress", 0.0f);
        outStatus.frameCount = json.value("frame_count", 0);
        outStatus.totalFrames = json.value("total_frames", 0);
        outStatus.etaSeconds = json.value("eta_seconds", 0.0f);
        outStatus.framesPerSecond = json.value("frames_per_second", 0.0f);
        outStatus.errorMessage = json.value("error", "");

        if (outStatus.status == "ready") {
            outStatus.streamUrl = json.value("stream_url", "");
        }

        return true;
    } catch (const std::exception& e) {
        LOGE("Failed to parse HLS status: %s\n", e.what());
        return false;
    }
}

std::string DepthStreamClient::getHlsMetadataPath(const std::string& sessionId) {
    // Returns the local path to metadata.json based on data root
    // The backend stores HLS at {data_root}/{session_id}/hls/metadata.json
    std::filesystem::path dataRoot = std::filesystem::path("/home/maxious/vk_gaussian_splatting/python/tmp/sessions");
    return (dataRoot / sessionId / "hls" / "metadata.json").string();
}

} // namespace vk_viewer
