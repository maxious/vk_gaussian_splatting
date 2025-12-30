#include "depth_stream_client.h"
#include <nvutils/logger.hpp>
#include <fstream>
#include <sstream>
#include <tinygltf/json.hpp>

#include <ixwebsocket/IXNetSystem.h>

namespace vk_gaussian_splatting {

#define BACKEND_HOST "127.0.0.1"
#define BACKEND_PORT 8000

DepthStreamClient::DepthStreamClient() {
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

bool DepthStreamClient::connectWebSocket(const std::string& sessionId) {
    if (m_connected.load()) {
        disconnectWebSocket();
    }

    try {
        std::string uri = "ws://" + std::string(BACKEND_HOST) + ":" + std::to_string(BACKEND_PORT) + "/api/sessions/" + sessionId + "/stream";
        
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
        LOGI("Depth WebSocket connected\n");
        if (m_statusCallback) m_statusCallback(true);
    }
    else if (msg->type == ix::WebSocketMessageType::Close) {
        m_connected.store(false);
        LOGI("Depth WebSocket closed\n");
        if (m_statusCallback) m_statusCallback(false);
    }
    else if (msg->type == ix::WebSocketMessageType::Error) {
        LOGE("Depth WebSocket error: %s\n", msg->errorInfo.reason.c_str());
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
        return false;
    }

    try {
        nlohmann::json request;
        request["time_ms"] = timestampMs;

        std::string msg = request.dump();
        m_webSocket.send(msg);

        return true;
    } catch (const std::exception& e) {
        LOGE("Failed to send depth request: %s\n", e.what());
        return false;
    }
}

bool DepthStreamClient::getFrame(uint64_t targetMs, DepthFrame& outFrame) {
    return m_depthBuffer.getFrame(targetMs, outFrame);
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
    
    std::string url = "http://" + std::string(BACKEND_HOST) + ":" + std::to_string(BACKEND_PORT) + endpoint;
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
    
    std::string url = "http://" + std::string(BACKEND_HOST) + ":" + std::to_string(BACKEND_PORT) + endpoint;
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

bool DepthStreamClient::sendHttpDelete(const std::string& endpoint) {
    ix::HttpClient httpClient;
    auto args = std::make_shared<ix::HttpRequestArgs>();
    
    std::string url = "http://" + std::string(BACKEND_HOST) + ":" + std::to_string(BACKEND_PORT) + endpoint;
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

} // namespace vk_gaussian_splatting
