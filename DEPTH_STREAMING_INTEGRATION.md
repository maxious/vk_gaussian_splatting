# VideoDepthViewer3D C++ Integration Guide for vk_gaussian_splatting

This document provides suggestions for integrating VideoDepthViewer3D's depth streaming API into vk_gaussian_splatting as a C++ client.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [File Structure](#file-structure)
3. [HTTP Client Implementation](#http-client-implementation)
4. [WebSocket Client Implementation](#websocket-client-implementation)
5. [Vulkan Integration](#vulkan-integration)
6. [Flow Control & Buffering](#flow-control--buffering)
7. [Performance Considerations](#performance-considerations)
8. [Code Examples](#code-examples)

---

## Architecture Overview

### System Flow

```
┌─────────────────┐         ┌──────────────────┐         ┌──────────────────┐
│ vk_gaussian   │         │ VideoDepthViewer │         │  Vulkan        │
│   Splatting   │◄──────►│   3D Backend    │◄──────►│  Texture      │
│   (Client)    │  HTTP/  │   (Server)      │  Binary   │  │  Upload      │
│               │   WS     │                 │  Depth   │  │              │
└─────────────────┘         └──────────────────┘         └──────────────────┘
     │                         ▲                         │
     │                         │                         │
     └─────────────────────────┘─────────────────┘
           1. Upload video      2. Request depth
           2. Get session info    3. Receive binary
           3. Connect WS         4. Parse to float32
                                 5. Upload to GPU
```

### Key Components

1. **HTTP Client** - Upload video files, poll session status
2. **WebSocket Client** - Request depth frames, receive binary data
3. **Depth Parser** - Decode binary protocol (32-byte header + payload)
4. **Vulkan Manager** - Upload depth textures, create mesh/point cloud
5. **Depth Buffer** - Manage frame queue, handle drops/re-requests

---

## File Structure

Create new files following your existing patterns:

```
src/
├── depth_stream_client.h        # Main client interface
├── depth_stream_client.cpp      # HTTP + WebSocket implementation
├── depth_parser.h             # Binary protocol parser
├── depth_parser.cpp
├── depth_buffer.h             # Frame queue management
├── depth_buffer.cpp
└── depth_to_vk.h            # Vulkan texture upload utilities
```

### Integration Points

1. **`gaussian_splatting.h`** - Add `DepthStreamClient` member
2. **`gaussian_splatting.cpp`** - Initialize/update depth client
3. **`gaussian_splatting_ui.cpp`** - Add UI controls for depth streaming

---

## HTTP Client Implementation

### Option 1: Reuse Raw Socket Pattern (Recommended)

Since your codebase doesn't have an HTTP library and already uses raw sockets in `comfyui_client.cpp`, follow that pattern:

```cpp
// depth_stream_client.h
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
            float depthFps;
            float latencyMs;
            float inferAvgS;
            float queueAvgS;
            float wsSendAvgS;
            float dropCount;
        } rollingStats;
        struct Config {
            int inferenceWorkers;
            int processRes;
            int downsampleFactor;
        } config;
    };

    bool uploadVideo(const std::filesystem::path& videoPath, SessionInfo& outSession);
    bool getSessionStatus(const std::string& sessionId, SessionStatus& outStatus);
    bool deleteSession(const std::string& sessionId);
};
```

```cpp
// depth_stream_client.cpp - HTTP implementation
bool DepthStreamClient::uploadVideo(const std::filesystem::path& videoPath, SessionInfo& outSession) {
#ifdef _WIN32
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        return false;
    }
#endif

    int sock = static_cast<int>(socket(AF_INET, SOCK_STREAM, 0));
    if (sock < 0) {
        LOGE("Failed to create socket\n");
        return false;
    }

    struct sockaddr_in server{};
    server.sin_family = AF_INET;
    server.sin_port = htons(8000);  // Backend port
    inet_pton(AF_INET, "127.0.0.1", &server.sin_addr);

    if (::connect(sock, (struct sockaddr*)&server, sizeof(server)) < 0) {
        LOGE("Failed to connect to backend\n");
#ifdef _WIN32
        closesocket(sock);
        WSACleanup();
#else
        close(sock);
#endif
        return false;
    }

    // Read video file into memory
    std::ifstream file(videoPath, std::ios::binary);
    if (!file.is_open()) {
        LOGE("Failed to open video file: %s\n", videoPath.string().c_str());
        return false;
    }

    std::vector<uint8_t> videoData((std::istreambuf_iterator<char>(file),
                                   std::istreambuf_iterator<char>());

    // Create multipart/form-data boundary
    std::string boundary = "----WebKitFormBoundary7MA4YWxkTrZu0";
    std::stringstream request;

    request << "POST /api/sessions HTTP/1.1\r\n";
    request << "Host: 127.0.0.1:8000\r\n";
    request << "Content-Type: multipart/form-data; boundary=" << boundary << "\r\n";
    request << "Content-Length: " << (calculateContentLength(videoPath.string(), boundary, videoData.size())) << "\r\n";
    request << "Connection: close\r\n\r\n";

    request << "--" << boundary << "\r\n";
    request << "Content-Disposition: form-data; name=\"file\"; filename=\"" << videoPath.filename().string() << "\"\r\n";
    request << "Content-Type: video/mp4\r\n\r\n";

    // Send headers
    std::string requestStr = request.str();
    send(sock, requestStr.c_str(), static_cast<int>(requestStr.size()), 0);

    // Send video data
    send(sock, reinterpret_cast<const char*>(videoData.data()), static_cast<int>(videoData.size()), 0);
    request << "\r\n--" << boundary << "--\r\n";
    send(sock, "\r\n--", 5, 0);

    // Read response
    char buffer[8192];
    std::string response;
    int bytesReceived;
    while ((bytesReceived = recv(sock, buffer, sizeof(buffer) - 1, 0)) > 0) {
        buffer[bytesReceived] = '\0';
        response += buffer;
    }

#ifdef _WIN32
    closesocket(sock);
    WSACleanup();
#else
    close(sock);
#endif

    // Parse JSON response
    auto bodyStart = response.find("\r\n\r\n");
    if (bodyStart == std::string::npos) {
        LOGE("Invalid HTTP response\n");
        return false;
    }

    std::string body = response.substr(bodyStart + 4);
    try {
        auto json = nlohmann::json::parse(body);
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
```

### Option 2: Use cpr (C++ Requests Library)

If you prefer a more maintainable solution, add cpr to CMakeLists.txt:

```cmake
# In CMakeLists.txt
find_package(cpr REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE cpr::cpr)
```

```cpp
#include <cpr/cpr.h>

bool DepthStreamClient::uploadVideo(const std::filesystem::path& videoPath, SessionInfo& outSession) {
    cpr::Multipart multipart{
        cpr::Part{"file",
         cpr::File{videoPath.string()},
         "video/mp4"}
    };

    auto response = cpr::Post(
        cpr::Url{"http://127.0.0.1:8000/api/sessions"},
        cpr::Body{multipart},
        cpr::Timeout{std::chrono::seconds(30)}
    );

    if (response.status_code != 200) {
        LOGE("Upload failed with status: %d\n", response.status_code);
        return false;
    }

    try {
        auto json = nlohmann::json::parse(response.text);
        outSession.sessionId = json["session_id"].get<std::string>();
        outSession.width = json["width"].get<uint32_t>();
        outSession.height = json["height"].get<uint32_t>();
        outSession.fps = json["fps"].get<float>();
        outSession.durationMs = json.value("duration_ms", static_cast<uint64_t>(0));
        return true;
    } catch (const std::exception& e) {
        LOGE("Failed to parse session response: %s\n", e.what());
        return false;
    }
}
```

---

## WebSocket Client Implementation

### Header Structure

The depth frame uses a 32-byte binary header:

```cpp
// depth_parser.h
struct DepthHeader {
    char     magic[4];       // "VDZ1" (raw) or "VDZ2" (compressed)
    uint16_t version;       // Always 1
    uint16_t dataType;      // Always 1 (uint16)
    uint32_t timestampMs;   // Frame timestamp in ms
    uint32_t width;        // Depth map width
    uint32_t height;       // Depth map height
    float     scale;         // Quantization scale
    float     bias;          // Quantization bias
    float     zMax;         // Maximum depth value

    // Total: 32 bytes
} __attribute__((packed));
```

### Parser Implementation

```cpp
// depth_parser.cpp
#include "depth_parser.h"
#include <zlib.h>  // For decompression

struct DepthFrame {
    uint32_t timestampMs;
    uint32_t width;
    uint32_t height;
    std::vector<float> data;  // Depth values in meters
    float scale;
    float bias;
    float zMax;
};

bool parseDepthFrame(const std::vector<uint8_t>& buffer, DepthFrame& outFrame) {
    const size_t HEADER_SIZE = 32;
    if (buffer.size() < HEADER_SIZE) {
        LOGW("Buffer too small for depth frame\n");
        return false;
    }

    // Parse header
    DepthHeader header;
    memcpy(&header, buffer.data(), HEADER_SIZE);

    // Check magic bytes
    if (memcmp(header.magic, "VDZ1", 4) != 0 &&
        memcmp(header.magic, "VDZ2", 4) != 0) {
        LOGW("Invalid magic bytes in depth frame\n");
        return false;
    }

    if (header.version != 1 || header.dataType != 1) {
        LOGW("Unsupported depth frame version or data type\n");
        return false;
    }

    // Handle compression
    std::vector<uint16_t> samples;
    if (memcmp(header.magic, "VDZ2", 4) == 0) {
        // Decompress zlib
        uLongf decompressedSize = header.width * header.height * sizeof(uint16_t);
        samples.resize(decompressedSize);

        z_stream stream{};
        stream.next_in = buffer.data() + HEADER_SIZE;
        stream.avail_in = static_cast<uInt>(buffer.size() - HEADER_SIZE);
        stream.next_out = reinterpret_cast<Bytef*>(samples.data());
        stream.avail_out = decompressedSize;

        if (inflateInit2(&stream, -MAX_WBITS) != Z_OK) {
            LOGE("Failed to initialize zlib\n");
            return false;
        }

        int ret = inflate(&stream, Z_FINISH);
        inflateEnd(&stream);

        if (ret != Z_STREAM_END) {
            LOGE("Decompression failed with code: %d\n", ret);
            return false;
        }
    } else {
        // Raw data (VDZ1)
        const size_t dataSize = buffer.size() - HEADER_SIZE;
        const uint16_t* rawData = reinterpret_cast<const uint16_t*>(buffer.data() + HEADER_SIZE);
        samples.assign(rawData, rawData + dataSize / sizeof(uint16_t));
    }

    // Convert uint16 to float32 (meters)
    outFrame.timestampMs = header.timestampMs;
    outFrame.width = header.width;
    outFrame.height = header.height;
    outFrame.scale = header.scale;
    outFrame.bias = header.bias;
    outFrame.zMax = header.zMax;

    outFrame.data.resize(samples.size());
    for (size_t i = 0; i < samples.size(); ++i) {
        outFrame.data[i] = static_cast<float>(samples[i]) * header.scale + header.bias;
    }

    return true;
}
```

### WebSocket Client

Follow your existing `comfyui_client.cpp` pattern:

```cpp
// depth_stream_client.h
class DepthStreamClient {
private:
    using WsClient = websocketpp::client<websocketpp::config::asio_client>;

    void onDepthFrame(websocketpp::connection_hdl hdl, WsClient::message_ptr msg);
    void onDepthError(const std::string& message);

    WsClient                m_wsClient;
    websocketpp::connection_hdl m_wsConnection;
    std::thread              m_wsThread;
    std::atomic<bool>        m_wsShouldRun{false};
    std::atomic<bool>        m_connected{false};

    DepthBuffer             m_depthBuffer;  // Custom buffer for frame management

public:
    using DepthFrameCallback = std::function<void(const DepthFrame&)>;
    using StatusCallback = std::function<void(bool)>;

    DepthFrameCallback m_depthCallback;
    StatusCallback     m_statusCallback;
};
```

```cpp
// depth_stream_client.cpp
bool DepthStreamClient::connectWebSocket(const std::string& sessionId) {
    if (m_connected.load()) {
        disconnectWebSocket();
    }

    m_wsClient.clear_access_channels(websocketpp::log::alevel::all);
    m_wsClient.clear_error_channels(websocketpp::log::elevel::all);

    m_wsClient.init_asio();
    m_wsClient.set_message_handler([this](auto hdl, auto msg) { onDepthFrame(hdl, msg); });
    m_wsClient.set_open_handler([this](auto hdl) {
        m_wsConnection = hdl;
        m_connected.store(true);
        LOGI("Depth WebSocket connected\n");
        if (m_statusCallback) m_statusCallback(true);
    });
    m_wsClient.set_close_handler([this](auto hdl) {
        m_connected.store(false);
        LOGI("Depth WebSocket closed\n");
        if (m_statusCallback) m_statusCallback(false);
    });
    m_wsClient.set_fail_handler([this](auto hdl) {
        LOGE("Depth WebSocket connection failed\n");
        if (m_statusCallback) m_statusCallback(false);
    });

    try {
        std::string uri = "ws://127.0.0.1:8000/api/sessions/" + sessionId + "/stream";
        websocketpp::lib::error_code ec;
        auto con = m_wsClient.get_connection(uri, ec);

        if (ec) {
            LOGE("WebSocket URI error: %s\n", ec.message().c_str());
            return false;
        }

        m_wsClient.connect(con);
        m_wsShouldRun.store(true);
        m_wsThread = std::thread([this]() { m_wsClient.run(); });

        return true;
    } catch (const std::exception& e) {
        LOGE("WebSocket connect exception: %s\n", e.what());
        return false;
    }
}

void DepthStreamClient::disconnectWebSocket() {
    m_wsShouldRun.store(false);

    if (m_connected.load()) {
        try {
            m_wsClient.close(m_wsConnection,
                           websocketpp::close::status::normal,
                           "Client disconnecting");
        } catch (...) {}
    }

    m_wsClient.stop();

    if (m_wsThread.joinable()) {
        m_wsThread.join();
    }

    m_connected.store(false);
}

void DepthStreamClient::onDepthFrame(websocketpp::connection_hdl hdl,
                                   WsClient::message_ptr msg) {
    try {
        auto payload = msg->get_payload();

        // Check for JSON error message
        if (msg->get_opcode() == websocketpp::frame::opcode::text) {
            auto json = nlohmann::json::parse(payload);
            if (json.contains("type") && json["type"] == "error") {
                LOGE("Depth stream error: %s\n", json["message"].get<std::string>().c_str());
            }
            return;
        }

        // Binary depth frame
        std::vector<uint8_t> buffer(payload.begin(), payload.end());
        DepthFrame frame;

        if (parseDepthFrame(buffer, frame)) {
            m_depthBuffer.addFrame(frame);
            if (m_depthCallback) m_depthCallback(frame);
        }
    } catch (const std::exception& e) {
        LOGE("Error parsing depth frame: %s\n", e.what());
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
        m_wsClient.send(m_wsConnection, msg,
                      websocketpp::frame::opcode::text);

        return true;
    } catch (const std::exception& e) {
        LOGE("Failed to send depth request: %s\n", e.what());
        return false;
    }
}
```

---

## Vulkan Integration

### Texture Upload Strategy

Create depth textures and upload efficiently:

```cpp
// depth_to_vk.h
#include <vulkan/vulkan.hpp>
#include <nvvk/staging.hpp>

class DepthTextureManager {
private:
    vk::Device                 m_device;
    vk::PhysicalDevice           m_physicalDevice;
    vk::Queue                  m_graphicsQueue;
    std::unique_ptr<nvvk::StagingAllocator> m_stagingAllocator;

    struct DepthTexture {
        vk::Image image;
        vk::ImageView imageView;
        vk::DeviceMemory memory;
        vk::DescriptorSet descriptorSet;
        uint32_t width;
        uint32_t height;
        uint64_t lastUsedMs;
    };

    std::vector<DepthTexture> m_textures;
    uint32_t m_currentTextureIndex{0};

public:
    bool initialize(VkDevice device, VkPhysicalDevice physicalDevice,
                 VkQueue graphicsQueue, nvvk::Allocator* allocator);

    void uploadDepthFrame(const DepthFrame& frame);
    void cleanup();

    // Get current depth image for rendering
    const DepthTexture& getCurrentTexture() const;
};
```

```cpp
// depth_to_vk.cpp
bool DepthTextureManager::initialize(VkDevice device, VkPhysicalDevice physicalDevice,
                                VkQueue graphicsQueue, nvvk::Allocator* allocator) {
    m_device = device;
    m_physicalDevice = physicalDevice;
    m_graphicsQueue = graphicsQueue;

    m_stagingAllocator = std::make_unique<nvvk::StagingAllocator>(allocator);

    return true;
}

void DepthTextureManager::uploadDepthFrame(const DepthFrame& frame) {
    // Find or create texture
    DepthTexture* texture = findOrCreateTexture(frame.width, frame.height);

    if (!texture) {
        return;
    }

    // Upload using staging buffer
    size_t bufferSize = frame.data.size() * sizeof(float);

    auto staging = m_stagingAllocator->getBuffer(bufferSize,
                                            vk::BufferUsageFlagBits::eTransferSrc,
                                            vk::MemoryPropertyFlagBits::eHostVisible |
                                            vk::MemoryPropertyFlagBits::eHostCoherent);

    // Copy depth data to staging
    void* data = staging.map();
    memcpy(data, frame.data.data(), bufferSize);
    staging.unmap();

    // Copy to GPU
    vk::CommandBuffer cmd = ...;  // Allocate command buffer

    vk::BufferImageCopy region{};
    region.bufferOffset = 0;
    region.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = vk::Offset3D{0, 0, 0};
    region.imageExtent = vk::Extent3D{frame.width, frame.height, 1};

    cmd.copyBufferToImage(staging.buffer, texture->image, 1, &region);
    cmd.end();  // Submit and wait

    texture->width = frame.width;
    texture->height = frame.height;
    texture->lastUsedMs = frame.timestampMs;
}
```

### Integration with Gaussian Splatting

Two main approaches:

#### Option 1: Depth Displacement Mesh (Recommended)

Create a mesh from depth, similar to Three.js implementation:

```cpp
// In gaussian_splatting.cpp
class GaussianSplatting {
private:
    std::unique_ptr<DepthTextureManager> m_depthManager;
    std::unique_ptr<DepthStreamClient> m_depthClient;

    struct DepthMesh {
        vk::Buffer vertexBuffer;
        vk::Buffer indexBuffer;
        uint32_t vertexCount;
        uint32_t indexCount;
    } m_depthMesh;

public:
    void updateDepthMesh(const DepthFrame& frame) {
        // Generate vertices from depth
        generateDepthMesh(frame);

        // Update vertex buffers
        updateVertexBuffers();
    }

private:
    void generateDepthMesh(const DepthFrame& frame) {
        const float* depth = frame.data.data();
        const uint32_t width = frame.width;
        const uint32_t height = frame.height;

        std::vector<glm::vec3> vertices;
        std::vector<uint32_t> indices;

        // Create grid (like Three.js PlaneGeometry)
        for (uint32_t y = 0; y < height; ++y) {
            for (uint32_t x = 0; x < width; ++x) {
                size_t idx = y * width + x;
                float d = depth[idx];

                // Vertex position (depth in meters)
                vertices.push_back({
                    (static_cast<float>(x) / width) * 2.0f - 1.0f,  // X: [-1, 1]
                    (static_cast<float>(y) / height) * 2.0f - 1.0f, // Y: [-1, 1]
                    d * m_depthScale  // Z: depth
                });
            }
        }

        // Generate indices for triangles
        for (uint32_t y = 0; y < height - 1; ++y) {
            for (uint32_t x = 0; x < width - 1; ++x) {
                uint32_t i0 = y * width + x;
                uint32_t i1 = i0 + 1;
                uint32_t i2 = i0 + width;
                uint32_t i3 = i1 + width;

                indices.insert(indices.end(), {i0, i1, i2, i3});
            }
        }

        // Upload to GPU
        uploadMeshToGPU(vertices, indices);
    }
};
```

#### Option 2: Depth as Gaussian Splat

Convert depth to point cloud and render as splats:

```cpp
void GaussianSplatting::generateDepthSplats(const DepthFrame& frame) {
    const float* depth = frame.data.data();
    const uint32_t width = frame.width;
    const uint32_t height = frame.height;

    std::vector<Splat> splats;

    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            size_t idx = y * width + x;
            float d = depth[idx];

            // Only add valid depth points
            if (d > 0.01f && d < frame.zMax) {
                Splat splat;
                splat.position = {
                    (static_cast<float>(x) / width) * 2.0f - 1.0f,
                    (static_cast<float>(y) / height) * 2.0f - 1.0f,
                    d * m_depthScale
                };

                // Color could be from video frame
                splat.color = {1.0f, 1.0f, 1.0f};
                splat.opacity = 0.8f;

                splats.push_back(splat);
            }
        }
    }

    // Upload as point cloud
    uploadSplatCloud(splats);
}
```

---

## Flow Control & Buffering

### Depth Buffer Implementation

Manage frame queue, handle drops, re-requests:

```cpp
// depth_buffer.h
class DepthBuffer {
public:
    DepthBuffer(size_t maxPending = 60) : m_maxPending(maxPending) {}

    void addFrame(const DepthFrame& frame);

    // Get best frame for current timestamp
    bool getFrame(uint64_t targetMs, DepthFrame& outFrame);

    size_t getPendingCount() const { return m_pendingFrames.size(); }

private:
    struct PendingFrame {
        uint64_t timestampMs;
        float rtt;  // Round-trip time
        double sentTime;
    };

    std::deque<PendingFrame> m_pendingFrames;
    std::map<uint64_t, DepthFrame> m_receivedFrames;  // Cache for re-requests
    size_t m_maxPending;

    uint64_t m_lastTimestamp{0};
    float m_rtt{0.0f};
};

void DepthBuffer::addFrame(const DepthFrame& frame) {
    // Store in cache
    m_receivedFrames[frame.timestampMs] = frame;

    // Calculate RTT
    double now = getCurrentTimeMs();
    auto it = std::find_if(m_pendingFrames.begin(), m_pendingFrames.end(),
                          [&](const PendingFrame& pf) {
                              return pf.timestampMs == frame.timestampMs;
                          });

    if (it != m_pendingFrames.end()) {
        float rtt = static_cast<float>(now - it->sentTime);

        // EMA for RTT (exponential moving average)
        const float alpha = 0.1f;
        m_rtt = m_rtt * (1.0f - alpha) + rtt * alpha;

        m_pendingFrames.erase(it);
    }
}

bool DepthBuffer::getFrame(uint64_t targetMs, DepthFrame& outFrame) {
    // Check cache
    auto cacheIt = m_receivedFrames.find(targetMs);
    if (cacheIt != m_receivedFrames.end()) {
        outFrame = cacheIt->second;
        LOGD("Cache hit for timestamp: %llu\n", targetMs);
        return true;
    }

    // Check pending queue
    auto it = std::find_if(m_pendingFrames.begin(), m_pendingFrames.end(),
                          [&](const PendingFrame& pf) {
                              return pf.timestampMs == targetMs;
                          });

    if (it != m_pendingFrames.end()) {
        // Frame is pending, wait
        return false;
    }

    // Add to pending queue
    if (m_pendingFrames.size() >= m_maxPending) {
        // Drop oldest (FIFO)
        LOGW("Dropping pending frame, queue full\n");
        m_pendingFrames.pop_front();
    }

    PendingFrame pf;
    pf.timestampMs = targetMs;
    pf.sentTime = getCurrentTimeMs();
    m_pendingFrames.push_back(pf);

    // Request from backend
    return m_depthClient->requestDepth(targetMs);
}
```

---

## Performance Considerations

### Memory Management

1. **Texture Pooling**: Reuse textures instead of creating/destroying each frame
2. **Staging Buffers**: Use ring buffer for staging uploads
3. **Frame Caching**: Keep recent frames in CPU cache for re-requests
4. **Batch Requests**: Request multiple frames ahead (predictive prefetching)

### Bandwidth Optimization

1. **Compression**: The backend uses zlib (VDZ2 magic) - enable decompression
2. **Downsampling**: Backend may downsample based on `VIDEO_DEPTH_DOWNSAMPLE`
3. **Request Throttling**: Respect `maxInflight` limit (default 8-16)

### Synchronization

1. **Frame Order**: Use timestamps for ordering, not arrival order
2. **Drop Handling**: When frames drop, skip to next valid frame
3. **Seek Support**: Detect large timestamp jumps (>500ms diff) and handle as seeks

---

## Code Examples

### Full Integration Example

```cpp
// gaussian_splatting.h
#pragma once

#include "depth_stream_client.h"
#include "depth_to_vk.h"
#include <memory>

class GaussianSplatting {
private:
    std::unique_ptr<DepthStreamClient> m_depthClient;
    std::unique_ptr<DepthTextureManager> m_depthManager;

    float m_depthScale{1.0f};
    float m_depthBias{0.0f};
    bool m_enableDepthRendering{false};

public:
    // Existing interface
    void loadScene(const std::filesystem::path& path);
    void render(const vk::CommandBuffer& cmd);

    // New depth streaming methods
    void enableDepthRendering(const std::string& videoPath);
    void updateDepthRendering();
};
```

```cpp
// gaussian_splatting.cpp
void GaussianSplatting::enableDepthRendering(const std::string& videoPath) {
    if (!m_depthClient) {
        m_depthClient = std::make_unique<DepthStreamClient>();
        m_depthClient->setDepthCallback([this](const DepthFrame& frame) {
            m_depthManager->uploadDepthFrame(frame);
        });
    }

    // Upload video
    DepthStreamClient::SessionInfo session;
    if (!m_depthClient->uploadVideo(videoPath, session)) {
        LOGE("Failed to upload video\n");
        return;
    }

    // Connect WebSocket
    if (!m_depthClient->connectWebSocket(session.sessionId)) {
        LOGE("Failed to connect depth stream\n");
        return;
    }

    m_enableDepthRendering = true;
}

void GaussianSplatting::updateDepthRendering() {
    if (!m_enableDepthRendering) {
        return;
    }

    // Update depth texture
    auto& texture = m_depthManager->getCurrentTexture();

    // Use in shader for displacement
    // Or render as separate pass
}

void GaussianSplatting::render(const vk::CommandBuffer& cmd) {
    // Render existing splats
    renderSplats(cmd);

    // Render depth mesh/splats if enabled
    if (m_enableDepthRendering) {
        renderDepthMesh(cmd);
    }
}

void GaussianSplatting::renderDepthMesh(const vk::CommandBuffer& cmd) {
    auto& depthTexture = m_depthManager->getCurrentTexture();
    if (!depthTexture.imageView) {
        return;
    }

    // Bind depth texture as displacement map
    vk::DescriptorImageInfo imageInfo;
    imageInfo.sampler = m_depthSampler;
    imageInfo.imageView = depthTexture.imageView;
    imageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

    cmd.bindDescriptorSets(..., {imageInfo});

    // Draw mesh
    cmd.bindPipeline(m_depthPipeline);
    cmd.drawIndexed(m_depthMesh.indexCount, 1, 0, 0);
}
```

---

## Additional Recommendations

### Error Handling

1. **WebSocket Disconnects**: Auto-reconnect with exponential backoff
2. **Session Expiry**: Delete sessions when done
3. **Timeout Handling**: Implement request timeouts
4. **Fallback Mode**: Show placeholder mesh when depth fails

### UI Integration

Add controls in `gaussian_splatting_ui.cpp`:

```cpp
void GaussianSplattingUI::renderDepthControls() {
    ImGui::Checkbox("Enable Depth Rendering", &m_enableDepth);

    if (ImGui::Button("Load Video File")) {
        ImGuiFileDialog::openFile("Select Video", ".mp4", [this](const auto& path) {
            m_app->enableDepthRendering(path);
        });
    }

    ImGui::SliderFloat("Depth Scale", &m_depthScale, 0.1f, 10.0f);
    ImGui::SliderFloat("Depth Bias", &m_depthBias, -5.0f, 5.0f);

    // Display session info
    if (m_depthClient) {
        auto status = m_depthClient->getSessionStatus();
        ImGui::Text("FPS: %.1f", status.info.fps);
        ImGui::Text("Buffer: %u", status.bufferLength);
        ImGui::Text("Dropped: %.0f", status.rollingStats.dropCount);
    }
}
```

### CMake Integration

Add to your `CMakeLists.txt`:

```cmake
# Add these near existing ComfyUI section

option(ENABLE_DEPTH_STREAMING "Enable VideoDepthViewer3D depth streaming" ON)

if(ENABLE_DEPTH_STREAMING)
    # Depth streaming source files
    set(DEPTH_STREAMING_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/depth_stream_client.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/depth_parser.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/depth_buffer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/depth_to_vk.cpp
    )

    add_library(depth_streaming STATIC ${DEPTH_STREAMING_SOURCES})

    target_include_directories(depth_streaming
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${websocketpp_SOURCE_DIR}
    )

    # Link zlib for decompression
    find_package(ZLIB REQUIRED)

    target_link_libraries(${PROJECT_NAME}
        PRIVATE
            depth_streaming
            ZLIB::ZLIB
    )

    message(STATUS "Depth streaming enabled")
endif()
```

---

## Troubleshooting

### Common Issues

1. **Socket Errors**: Ensure backend is running on port 8000
2. **Parse Failures**: Check magic bytes match expected ("VDZ1" or "VDZ2")
3. **Texture Upload Failures**: Check Vulkan memory allocation
4. **Stuttering**: Increase `maxInflight` or reduce resolution
5. **Desync**: Use timestamps for frame synchronization

### Debugging

Enable debug logging:

```cpp
#ifdef DEBUG_DEPTH_STREAMING
#define LOGD(...) LOGI(__VA_ARGS__)
#else
#define LOGD(...)
#endif
```

Check log output:
- Connection establishment
- Frame reception rates
- Parse errors
- Vulkan upload failures

---

## References

- **VideoDepthViewer3D Backend Protocol**: See `backend/routers/stream.py`
- **WebSocket++ Documentation**: https://github.com/zaphoyd/websocketpp
- **Zlib**: https://zlib.net/
- **Vulkan Texture Upload**: See your existing `nvvk::StagingAllocator` usage

---

## Summary

This integration provides real-time depth streaming from VideoDepthViewer3D to vk_gaussian_splatting, enabling:

1. **Video Upload** - Upload MP4 files to backend
2. **Depth Inference** - Request per-frame depth estimation
3. **Vulkan Rendering** - Upload depth as textures or point clouds
4. **Flow Control** - Manage frame drops and re-requests
5. **Error Handling** - Robust disconnect and error recovery

The implementation follows existing codebase patterns (WebSocket++, raw sockets, nlohmann/json, nvutils logging) and integrates seamlessly with your Vulkan rendering pipeline.
