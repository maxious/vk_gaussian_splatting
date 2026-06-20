#pragma once

#include <cstdint>
#include <cstring>

// Magic bytes: "DEPT" in ASCII
static constexpr uint32_t TCP_DEPTH_MAGIC = 0x44455054;

// Max message size (100MB)
static constexpr uint32_t MAX_FRAME_SIZE = 100 * 1024 * 1024;

// Message types
enum MessageType : uint32_t {
    MSG_FRAME_REQUEST = 0x01,
    MSG_DEPTH_RESPONSE = 0x02,
    MSG_SERVER_STATUS = 0x03,
    MSG_SHUTDOWN = 0x04,
    MSG_SHUTDOWN_ACK = 0x05,
    MSG_ERROR = 0xFF,
};

#pragma pack(push, 1)
struct FrameHeader {
    uint32_t magic;        // TCP_DEPTH_MAGIC
    uint32_t frame_length; // Total message size including this header
    uint32_t message_type; // One of MessageType enum
};

struct FrameRequestPayload {
    uint32_t width;
    uint32_t height;
    uint32_t timestamp_ms;
    // Followed by rgb_data[width * height * 3] (HWC uint8)
};

struct DepthResponsePayload {
    uint32_t width;
    uint32_t height;
    uint32_t timestamp_ms;
    float scale;
    float bias;
    float z_max;
    // Followed by depth_data[width * height] (float32)
};

struct ServerStatusPayload {
    uint32_t active_workers;
    uint32_t queue_depth;
    float avg_processing_ms;
    char model_hash[32];   // SHA256 hex of loaded model
    char backend_name[16]; // "cpu", "cuda", etc.
};

struct ShutdownPayload {
    uint32_t drain_timeout_ms;
};

struct ErrorPayload {
    uint32_t error_code;
    char error_msg[256];
};
#pragma pack(pop)

static_assert(sizeof(FrameHeader) == 12, "FrameHeader must be 12 bytes");

// Helper: validate header fields
inline bool validateHeader(const FrameHeader* header, size_t buffer_size) {
    if (!header) return false;
    if (buffer_size < sizeof(FrameHeader)) return false;
    if (header->magic != TCP_DEPTH_MAGIC) return false;
    if (header->frame_length < sizeof(FrameHeader)) return false;
    if (header->frame_length > MAX_FRAME_SIZE) return false;
    switch (header->message_type) {
        case MSG_FRAME_REQUEST:
        case MSG_DEPTH_RESPONSE:
        case MSG_SERVER_STATUS:
        case MSG_SHUTDOWN:
        case MSG_SHUTDOWN_ACK:
        case MSG_ERROR:
            return true;
        default:
            return false;
    }
}

// Helper: get human-readable type name
inline const char* getMessageTypeName(uint32_t type) {
    switch (type) {
        case MSG_FRAME_REQUEST: return "FRAME_REQUEST";
        case MSG_DEPTH_RESPONSE: return "DEPTH_RESPONSE";
        case MSG_SERVER_STATUS: return "SERVER_STATUS";
        case MSG_SHUTDOWN: return "SHUTDOWN";
        case MSG_SHUTDOWN_ACK: return "SHUTDOWN_ACK";
        case MSG_ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

// Helper: compute payload size from total frame length
inline size_t computePayloadSize(const FrameHeader* header) {
    if (!header || header->frame_length < sizeof(FrameHeader)) return 0;
    return header->frame_length - sizeof(FrameHeader);
}
