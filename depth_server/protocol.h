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
    // Free-splatter (image-to-3DGS) message types
    MSG_SPLAT_REQUEST  = 0x10,  // client -> server: submit N images for splat generation
    MSG_SPLAT_POLL     = 0x11,   // client -> server: poll job status
    MSG_SPLAT_PROGRESS = 0x12,   // server -> client: job progress update
    MSG_SPLAT_RESPONSE = 0x13,   // server -> client: job complete, .splat file path
    MSG_SPLAT_CANCEL   = 0x14,   // client -> server: cancel in-flight job
    MSG_SPLAT_ERROR    = 0x15,   // server -> client: job failed (like MSG_ERROR but with job_id)
    // Cloud streaming (point cloud reconstruction from JPEG frames)
    MSG_CLOUD_REQUEST  = 0x06,  // client -> server: submit N frame paths for cloud reconstruction
    MSG_CLOUD_POLL     = 0x07,  // client -> server: poll cloud job status
    MSG_CLOUD_PROGRESS = 0x08,  // server -> client: cloud job progress update
    MSG_CLOUD_RESPONSE = 0x09,  // server -> client: cloud job complete, .splat file path
    MSG_CLOUD_CANCEL   = 0x0A,  // client -> server: cancel in-flight cloud job
    MSG_CLOUD_ERROR    = 0x0B,  // server -> client: cloud job failed (like MSG_ERROR but with job_id)
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

// Free-splatter (image-to-3DGS) payload structs
static constexpr uint32_t MAX_IMAGE_SIZE        = 16 * 1024 * 1024;  // 16MB per image
static constexpr uint32_t MAX_PENDING_SPLAT_JOBS = 32;

struct SplatRequestOptions {  // flat binary, no JSON - 16 bytes
    float    downsample;      // 0.25-1.0, default 1.0
    uint8_t  sh_degree;       // 0 or 1, default 0
    uint8_t  variant;         // 0=scene, 1=object, default 0
    uint8_t  _pad[2];
    uint32_t timeout_ms;      // server-side timeout, 0=no timeout
    uint32_t flags;           // bit 0: use_vulkan
};

struct SplatRequestPayload {
    uint32_t n_views;          // 2-4
    uint32_t width;            // image width (post-decode, 0 = auto)
    uint32_t height;           // image height (post-decode, 0 = auto)
    uint32_t options_size;     // sizeof(SplatRequestOptions) = 16
    uint32_t image_data_size;  // total bytes of all images + length prefix
    // followed by: SplatRequestOptions (16 bytes), then for each view:
    //   uint32_t len + image_bytes[len]
};

struct SplatPollPayload {
    uint32_t job_id;
};

struct SplatProgressPayload {
    uint32_t job_id;
    uint8_t  stage;       // 0=queued, 1=preprocessing, 2=inference, 3=writing, 4=done
    uint8_t  percent;     // 0-100
    uint16_t _pad;
    uint32_t eta_ms;
};

struct SplatResponsePayload {
    uint32_t job_id;
    uint32_t n_gaussians;
    uint32_t path_length;  // length of null-terminated path string
    // followed by: char path[path_length]
};

struct SplatCancelPayload {
    uint32_t job_id;
};

struct SplatErrorPayload {
    uint32_t job_id;
    uint32_t error_code;  // 0=unknown, 1=queue_full, 2=invalid_input, 3=worker_crash, 4=timeout
    char     error_msg[252];
};
// Cloud streaming (video → coherent point cloud) payload structs
static constexpr uint32_t MAX_CLOUD_FRAMES   = 200;
static constexpr uint32_t MAX_CLOUD_PATH_LEN = 4096;

struct CloudRequestOptions {
    uint32_t n_frames;          // 2..MAX_CLOUD_FRAMES
    uint32_t chunk_size;        // 2..24
    uint32_t overlap;           // 0..chunk_size-1
    float    conf_pct;          // 0..100
    float    point_size;        // > 0
    uint32_t global_budget;     // 0 = unlimited
    uint8_t  icp_refine;        // 0/1
    uint8_t  loop_close;        // 0/1
    uint8_t  fuse;              // 0/1
    uint8_t  metric;            // 0/1
    float    fuse_voxel_frac;   // <=0 => 0.004
    float    fuse_trunc_mult;   // <=0 => 4
    uint32_t timeout_ms;        // 0 = no timeout
    uint32_t flags;             // bit 0 = use_vulkan
    uint8_t  _pad[20];          // align to 64 bytes
};

struct CloudRequestPayload {
    uint32_t n_frames;          // 2..MAX_CLOUD_FRAMES
    uint32_t options_size;      // sizeof(CloudRequestOptions)
    uint32_t frame_data_size;   // total bytes of all frame paths + length prefixes
};

struct CloudPollPayload {
    uint32_t job_id;
};

struct CloudCancelPayload {
    uint32_t job_id;
};

struct CloudProgressPayload {
    uint32_t job_id;
    uint8_t  stage;            // 0=queued, 1=extracting, 2=inference, 3=writing, 4=done
    uint8_t  percent;          // 0..100
    uint16_t windows_done;
    uint16_t windows_total;
    uint16_t _pad;
    uint32_t eta_ms;
};

struct CloudResponsePayload {
    uint32_t job_id;
    uint32_t n_points;
    uint32_t path_length;      // length of null-terminated path string
};

struct CloudErrorPayload {
    uint32_t job_id;
    uint32_t error_code;       // 0=unknown, 1=queue_full, 2=invalid_input, 3=worker_crash, 4=timeout, 5=invalid_model, 6=cancelled, 7=oom
    char     error_msg[252];
};
#pragma pack(pop)

static_assert(sizeof(FrameHeader) == 12, "FrameHeader must be 12 bytes");
static_assert(sizeof(CloudRequestOptions)  == 64, "CloudRequestOptions must be 64 bytes");
static_assert(sizeof(CloudRequestPayload)  == 12, "CloudRequestPayload must be 12 bytes");
static_assert(sizeof(CloudPollPayload)     == 4,  "CloudPollPayload must be 4 bytes");
static_assert(sizeof(CloudProgressPayload)  == 16, "CloudProgressPayload must be 16 bytes");
static_assert(sizeof(CloudResponsePayload) == 12, "CloudResponsePayload must be 12 bytes");
static_assert(sizeof(CloudCancelPayload)   == 4,  "CloudCancelPayload must be 4 bytes");
static_assert(sizeof(CloudErrorPayload)    == 260, "CloudErrorPayload must be 260 bytes");
static_assert(sizeof(SplatRequestOptions)  == 16, "SplatRequestOptions must be 16 bytes");
static_assert(sizeof(SplatRequestPayload)  == 20, "SplatRequestPayload must be 20 bytes");
static_assert(sizeof(SplatPollPayload)     == 4,  "SplatPollPayload must be 4 bytes");
static_assert(sizeof(SplatProgressPayload)  == 12, "SplatProgressPayload must be 12 bytes");
static_assert(sizeof(SplatResponsePayload) == 12, "SplatResponsePayload must be 12 bytes");
static_assert(sizeof(SplatCancelPayload)   == 4,  "SplatCancelPayload must be 4 bytes");
static_assert(sizeof(SplatErrorPayload)    == 260, "SplatErrorPayload must be 260 bytes");

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
        case MSG_CLOUD_REQUEST:
        case MSG_CLOUD_POLL:
        case MSG_CLOUD_PROGRESS:
        case MSG_CLOUD_RESPONSE:
        case MSG_CLOUD_CANCEL:
        case MSG_CLOUD_ERROR:
        case MSG_SPLAT_REQUEST:
        case MSG_SPLAT_POLL:
        case MSG_SPLAT_PROGRESS:
        case MSG_SPLAT_RESPONSE:
        case MSG_SPLAT_CANCEL:
        case MSG_SPLAT_ERROR:
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
        case MSG_CLOUD_REQUEST: return "CLOUD_REQUEST";
        case MSG_CLOUD_POLL: return "CLOUD_POLL";
        case MSG_CLOUD_PROGRESS: return "CLOUD_PROGRESS";
        case MSG_CLOUD_RESPONSE: return "CLOUD_RESPONSE";
        case MSG_CLOUD_CANCEL: return "CLOUD_CANCEL";
        case MSG_CLOUD_ERROR: return "CLOUD_ERROR";
        case MSG_SPLAT_REQUEST: return "SPLAT_REQUEST";
        case MSG_SPLAT_POLL: return "SPLAT_POLL";
        case MSG_SPLAT_PROGRESS: return "SPLAT_PROGRESS";
        case MSG_SPLAT_RESPONSE: return "SPLAT_RESPONSE";
        case MSG_SPLAT_CANCEL: return "SPLAT_CANCEL";
        case MSG_SPLAT_ERROR: return "SPLAT_ERROR";
        case MSG_ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

// Helper: compute payload size from total frame length
inline size_t computePayloadSize(const FrameHeader* header) {
    if (!header || header->frame_length < sizeof(FrameHeader)) return 0;
    return header->frame_length - sizeof(FrameHeader);
}
