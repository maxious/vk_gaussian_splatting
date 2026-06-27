#pragma once

#include "protocol.h"

#include <cstddef>
#include <cstdint>
#include <vector>

class TcpProtocolParser {
public:
    TcpProtocolParser() = default;

    int feed(const uint8_t* data, size_t len);

    const uint8_t* nextMessage(size_t& out_len, uint32_t& out_type);

    void reset();

    bool hasMessage() const;

private:
    std::vector<uint8_t> m_buffer;
    std::vector<std::vector<uint8_t>> m_messages;
    std::vector<std::vector<uint8_t>> m_returned_messages;
};

class TcpProtocolSerializer {
public:
    static std::vector<uint8_t> serializeFrameRequest(uint32_t frame_index,
                                                      uint32_t timestamp_ms,
                                                      uint32_t width,
                                                      uint32_t height,
                                                      const uint8_t* rgb_data);

    static std::vector<uint8_t> serializeDepthResponse(uint32_t frame_index,
                                                       uint32_t timestamp_ms,
                                                       uint32_t width,
                                                       uint32_t height,
                                                       float scale,
                                                       float bias,
                                                       float z_max,
                                                       const float* depth_data);

    static std::vector<uint8_t> serializeServerStatus(uint32_t active_workers,
                                                      uint32_t queue_depth,
                                                      float avg_processing_ms,
                                                      const char* model_hash,
                                                      const char* backend_name);

    static std::vector<uint8_t> serializeShutdown(uint32_t drain_timeout_ms);

    static std::vector<uint8_t> serializeShutdownAck();

    static std::vector<uint8_t> serializeError(uint32_t error_code, const char* error_msg);

    static std::vector<uint8_t> serializeSplatRequest(uint32_t n_views,
                                                      uint32_t width,
                                                      uint32_t height,
                                                      const SplatRequestOptions& options,
                                                      const uint8_t* const* image_ptrs,
                                                      const uint32_t* image_sizes,
                                                      uint32_t image_count);

    static std::vector<uint8_t> serializeSplatPoll(uint32_t job_id);

    static std::vector<uint8_t> serializeSplatProgress(uint32_t job_id,
                                                      uint8_t stage,
                                                      uint8_t percent,
                                                      uint32_t eta_ms);

    static std::vector<uint8_t> serializeSplatResponse(uint32_t job_id,
                                                       uint32_t n_gaussians,
                                                       const char* path);

    static std::vector<uint8_t> serializeSplatCancel(uint32_t job_id);

    static std::vector<uint8_t> serializeSplatError(uint32_t job_id,
                                                    uint32_t error_code,
                                                    const char* error_msg);
};

bool decodeSplatRequest(const uint8_t* message,
                        size_t message_len,
                        SplatRequestPayload& payload,
                        const uint8_t*& options,
                        const uint8_t*& images,
                        size_t& images_size);
