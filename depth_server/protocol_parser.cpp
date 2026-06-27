#include "protocol_parser.h"
#include "protocol.h"

#include <nvutils/logger.hpp>

#include <algorithm>
#include <cstring>
#include <limits>

namespace {

bool checkedAdd(size_t a, size_t b, size_t& out)
{
    if(a > std::numeric_limits<size_t>::max() - b)
    {
        return false;
    }
    out = a + b;
    return true;
}

bool checkedMul(size_t a, size_t b, size_t& out)
{
    if(a != 0 && b > std::numeric_limits<size_t>::max() / a)
    {
        return false;
    }
    out = a * b;
    return true;
}

size_t minimumFrameSize(uint32_t message_type)
{
    switch(message_type)
    {
    case MSG_FRAME_REQUEST: return sizeof(FrameHeader) + sizeof(FrameRequestPayload);
    case MSG_DEPTH_RESPONSE: return sizeof(FrameHeader) + sizeof(DepthResponsePayload);
    case MSG_SERVER_STATUS: return sizeof(FrameHeader) + sizeof(ServerStatusPayload);
    case MSG_SHUTDOWN: return sizeof(FrameHeader) + sizeof(ShutdownPayload);
    case MSG_SHUTDOWN_ACK: return sizeof(FrameHeader);
    case MSG_ERROR: return sizeof(FrameHeader) + sizeof(ErrorPayload);
    case MSG_SPLAT_REQUEST: return sizeof(FrameHeader) + sizeof(SplatRequestPayload) + sizeof(SplatRequestOptions);
    case MSG_SPLAT_POLL: return sizeof(FrameHeader) + sizeof(SplatPollPayload);
    case MSG_SPLAT_PROGRESS: return sizeof(FrameHeader) + sizeof(SplatProgressPayload);
    case MSG_SPLAT_RESPONSE: return sizeof(FrameHeader) + sizeof(SplatResponsePayload);
    case MSG_SPLAT_CANCEL: return sizeof(FrameHeader) + sizeof(SplatCancelPayload);
    case MSG_SPLAT_ERROR: return sizeof(FrameHeader) + sizeof(SplatErrorPayload);
    default: return 0;
    }
}

std::vector<uint8_t> makeFrame(uint32_t message_type, size_t payload_size)
{
    const size_t total_size = sizeof(FrameHeader) + payload_size;
    if(total_size > MAX_FRAME_SIZE || total_size > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
    {
        LOGE("TcpProtocolSerializer: frame too large (%zu bytes)\n", total_size);
        return {};
    }

    std::vector<uint8_t> frame(total_size);

    FrameHeader header{};
    header.magic = TCP_DEPTH_MAGIC;
    header.frame_length = static_cast<uint32_t>(total_size);
    header.message_type = message_type;
    std::memcpy(frame.data(), &header, sizeof(header));
    return frame;
}

void copyTextField(char* dst, size_t dst_size, const char* src)
{
    std::memset(dst, 0, dst_size);
    if(src == nullptr || dst_size == 0)
    {
        return;
    }

    const size_t src_len = std::strlen(src);
    const size_t copy_len = std::min(dst_size - 1, src_len);
    if(copy_len != 0)
    {
        std::memcpy(dst, src, copy_len);
    }
}

} // namespace

int TcpProtocolParser::feed(const uint8_t* data, size_t len)
{
    if(data == nullptr || len == 0)
    {
        return 0;
    }

    m_buffer.insert(m_buffer.end(), data, data + len);

    int extracted = 0;
    for(;;)
    {
        if(m_buffer.size() < sizeof(FrameHeader))
        {
            break;
        }

        FrameHeader header{};
        std::memcpy(&header, m_buffer.data(), sizeof(header));
        if(!validateHeader(&header, m_buffer.size()))
        {
            LOGE("TcpProtocolParser: invalid frame header (magic=0x%08x, type=%u, length=%u)\n",
                 header.magic, header.message_type, header.frame_length);
            reset();
            return extracted;
        }

        const size_t min_size = minimumFrameSize(header.message_type);
        if(min_size == 0 || header.frame_length < min_size)
        {
            LOGE("TcpProtocolParser: frame too small for %s (length=%u, min=%zu)\n",
                 getMessageTypeName(header.message_type), header.frame_length, min_size);
            reset();
            return extracted;
        }

        if(m_buffer.size() < header.frame_length)
        {
            break;
        }

        std::vector<uint8_t> message(header.frame_length);
        std::memcpy(message.data(), m_buffer.data(), header.frame_length);
        m_messages.push_back(std::move(message));
        m_buffer.erase(m_buffer.begin(), m_buffer.begin() + static_cast<std::ptrdiff_t>(header.frame_length));
        ++extracted;
    }

    return extracted;
}

const uint8_t* TcpProtocolParser::nextMessage(size_t& out_len, uint32_t& out_type)
{
    if(m_messages.empty())
    {
        out_len = 0;
        out_type = 0;
        return nullptr;
    }

    m_returned_messages.push_back(std::move(m_messages.front()));
    m_messages.erase(m_messages.begin());

    const std::vector<uint8_t>& message = m_returned_messages.back();
    FrameHeader header{};
    std::memcpy(&header, message.data(), sizeof(header));
    out_len = message.size();
    out_type = header.message_type;
    return message.data();
}

void TcpProtocolParser::reset()
{
    m_buffer.clear();
    m_messages.clear();
    m_returned_messages.clear();
}

bool TcpProtocolParser::hasMessage() const
{
    return !m_messages.empty();
}

std::vector<uint8_t> TcpProtocolSerializer::serializeFrameRequest(uint32_t frame_index,
                                                                  uint32_t timestamp_ms,
                                                                  uint32_t width,
                                                                  uint32_t height,
                                                                  const uint8_t* rgb_data)
{
    (void)frame_index;
    if(rgb_data == nullptr && width != 0 && height != 0)
    {
        LOGE("TcpProtocolSerializer: null rgb_data for frame request\n");
        return {};
    }

    size_t pixel_count = 0;
    size_t rgb_size = 0;
    if(!checkedMul(static_cast<size_t>(width), static_cast<size_t>(height), pixel_count) ||
       !checkedMul(pixel_count, static_cast<size_t>(3), rgb_size))
    {
        LOGE("TcpProtocolSerializer: frame request size overflow (%u x %u)\n", width, height);
        return {};
    }

    size_t payload_size = 0;
    if(!checkedAdd(sizeof(FrameRequestPayload), rgb_size, payload_size))
    {
        LOGE("TcpProtocolSerializer: frame request payload overflow\n");
        return {};
    }

    auto frame = makeFrame(MSG_FRAME_REQUEST, payload_size);
    FrameRequestPayload payload{};
    payload.width = width;
    payload.height = height;
    payload.timestamp_ms = timestamp_ms;
    std::memcpy(frame.data() + sizeof(FrameHeader), &payload, sizeof(payload));
    if(rgb_size != 0)
    {
        std::memcpy(frame.data() + sizeof(FrameHeader) + sizeof(FrameRequestPayload), rgb_data, rgb_size);
    }
    return frame;
}

std::vector<uint8_t> TcpProtocolSerializer::serializeDepthResponse(uint32_t frame_index,
                                                                   uint32_t timestamp_ms,
                                                                   uint32_t width,
                                                                   uint32_t height,
                                                                   float scale,
                                                                   float bias,
                                                                   float z_max,
                                                                   const float* depth_data)
{
    (void)frame_index;
    if(depth_data == nullptr && width != 0 && height != 0)
    {
        LOGE("TcpProtocolSerializer: null depth_data for depth response\n");
        return {};
    }

    size_t depth_count = 0;
    size_t depth_bytes = 0;
    if(!checkedMul(static_cast<size_t>(width), static_cast<size_t>(height), depth_count) ||
       !checkedMul(depth_count, sizeof(float), depth_bytes))
    {
        LOGE("TcpProtocolSerializer: depth response size overflow (%u x %u)\n", width, height);
        return {};
    }

    size_t payload_size = 0;
    if(!checkedAdd(sizeof(DepthResponsePayload), depth_bytes, payload_size))
    {
        LOGE("TcpProtocolSerializer: depth response payload overflow\n");
        return {};
    }

    auto frame = makeFrame(MSG_DEPTH_RESPONSE, payload_size);
    DepthResponsePayload payload{};
    payload.width = width;
    payload.height = height;
    payload.timestamp_ms = timestamp_ms;
    payload.scale = scale;
    payload.bias = bias;
    payload.z_max = z_max;
    std::memcpy(frame.data() + sizeof(FrameHeader), &payload, sizeof(payload));
    if(depth_bytes != 0)
    {
        std::memcpy(frame.data() + sizeof(FrameHeader) + sizeof(DepthResponsePayload), depth_data, depth_bytes);
    }
    return frame;
}

std::vector<uint8_t> TcpProtocolSerializer::serializeServerStatus(uint32_t active_workers,
                                                                  uint32_t queue_depth,
                                                                  float avg_processing_ms,
                                                                  const char* model_hash,
                                                                  const char* backend_name)
{
    auto frame = makeFrame(MSG_SERVER_STATUS, sizeof(ServerStatusPayload));
    ServerStatusPayload payload{};
    payload.active_workers = active_workers;
    payload.queue_depth = queue_depth;
    payload.avg_processing_ms = avg_processing_ms;
    copyTextField(payload.model_hash, sizeof(payload.model_hash), model_hash);
    copyTextField(payload.backend_name, sizeof(payload.backend_name), backend_name);
    std::memcpy(frame.data() + sizeof(FrameHeader), &payload, sizeof(payload));
    return frame;
}

std::vector<uint8_t> TcpProtocolSerializer::serializeShutdown(uint32_t drain_timeout_ms)
{
    auto frame = makeFrame(MSG_SHUTDOWN, sizeof(ShutdownPayload));
    ShutdownPayload payload{};
    payload.drain_timeout_ms = drain_timeout_ms;
    std::memcpy(frame.data() + sizeof(FrameHeader), &payload, sizeof(payload));
    return frame;
}

std::vector<uint8_t> TcpProtocolSerializer::serializeShutdownAck()
{
    return makeFrame(MSG_SHUTDOWN_ACK, 0);
}

std::vector<uint8_t> TcpProtocolSerializer::serializeError(uint32_t error_code, const char* error_msg)
{
    auto frame = makeFrame(MSG_ERROR, sizeof(ErrorPayload));
    ErrorPayload payload{};
    payload.error_code = error_code;
    copyTextField(payload.error_msg, sizeof(payload.error_msg), error_msg);
    std::memcpy(frame.data() + sizeof(FrameHeader), &payload, sizeof(payload));
    return frame;
}

std::vector<uint8_t> TcpProtocolSerializer::serializeSplatRequest(uint32_t n_views,
                                                                  uint32_t width,
                                                                  uint32_t height,
                                                                  const SplatRequestOptions& options,
                                                                  const uint8_t* const* image_ptrs,
                                                                  const uint32_t* image_sizes,
                                                                  uint32_t image_count)
{
    if(image_count != n_views || n_views < 2 || n_views > 4)
    {
        LOGE("TcpProtocolSerializer: invalid n_views (%u, image_count=%u)\n", n_views, image_count);
        return {};
    }
    if(image_ptrs == nullptr || image_sizes == nullptr)
    {
        LOGE("TcpProtocolSerializer: null image_ptrs/image_sizes for splat request\n");
        return {};
    }

    size_t image_data_size = 0;
    for(uint32_t i = 0; i < image_count; ++i)
    {
        if(image_sizes[i] > MAX_IMAGE_SIZE)
        {
            LOGE("TcpProtocolSerializer: image %u exceeds MAX_IMAGE_SIZE (%u > %u)\n",
                 i, image_sizes[i], MAX_IMAGE_SIZE);
            return {};
        }
        size_t with_prefix = 0;
        if(!checkedAdd(sizeof(uint32_t), image_sizes[i], with_prefix))
        {
            LOGE("TcpProtocolSerializer: image %u size overflow\n", i);
            return {};
        }
        if(!checkedAdd(image_data_size, with_prefix, image_data_size))
        {
            LOGE("TcpProtocolSerializer: total image_data_size overflow\n");
            return {};
        }
    }

    size_t payload_size = 0;
    if(!checkedAdd(sizeof(SplatRequestPayload), sizeof(SplatRequestOptions), payload_size) ||
       !checkedAdd(payload_size, image_data_size, payload_size))
    {
        LOGE("TcpProtocolSerializer: splat request payload overflow\n");
        return {};
    }

    auto frame = makeFrame(MSG_SPLAT_REQUEST, payload_size);
    if(frame.empty())
    {
        return {};
    }

    SplatRequestPayload payload{};
    payload.n_views = n_views;
    payload.width = width;
    payload.height = height;
    payload.options_size = sizeof(SplatRequestOptions);
    payload.image_data_size = static_cast<uint32_t>(image_data_size);

    uint8_t* write_ptr = frame.data() + sizeof(FrameHeader);
    std::memcpy(write_ptr, &payload, sizeof(payload));
    write_ptr += sizeof(payload);

    std::memcpy(write_ptr, &options, sizeof(SplatRequestOptions));
    write_ptr += sizeof(SplatRequestOptions);

    for(uint32_t i = 0; i < image_count; ++i)
    {
        uint32_t len = image_sizes[i];
        std::memcpy(write_ptr, &len, sizeof(len));
        write_ptr += sizeof(len);
        if(len != 0)
        {
            std::memcpy(write_ptr, image_ptrs[i], len);
            write_ptr += len;
        }
    }

    return frame;
}

std::vector<uint8_t> TcpProtocolSerializer::serializeSplatPoll(uint32_t job_id)
{
    auto frame = makeFrame(MSG_SPLAT_POLL, sizeof(SplatPollPayload));
    SplatPollPayload payload{};
    payload.job_id = job_id;
    std::memcpy(frame.data() + sizeof(FrameHeader), &payload, sizeof(payload));
    return frame;
}

std::vector<uint8_t> TcpProtocolSerializer::serializeSplatProgress(uint32_t job_id,
                                                                   uint8_t stage,
                                                                   uint8_t percent,
                                                                   uint32_t eta_ms)
{
    auto frame = makeFrame(MSG_SPLAT_PROGRESS, sizeof(SplatProgressPayload));
    SplatProgressPayload payload{};
    payload.job_id = job_id;
    payload.stage = stage;
    payload.percent = percent;
    payload.eta_ms = eta_ms;
    std::memcpy(frame.data() + sizeof(FrameHeader), &payload, sizeof(payload));
    return frame;
}

std::vector<uint8_t> TcpProtocolSerializer::serializeSplatResponse(uint32_t job_id,
                                                                    uint32_t n_gaussians,
                                                                    const char* path)
{
    if(path == nullptr)
    {
        LOGE("TcpProtocolSerializer: null path for splat response\n");
        return {};
    }

    const size_t path_len = std::strlen(path) + 1;
    if(path_len > MAX_FRAME_SIZE)
    {
        LOGE("TcpProtocolSerializer: splat response path too long (%zu)\n", path_len);
        return {};
    }

    size_t payload_size = 0;
    if(!checkedAdd(sizeof(SplatResponsePayload), path_len, payload_size))
    {
        LOGE("TcpProtocolSerializer: splat response payload overflow\n");
        return {};
    }

    auto frame = makeFrame(MSG_SPLAT_RESPONSE, payload_size);
    SplatResponsePayload payload{};
    payload.job_id = job_id;
    payload.n_gaussians = n_gaussians;
    payload.path_length = static_cast<uint32_t>(path_len);
    std::memcpy(frame.data() + sizeof(FrameHeader), &payload, sizeof(payload));
    std::memcpy(frame.data() + sizeof(FrameHeader) + sizeof(SplatResponsePayload), path, path_len);
    return frame;
}

std::vector<uint8_t> TcpProtocolSerializer::serializeSplatCancel(uint32_t job_id)
{
    auto frame = makeFrame(MSG_SPLAT_CANCEL, sizeof(SplatCancelPayload));
    SplatCancelPayload payload{};
    payload.job_id = job_id;
    std::memcpy(frame.data() + sizeof(FrameHeader), &payload, sizeof(payload));
    return frame;
}

std::vector<uint8_t> TcpProtocolSerializer::serializeSplatError(uint32_t job_id,
                                                                uint32_t error_code,
                                                                const char* error_msg)
{
    auto frame = makeFrame(MSG_SPLAT_ERROR, sizeof(SplatErrorPayload));
    SplatErrorPayload payload{};
    payload.job_id = job_id;
    payload.error_code = error_code;
    copyTextField(payload.error_msg, sizeof(payload.error_msg), error_msg);
    std::memcpy(frame.data() + sizeof(FrameHeader), &payload, sizeof(payload));
    return frame;
}

bool decodeSplatRequest(const uint8_t* message,
                        size_t message_len,
                        SplatRequestPayload& payload,
                        const uint8_t*& options,
                        const uint8_t*& images,
                        size_t& images_size)
{
    options = nullptr;
    images = nullptr;
    images_size = 0;

    if(message == nullptr || message_len < sizeof(FrameHeader) + sizeof(SplatRequestPayload) + sizeof(SplatRequestOptions))
    {
        return false;
    }

    std::memcpy(&payload, message + sizeof(FrameHeader), sizeof(payload));

    if(payload.n_views < 2 || payload.n_views > 4)
    {
        return false;
    }
    if(payload.options_size != sizeof(SplatRequestOptions))
    {
        return false;
    }

    const size_t header_and_payload = sizeof(FrameHeader) + sizeof(SplatRequestPayload);
    const size_t expected_size = header_and_payload + sizeof(SplatRequestOptions) + payload.image_data_size;
    if(expected_size != message_len)
    {
        return false;
    }

    options = message + header_and_payload;
    images = message + header_and_payload + sizeof(SplatRequestOptions);
    images_size = payload.image_data_size;

    size_t remaining = images_size;
    const uint8_t* cursor = images;
    for(uint32_t i = 0; i < payload.n_views; ++i)
    {
        if(remaining < sizeof(uint32_t))
        {
            return false;
        }
        uint32_t len = 0;
        std::memcpy(&len, cursor, sizeof(len));
        remaining -= sizeof(uint32_t);
        cursor += sizeof(uint32_t);

        if(len > MAX_IMAGE_SIZE || remaining < len)
        {
            return false;
        }
        remaining -= len;
        cursor += len;
    }

    return true;
}
