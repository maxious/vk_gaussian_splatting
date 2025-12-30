#include "depth_parser.h"
#include <nvutils/logger.hpp>
#include <cstring>
#include <zlib.h>

bool parseDepthFrame(const std::vector<uint8_t>& buffer, DepthFrame& outFrame) {
    const size_t HEADER_SIZE = 32;
    if (buffer.size() < HEADER_SIZE) {
        LOGW("Buffer too small for depth frame\n");
        return false;
    }

    

    DepthHeader header;
    std::memcpy(&header, buffer.data(), HEADER_SIZE);

    bool isCompressed = false;
    if (std::memcmp(header.magic, "VDZ2", 4) == 0) {
        isCompressed = true;
    } else if (std::memcmp(header.magic, "VDZ1", 4) != 0) {
        LOGW("Invalid magic bytes in depth frame\n");
        return false;
    }

    if (header.version != 1 || header.dataType != 1) {
        LOGW("Unsupported depth frame version or data type\n");
        return false;
    }

    std::vector<uint16_t> samples;
    
    if (isCompressed) {
        uLongf decompressedSize = header.width * header.height * sizeof(uint16_t);
        samples.resize(header.width * header.height);

        z_stream stream{};
        stream.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(buffer.data() + HEADER_SIZE));
        stream.avail_in = static_cast<uInt>(buffer.size() - HEADER_SIZE);
        stream.next_out = reinterpret_cast<Bytef*>(samples.data());
        stream.avail_out = decompressedSize;

        if (inflateInit2(&stream, -MAX_WBITS) != Z_OK) {
             if (inflateInit(&stream) != Z_OK) {
                LOGE("Failed to initialize zlib\n");
                return false;
             }
        }

        int ret = inflate(&stream, Z_FINISH);
        inflateEnd(&stream);

        if (ret != Z_STREAM_END) {
             LOGE("Decompression failed with code: %d\n", ret);
             return false;
        }
    } else {
        const size_t dataSize = buffer.size() - HEADER_SIZE;
        const uint16_t* rawData = reinterpret_cast<const uint16_t*>(buffer.data() + HEADER_SIZE);
        samples.assign(rawData, rawData + dataSize / sizeof(uint16_t));
    }

    // Validate scale is reasonable (allow 0.0 for initial frames, but not negative)
    if (header.scale < 0.0f || header.scale > 10.0f) {
        LOGW("Invalid scale %.8f in depth frame - rejecting\n", header.scale);
        return false;
    }
    
    // Sanity check for timestamp - video timestamps should be reasonable (0 to 1 hour)
    if (header.timestampMs > 3600000) {
        LOGW("Suspicious timestamp %u ms in depth frame - rejecting\n", header.timestampMs);
        return false;
    }

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
