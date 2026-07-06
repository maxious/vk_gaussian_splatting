#include "cloud_splat_encoder.h"

#include <cstring>

void cloud_splat_encoder_encode(const float* xyz, const uint8_t* rgb,
                                const float* radius, size_t n,
                                std::vector<uint8_t>& out_bytes)
{
    if (n == 0)
    {
        out_bytes.clear();
        return;
    }

    out_bytes.resize(n * kCloudSplatBytesPerRecord);

    for (size_t i = 0; i < n; ++i)
    {
        uint8_t* record = out_bytes.data() + i * kCloudSplatBytesPerRecord;

        std::memcpy(record + 0,  xyz + 3 * i + 0, sizeof(float));
        std::memcpy(record + 4,  xyz + 3 * i + 1, sizeof(float));
        std::memcpy(record + 8,  xyz + 3 * i + 2, sizeof(float));

        std::memcpy(record + 12, radius + i, sizeof(float));
        std::memcpy(record + 16, radius + i, sizeof(float));
        std::memcpy(record + 20, radius + i, sizeof(float));

        record[24] = rgb[3 * i + 0];
        record[25] = rgb[3 * i + 1];
        record[26] = rgb[3 * i + 2];
        record[27] = 255;

        record[28] = 128;
        record[29] = 255;
        record[30] = 128;
        record[31] = 128;
    }
}
