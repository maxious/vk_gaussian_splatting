#include "cloud_splat_encoder.h"

#include <algorithm>
#include <cstring>

namespace {
// Depth-anything returns radius values that are ~100x too small for the
// Antimatter15 .splat format (which stores linear world-space gaussian scales).
// Without this factor the cloud points render as invisible specks.
constexpr float kCloudSplatRadiusScale = 100.0f;
}  // namespace

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

        const float scaled_radius = radius[i] * kCloudSplatRadiusScale;
        std::memcpy(record + 12, &scaled_radius, sizeof(float));
        std::memcpy(record + 16, &scaled_radius, sizeof(float));
        std::memcpy(record + 20, &scaled_radius, sizeof(float));

        record[24] = rgb[3 * i + 0];
        record[25] = rgb[3 * i + 1];
        record[26] = rgb[3 * i + 2];
        record[27] = 255;

        // Antimatter15 .splat rotation quaternion (w,x,y,z), 128-centered
        // Identity rotation: w=1.0 → 255, x=y=z=0.0 → 128
        record[28] = 255;
        record[29] = 128;
        record[30] = 128;
        record[31] = 128;
    }
}
