#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

constexpr size_t kCloudSplatBytesPerRecord = 32;

void cloud_splat_encoder_encode(const float* xyz, const uint8_t* rgb,
                                const float* radius, size_t n,
                                std::vector<uint8_t>& out_bytes);
