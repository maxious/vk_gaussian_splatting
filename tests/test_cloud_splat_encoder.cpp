// Copyright (c) 2025, vk_gaussian_splatting contributors
// Licensed under Apache 2.0 - see LICENSE for details

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "cloud_splat_encoder.h"

#include <array>
#include <cfloat>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <vector>

static float read_le_f32(const uint8_t* buf, size_t off)
{
    float v;
    std::memcpy(&v, buf + off, sizeof(v));
    return v;
}

static uint8_t read_u8(const uint8_t* buf, size_t off)
{
    return buf[off];
}

TEST_CASE("cloud_splat_encoder empty input")
{
    std::vector<uint8_t> out;
    cloud_splat_encoder_encode(nullptr, nullptr, nullptr, 0, out);
    CHECK(out.empty());
}

TEST_CASE("cloud_splat_encoder single point")
{
    std::array<float, 3>     xyz    = {1.0f, 2.0f, 3.0f};
    std::array<uint8_t, 3>   rgb    = {128, 64, 32};
    std::array<float, 1>     radius = {0.75f};

    std::vector<uint8_t> out;
    cloud_splat_encoder_encode(xyz.data(), rgb.data(), radius.data(), 1, out);

    REQUIRE(out.size() == 32);

    CHECK(read_le_f32(out.data(), 0) == doctest::Approx(1.0f));
    CHECK(read_le_f32(out.data(), 4) == doctest::Approx(2.0f));
    CHECK(read_le_f32(out.data(), 8) == doctest::Approx(3.0f));

    CHECK(read_le_f32(out.data(), 12) == doctest::Approx(75.0f));
    CHECK(read_le_f32(out.data(), 16) == doctest::Approx(75.0f));
    CHECK(read_le_f32(out.data(), 20) == doctest::Approx(75.0f));

    CHECK(read_u8(out.data(), 24) == 128);
    CHECK(read_u8(out.data(), 25) == 64);
    CHECK(read_u8(out.data(), 26) == 32);

    CHECK(read_u8(out.data(), 27) == 255);

    CHECK(read_u8(out.data(), 28) == 255);
    CHECK(read_u8(out.data(), 29) == 128);
    CHECK(read_u8(out.data(), 30) == 128);
    CHECK(read_u8(out.data(), 31) == 128);
}

TEST_CASE("cloud_splat_encoder large_n")
{
    constexpr size_t N = 10000;
    std::vector<float>     xyz(N * 3, 1.0f);
    std::vector<uint8_t>   rgb(N * 3, 128);
    std::vector<float>     radius(N, 0.5f);

    std::vector<uint8_t> out;
    cloud_splat_encoder_encode(xyz.data(), rgb.data(), radius.data(), N, out);

    REQUIRE(out.size() == N * 32);
    CHECK(read_le_f32(out.data(), 0) == doctest::Approx(1.0f));
    CHECK(read_u8(out.data(), 24) == 128);
    CHECK(read_u8(out.data(), 27) == 255);

    size_t last = (N - 1) * 32;
    CHECK(read_le_f32(out.data(), last) == doctest::Approx(1.0f));
    CHECK(read_u8(out.data(), last + 27) == 255);
}

TEST_CASE("cloud_splat_encoder axis_boundary_values")
{
    // clang-format off
    std::array<float, 3 * 5> xyz = {
        FLT_MAX,  0.0f,     0.0f,
       -FLT_MAX,  0.0f,     0.0f,
        0.0f,     1.0f,    -1.0f,
        1.5f,    -2.5f,     3.5f,
        0.0f,     0.0f,     0.0f,
    };
    // clang-format on
    std::array<uint8_t, 3 * 5> rgb = {};
    std::array<float, 5>       radius = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

    std::vector<uint8_t> out;
    cloud_splat_encoder_encode(xyz.data(), rgb.data(), radius.data(), 5, out);

    REQUIRE(out.size() == 5 * 32);

    CHECK(read_le_f32(out.data(), 0) == doctest::Approx(FLT_MAX));
    CHECK(read_le_f32(out.data(), 4) == doctest::Approx(0.0f));
    CHECK(read_le_f32(out.data(), 8) == doctest::Approx(0.0f));

    CHECK(read_le_f32(out.data(), 32) == doctest::Approx(-FLT_MAX));
    CHECK(read_le_f32(out.data(), 36) == doctest::Approx(0.0f));

    CHECK(read_le_f32(out.data(), 64) == doctest::Approx(0.0f));
    CHECK(read_le_f32(out.data(), 68) == doctest::Approx(1.0f));
    CHECK(read_le_f32(out.data(), 72) == doctest::Approx(-1.0f));

    CHECK(read_le_f32(out.data(), 96) == doctest::Approx(1.5f));
    CHECK(read_le_f32(out.data(), 100) == doctest::Approx(-2.5f));
    CHECK(read_le_f32(out.data(), 104) == doctest::Approx(3.5f));

    CHECK(read_le_f32(out.data(), 128) == doctest::Approx(0.0f));
    CHECK(read_le_f32(out.data(), 132) == doctest::Approx(0.0f));
    CHECK(read_le_f32(out.data(), 136) == doctest::Approx(0.0f));
}

TEST_CASE("cloud_splat_encoder NaN_passthrough")
{
    float nan_val = std::numeric_limits<float>::quiet_NaN();

    std::array<float, 3>     xyz    = {nan_val, 0.0f, 0.0f};
    std::array<uint8_t, 3>   rgb    = {0, 0, 0};
    std::array<float, 1>     radius = {1.0f};

    std::vector<uint8_t> out;
    cloud_splat_encoder_encode(xyz.data(), rgb.data(), radius.data(), 1, out);

    REQUIRE(out.size() == 32);

    // Verify NaN bytes preserved as-is: IE QNaN in LE is 0x7FC00000
    uint32_t raw_bits;
    std::memcpy(&raw_bits, out.data(), sizeof(raw_bits));
    CHECK(raw_bits == 0x7FC00000u);
}

TEST_CASE("cloud_splat_encoder golden_comparison")
{
    const std::string golden_path = TEST_FIXTURE_DIR
                                    "/cloud_splat_encoder_golden.bin";

    std::ifstream f(golden_path, std::ios::binary | std::ios::ate);
    REQUIRE(f.good());
    std::streamsize golden_size = f.tellg();
    REQUIRE(golden_size == 4 * 32);
    f.seekg(0);

    std::vector<uint8_t> golden(golden_size);
    f.read(reinterpret_cast<char*>(golden.data()), golden_size);
    REQUIRE(f.good());

    // clang-format off
    std::array<float, 12> xyz = {
        1.0f,   2.0f,   3.0f,
       -1.0f,  -2.0f,  -3.0f,
        0.0f,   0.0f,   0.0f,
      100.0f, -100.0f,  0.5f
    };
    std::array<uint8_t, 12> rgb = {
        255,   0,   0,
          0, 255,   0,
          0,   0, 255,
        128, 128, 128
    };
    // clang-format on
    std::array<float, 4> radius = {1.0f, 2.0f, 0.5f, 1.5f};

    std::vector<uint8_t> out;
    cloud_splat_encoder_encode(xyz.data(), rgb.data(), radius.data(), 4, out);

    REQUIRE(out.size() == golden_size);
    CHECK(std::memcmp(out.data(), golden.data(), golden_size) == 0);
}

TEST_CASE("cloud_splat_encoder RGB_clamping")
{
    std::array<float, 3>     xyz    = {0, 0, 0};
    std::array<uint8_t, 3>   rgb    = {0, 255, 128};
    std::array<float, 1>     radius = {1.0f};

    std::vector<uint8_t> out;
    cloud_splat_encoder_encode(xyz.data(), rgb.data(), radius.data(), 1, out);

    REQUIRE(out.size() == 32);
    CHECK(read_u8(out.data(), 24) == 0);
    CHECK(read_u8(out.data(), 25) == 255);
    CHECK(read_u8(out.data(), 26) == 128);
}

TEST_CASE("cloud_splat_encoder alpha_always_255")
{
    std::array<float, 6>     xyz    = {0, 0, 0, 1, 1, 1};
    std::array<uint8_t, 6>   rgb    = {255, 0, 0, 0, 255, 0};
    std::array<float, 2>     radius = {1, 1};

    std::vector<uint8_t> out;
    cloud_splat_encoder_encode(xyz.data(), rgb.data(), radius.data(), 2, out);

    REQUIRE(out.size() == 64);
    CHECK(read_u8(out.data(), 27) == 255);
    CHECK(read_u8(out.data(), 59) == 255);
}
