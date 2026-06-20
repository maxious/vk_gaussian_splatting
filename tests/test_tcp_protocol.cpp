// Copyright (c) 2025, vk_gaussian_splatting contributors
// Licensed under Apache 2.0 - see LICENSE for details

#include "doctest.h"

#include "../depth_server/protocol.h"
#include "../depth_server/protocol_parser.h"

#include <array>
#include <cstring>
#include <string>
#include <vector>

TEST_CASE("tcp_protocol FrameHeader serialize/deserialize round-trip")
{
    FrameHeader header{};
    header.magic = TCP_DEPTH_MAGIC;
    header.frame_length = sizeof(FrameHeader) + sizeof(DepthResponsePayload) + 640u * 480u * sizeof(float);
    header.message_type = MSG_DEPTH_RESPONSE;

    std::array<uint8_t, sizeof(FrameHeader)> buffer{};
    std::memcpy(buffer.data(), &header, sizeof(FrameHeader));

    FrameHeader decoded{};
    std::memcpy(&decoded, buffer.data(), sizeof(FrameHeader));

    CHECK(decoded.magic == TCP_DEPTH_MAGIC);
    CHECK(decoded.frame_length == header.frame_length);
    CHECK(decoded.message_type == MSG_DEPTH_RESPONSE);
}

TEST_CASE("tcp_protocol Magic byte validation")
{
    FrameHeader header{};
    header.magic = TCP_DEPTH_MAGIC;
    header.frame_length = sizeof(FrameHeader);
    header.message_type = MSG_FRAME_REQUEST;
    CHECK(validateHeader(&header, sizeof(header)));

    header.magic = 0xDEADBEEF;
    CHECK_FALSE(validateHeader(&header, sizeof(header)));

    header.magic = TCP_DEPTH_MAGIC;
    header.frame_length = 0;
    CHECK_FALSE(validateHeader(&header, sizeof(header)));
}

TEST_CASE("tcp_protocol Fragmented TCP receive reassembles 3 segments")
{
    std::array<uint8_t, 12> rgb{};
    for(size_t i = 0; i < rgb.size(); ++i) {
        rgb[i] = static_cast<uint8_t>(i + 1);
    }

    const auto message = TcpProtocolSerializer::serializeFrameRequest(7, 1234, 2, 2, rgb.data());
    REQUIRE(message.size() > sizeof(FrameHeader));

    TcpProtocolParser parser;
    const size_t a = 5;
    const size_t b = 11;

    CHECK(parser.feed(message.data(), a) == 0);
    CHECK_FALSE(parser.hasMessage());
    CHECK(parser.feed(message.data() + a, b - a) == 0);
    CHECK_FALSE(parser.hasMessage());
    CHECK(parser.feed(message.data() + b, message.size() - b) == 1);
    CHECK(parser.hasMessage());

    size_t out_len = 0;
    uint32_t out_type = 0;
    const uint8_t* decoded = parser.nextMessage(out_len, out_type);

    REQUIRE(decoded != nullptr);
    CHECK(out_len == message.size());
    CHECK(out_type == MSG_FRAME_REQUEST);
    CHECK(std::memcmp(decoded, message.data(), message.size()) == 0);
}

TEST_CASE("tcp_protocol Oversized frame rejection")
{
    FrameHeader header{};
    header.magic = TCP_DEPTH_MAGIC;
    header.frame_length = MAX_FRAME_SIZE + 1u;
    header.message_type = MSG_DEPTH_RESPONSE;
    CHECK_FALSE(validateHeader(&header, sizeof(header)));
}

TEST_CASE("tcp_protocol Malformed header rejects zero length and unknown type")
{
    FrameHeader header{};
    header.magic = TCP_DEPTH_MAGIC;
    header.frame_length = sizeof(FrameHeader);
    header.message_type = 0x06u;
    CHECK_FALSE(validateHeader(&header, sizeof(header)));

    header.message_type = MSG_FRAME_REQUEST;
    header.frame_length = 0;
    CHECK_FALSE(validateHeader(&header, sizeof(header)));
}

TEST_CASE("tcp_protocol FRAME_REQUEST serialization round-trip")
{
    const std::array<uint8_t, 18> rgb = {1, 2, 3, 4, 5, 6, 7, 8, 9,
                                         10, 11, 12, 13, 14, 15, 16, 17, 18};
    const auto frame = TcpProtocolSerializer::serializeFrameRequest(42, 777, 3, 2, rgb.data());

    REQUIRE(frame.size() == sizeof(FrameHeader) + sizeof(FrameRequestPayload) + rgb.size());

    FrameHeader header{};
    std::memcpy(&header, frame.data(), sizeof(header));
    CHECK(header.magic == TCP_DEPTH_MAGIC);
    CHECK(header.message_type == MSG_FRAME_REQUEST);
    CHECK(header.frame_length == frame.size());

    FrameRequestPayload payload{};
    std::memcpy(&payload, frame.data() + sizeof(FrameHeader), sizeof(payload));
    CHECK(payload.width == 3);
    CHECK(payload.height == 2);
    CHECK(payload.timestamp_ms == 777);
    CHECK(std::memcmp(frame.data() + sizeof(FrameHeader) + sizeof(FrameRequestPayload), rgb.data(), rgb.size()) == 0);
}

TEST_CASE("tcp_protocol DEPTH_RESPONSE serialization round-trip")
{
    const std::array<float, 6> depth = {0.5f, 1.25f, 2.5f, 3.75f, 4.0f, 5.5f};
    const auto frame = TcpProtocolSerializer::serializeDepthResponse(9, 2468, 3, 2, 0.000001f, 0.646956f, 0.734110f, depth.data());

    REQUIRE(frame.size() == sizeof(FrameHeader) + sizeof(DepthResponsePayload) + depth.size() * sizeof(float));

    DepthResponsePayload payload{};
    std::memcpy(&payload, frame.data() + sizeof(FrameHeader), sizeof(payload));
    CHECK(payload.width == 3);
    CHECK(payload.height == 2);
    CHECK(payload.timestamp_ms == 2468);
    CHECK(payload.scale == doctest::Approx(0.000001f));
    CHECK(payload.bias == doctest::Approx(0.646956f));
    CHECK(payload.z_max == doctest::Approx(0.734110f));
    CHECK(std::memcmp(frame.data() + sizeof(FrameHeader) + sizeof(DepthResponsePayload), depth.data(), depth.size() * sizeof(float)) == 0);
}

TEST_CASE("tcp_protocol SERVER_STATUS payload round-trip")
{
    const auto frame = TcpProtocolSerializer::serializeServerStatus(4, 12, 8.5f, "0123456789abcdef", "cuda");

    REQUIRE(frame.size() == sizeof(FrameHeader) + sizeof(ServerStatusPayload));

    ServerStatusPayload payload{};
    std::memcpy(&payload, frame.data() + sizeof(FrameHeader), sizeof(payload));
    CHECK(payload.active_workers == 4);
    CHECK(payload.queue_depth == 12);
    CHECK(payload.avg_processing_ms == doctest::Approx(8.5f));
    CHECK(std::string(payload.model_hash) == "0123456789abcdef");
    CHECK(std::string(payload.backend_name) == "cuda");
}

TEST_CASE("tcp_protocol ERROR message round-trip")
{
    const auto frame = TcpProtocolSerializer::serializeError(404, "not found");

    REQUIRE(frame.size() == sizeof(FrameHeader) + sizeof(ErrorPayload));

    ErrorPayload payload{};
    std::memcpy(&payload, frame.data() + sizeof(FrameHeader), sizeof(payload));
    CHECK(payload.error_code == 404);
    CHECK(std::string(payload.error_msg).find("not found") == 0);
}

TEST_CASE("tcp_protocol SHUTDOWN integrity")
{
    SUBCASE("SHUTDOWN")
    {
        const auto frame = TcpProtocolSerializer::serializeShutdown(1500);
        REQUIRE(frame.size() == sizeof(FrameHeader) + sizeof(ShutdownPayload));

        FrameHeader header{};
        std::memcpy(&header, frame.data(), sizeof(header));
        CHECK(header.message_type == MSG_SHUTDOWN);

        ShutdownPayload payload{};
        std::memcpy(&payload, frame.data() + sizeof(FrameHeader), sizeof(payload));
        CHECK(payload.drain_timeout_ms == 1500);
    }

    SUBCASE("SHUTDOWN_ACK")
    {
        const auto frame = TcpProtocolSerializer::serializeShutdownAck();
        REQUIRE(frame.size() == sizeof(FrameHeader));

        FrameHeader header{};
        std::memcpy(&header, frame.data(), sizeof(header));
        CHECK(header.message_type == MSG_SHUTDOWN_ACK);
        CHECK(header.frame_length == sizeof(FrameHeader));
    }
}
