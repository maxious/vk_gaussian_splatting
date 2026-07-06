// Copyright (c) 2025, vk_gaussian_splatting contributors
// Licensed under Apache 2.0 - see LICENSE for details

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "../depth_server/protocol.h"
#include "../depth_server/protocol_parser.h"

#include <array>
#include <cstring>
#include <string>
#include <vector>

TEST_CASE("cloud_protocol FrameHeader round-trip with MSG_CLOUD_REQUEST")
{
    FrameHeader header{};
    header.magic           = TCP_DEPTH_MAGIC;
    header.frame_length    = sizeof(FrameHeader) + sizeof(CloudRequestPayload) + sizeof(CloudRequestOptions);
    header.message_type    = MSG_CLOUD_REQUEST;

    std::array<uint8_t, sizeof(FrameHeader)> buffer{};
    std::memcpy(buffer.data(), &header, sizeof(FrameHeader));

    FrameHeader decoded{};
    std::memcpy(&decoded, buffer.data(), sizeof(FrameHeader));

    CHECK(decoded.magic == TCP_DEPTH_MAGIC);
    CHECK(decoded.frame_length == header.frame_length);
    CHECK(decoded.message_type == MSG_CLOUD_REQUEST);
    CHECK(validateHeader(&decoded, sizeof(FrameHeader)));
}

TEST_CASE("cloud_protocol CloudRequestOptions round-trip byte-exact")
{
    CloudRequestOptions opts{};
    opts.chunk_size      = 8;
    opts.overlap         = 4;
    opts.conf_pct        = 95.0f;
    opts.point_size      = 0.01f;
    opts.global_budget   = 0;
    opts.icp_refine      = 1;
    opts.loop_close      = 0;
    opts.fuse            = 1;
    opts.metric          = 1;
    opts.fuse_voxel_frac = 0.004f;
    opts.fuse_trunc_mult = 4.0f;
    opts.timeout_ms      = 60000;
    opts.flags           = 1;

    std::array<uint8_t, sizeof(CloudRequestOptions)> buffer{};
    std::memcpy(buffer.data(), &opts, sizeof(CloudRequestOptions));

    CloudRequestOptions decoded{};
    std::memcpy(&decoded, buffer.data(), sizeof(CloudRequestOptions));

    CHECK(decoded.chunk_size      == 8);
    CHECK(decoded.overlap         == 4);
    CHECK(decoded.conf_pct        == doctest::Approx(95.0f));
    CHECK(decoded.point_size      == doctest::Approx(0.01f));
    CHECK(decoded.global_budget   == 0);
    CHECK(decoded.icp_refine      == 1);
    CHECK(decoded.loop_close      == 0);
    CHECK(decoded.fuse            == 1);
    CHECK(decoded.metric          == 1);
    CHECK(decoded.fuse_voxel_frac == doctest::Approx(0.004f));
    CHECK(decoded.fuse_trunc_mult == doctest::Approx(4.0f));
    CHECK(decoded.timeout_ms      == 60000);
    CHECK(decoded.flags           == 1);

    CHECK(std::memcmp(&opts, &decoded, sizeof(CloudRequestOptions)) == 0);
}

TEST_CASE("cloud_protocol CloudRequestPayload + options + frame paths round-trip")
{
    const std::vector<std::string> frame_paths = {
        "/tmp/frame_01.jpg",
        "/tmp/frame_02.jpg",
        "/tmp/frame_03.jpg"
    };

    uint32_t frame_data_size = 0;
    for(const auto& p : frame_paths)
        frame_data_size += static_cast<uint32_t>(sizeof(uint32_t) + p.size());

    CloudRequestPayload payload{};
    payload.n_frames       = static_cast<uint32_t>(frame_paths.size());
    payload.options_size   = sizeof(CloudRequestOptions);
    payload.frame_data_size = frame_data_size;

    CloudRequestOptions opts{};
    opts.chunk_size      = 3;
    opts.overlap         = 1;
    opts.conf_pct        = 90.0f;
    opts.point_size      = 0.005f;
    opts.global_budget   = 5000000;
    opts.icp_refine      = 1;
    opts.loop_close      = 1;
    opts.fuse            = 1;
    opts.metric          = 1;
    opts.fuse_voxel_frac = 0.004f;
    opts.fuse_trunc_mult = 4.0f;
    opts.timeout_ms      = 120000;
    opts.flags           = 1;

    const uint32_t total_size = sizeof(FrameHeader)
                              + sizeof(CloudRequestPayload)
                              + sizeof(CloudRequestOptions)
                              + frame_data_size;
    std::vector<uint8_t> buffer(total_size);

    FrameHeader header{};
    header.magic        = TCP_DEPTH_MAGIC;
    header.frame_length = total_size;
    header.message_type = MSG_CLOUD_REQUEST;

    size_t offset = 0;
    std::memcpy(buffer.data() + offset, &header, sizeof(FrameHeader));
    offset += sizeof(FrameHeader);
    std::memcpy(buffer.data() + offset, &payload, sizeof(CloudRequestPayload));
    offset += sizeof(CloudRequestPayload);
    std::memcpy(buffer.data() + offset, &opts, sizeof(CloudRequestOptions));
    offset += sizeof(CloudRequestOptions);

    for(const auto& path : frame_paths)
    {
        uint32_t len = static_cast<uint32_t>(path.size());
        std::memcpy(buffer.data() + offset, &len, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        std::memcpy(buffer.data() + offset, path.data(), path.size());
        offset += path.size();
    }

    REQUIRE(offset == total_size);

    FrameHeader dh{};
    std::memcpy(&dh, buffer.data(), sizeof(FrameHeader));
    CHECK(dh.magic        == TCP_DEPTH_MAGIC);
    CHECK(dh.frame_length == total_size);
    CHECK(dh.message_type == MSG_CLOUD_REQUEST);
    CHECK(validateHeader(&dh, buffer.size()));

    CloudRequestPayload dp{};
    std::memcpy(&dp, buffer.data() + sizeof(FrameHeader), sizeof(CloudRequestPayload));
    CHECK(dp.n_frames       == 3);
    CHECK(dp.options_size   == sizeof(CloudRequestOptions));
    CHECK(dp.frame_data_size == frame_data_size);

    CloudRequestOptions dopts{};
    std::memcpy(&dopts, buffer.data() + sizeof(FrameHeader) + sizeof(CloudRequestPayload),
                sizeof(CloudRequestOptions));
    CHECK(dopts.chunk_size == 3);

    size_t ro = sizeof(FrameHeader) + sizeof(CloudRequestPayload) + sizeof(CloudRequestOptions);
    for(size_t i = 0; i < frame_paths.size(); ++i)
    {
        uint32_t len = 0;
        std::memcpy(&len, buffer.data() + ro, sizeof(uint32_t));
        ro += sizeof(uint32_t);
        std::string path(reinterpret_cast<const char*>(buffer.data() + ro), len);
        ro += len;
        CHECK(path == frame_paths[i]);
    }
}

TEST_CASE("cloud_protocol CloudPollPayload round-trip")
{
    CloudPollPayload payload{};
    payload.job_id = 99;

    std::array<uint8_t, sizeof(CloudPollPayload)> buffer{};
    std::memcpy(buffer.data(), &payload, sizeof(CloudPollPayload));

    CloudPollPayload decoded{};
    std::memcpy(&decoded, buffer.data(), sizeof(CloudPollPayload));
    CHECK(decoded.job_id == 99);
}

TEST_CASE("cloud_protocol CloudProgressPayload round-trip")
{
    CloudProgressPayload payload{};
    payload.job_id        = 42;
    payload.stage         = 2;
    payload.percent       = 65;
    payload.windows_done  = 13;
    payload.windows_total = 20;
    payload._pad          = 0;
    payload.eta_ms        = 45000;

    std::array<uint8_t, sizeof(CloudProgressPayload)> buffer{};
    std::memcpy(buffer.data(), &payload, sizeof(CloudProgressPayload));

    CloudProgressPayload decoded{};
    std::memcpy(&decoded, buffer.data(), sizeof(CloudProgressPayload));

    CHECK(decoded.job_id        == 42);
    CHECK(decoded.stage         == 2);
    CHECK(decoded.percent       == 65);
    CHECK(decoded.windows_done  == 13);
    CHECK(decoded.windows_total == 20);
    CHECK(decoded.eta_ms        == 45000);
}

TEST_CASE("cloud_protocol CloudResponsePayload round-trip")
{
    const std::string response_path = "/tmp/cloud_output.ply";

    CloudResponsePayload payload{};
    payload.job_id      = 100;
    payload.n_points    = 500000;
    payload.path_length = static_cast<uint32_t>(response_path.size());

    const uint32_t total_size = sizeof(FrameHeader) + sizeof(CloudResponsePayload) + payload.path_length;
    std::vector<uint8_t> buffer(total_size);

    FrameHeader header{};
    header.magic        = TCP_DEPTH_MAGIC;
    header.frame_length = total_size;
    header.message_type = MSG_CLOUD_RESPONSE;

    size_t offset = 0;
    std::memcpy(buffer.data() + offset, &header, sizeof(FrameHeader));
    offset += sizeof(FrameHeader);
    std::memcpy(buffer.data() + offset, &payload, sizeof(CloudResponsePayload));
    offset += sizeof(CloudResponsePayload);
    std::memcpy(buffer.data() + offset, response_path.data(), response_path.size());
    offset += response_path.size();

    REQUIRE(offset == total_size);

    FrameHeader dh{};
    std::memcpy(&dh, buffer.data(), sizeof(FrameHeader));
    CHECK(dh.message_type == MSG_CLOUD_RESPONSE);
    CHECK(validateHeader(&dh, buffer.size()));

    CloudResponsePayload dp{};
    std::memcpy(&dp, buffer.data() + sizeof(FrameHeader), sizeof(CloudResponsePayload));
    CHECK(dp.job_id      == 100);
    CHECK(dp.n_points    == 500000);
    CHECK(dp.path_length == response_path.size());

    std::string decoded_path(
        reinterpret_cast<const char*>(buffer.data() + sizeof(FrameHeader) + sizeof(CloudResponsePayload)),
        dp.path_length);
    CHECK(decoded_path == response_path);
}

TEST_CASE("cloud_protocol CloudCancelPayload round-trip")
{
    CloudCancelPayload payload{};
    payload.job_id = 7;

    std::array<uint8_t, sizeof(CloudCancelPayload)> buffer{};
    std::memcpy(buffer.data(), &payload, sizeof(CloudCancelPayload));

    CloudCancelPayload decoded{};
    std::memcpy(&decoded, buffer.data(), sizeof(CloudCancelPayload));
    CHECK(decoded.job_id == 7);
}

TEST_CASE("cloud_protocol CloudErrorPayload round-trip")
{
    CloudErrorPayload payload{};
    payload.job_id     = 42;
    payload.error_code = 5;
    std::memset(payload.error_msg, 0, sizeof(payload.error_msg));
    std::memcpy(payload.error_msg, "requires a pose-capable model", 29);

    std::array<uint8_t, sizeof(CloudErrorPayload)> buffer{};
    std::memcpy(buffer.data(), &payload, sizeof(CloudErrorPayload));

    CloudErrorPayload decoded{};
    std::memcpy(&decoded, buffer.data(), sizeof(CloudErrorPayload));

    CHECK(decoded.job_id     == 42);
    CHECK(decoded.error_code == 5);
    CHECK(std::string(decoded.error_msg) == "requires a pose-capable model");
}

TEST_CASE("cloud_protocol validateHeader rejects unknown type 0xFE")
{
    FrameHeader header{};
    header.magic        = TCP_DEPTH_MAGIC;
    header.frame_length = sizeof(FrameHeader);
    header.message_type = 0xFE;
    CHECK_FALSE(validateHeader(&header, sizeof(header)));
}

TEST_CASE("cloud_protocol getMessageTypeName returns correct names")
{
    CHECK(std::string(getMessageTypeName(MSG_CLOUD_REQUEST))  == "CLOUD_REQUEST");
    CHECK(std::string(getMessageTypeName(MSG_CLOUD_POLL))     == "CLOUD_POLL");
    CHECK(std::string(getMessageTypeName(MSG_CLOUD_PROGRESS)) == "CLOUD_PROGRESS");
    CHECK(std::string(getMessageTypeName(MSG_CLOUD_RESPONSE)) == "CLOUD_RESPONSE");
    CHECK(std::string(getMessageTypeName(MSG_CLOUD_CANCEL))   == "CLOUD_CANCEL");
    CHECK(std::string(getMessageTypeName(MSG_CLOUD_ERROR))    == "CLOUD_ERROR");
    CHECK(std::string(getMessageTypeName(0xFE))               == "UNKNOWN");
}

TEST_CASE("cloud_protocol computePayloadSize returns correct sizes")
{
    SUBCASE("CLOUD_REQUEST minimum")
    {
        FrameHeader h{};
        h.magic        = TCP_DEPTH_MAGIC;
        h.frame_length = sizeof(FrameHeader) + sizeof(CloudRequestPayload) + sizeof(CloudRequestOptions);
        h.message_type = MSG_CLOUD_REQUEST;
        CHECK(computePayloadSize(&h) == sizeof(CloudRequestPayload) + sizeof(CloudRequestOptions));
    }
    SUBCASE("CLOUD_POLL")
    {
        FrameHeader h{};
        h.magic        = TCP_DEPTH_MAGIC;
        h.frame_length = sizeof(FrameHeader) + sizeof(CloudPollPayload);
        h.message_type = MSG_CLOUD_POLL;
        CHECK(computePayloadSize(&h) == sizeof(CloudPollPayload));
    }
    SUBCASE("CLOUD_PROGRESS")
    {
        FrameHeader h{};
        h.magic        = TCP_DEPTH_MAGIC;
        h.frame_length = sizeof(FrameHeader) + sizeof(CloudProgressPayload);
        h.message_type = MSG_CLOUD_PROGRESS;
        CHECK(computePayloadSize(&h) == sizeof(CloudProgressPayload));
    }
    SUBCASE("CLOUD_RESPONSE minimum")
    {
        FrameHeader h{};
        h.magic        = TCP_DEPTH_MAGIC;
        h.frame_length = sizeof(FrameHeader) + sizeof(CloudResponsePayload);
        h.message_type = MSG_CLOUD_RESPONSE;
        CHECK(computePayloadSize(&h) == sizeof(CloudResponsePayload));
    }
    SUBCASE("CLOUD_CANCEL")
    {
        FrameHeader h{};
        h.magic        = TCP_DEPTH_MAGIC;
        h.frame_length = sizeof(FrameHeader) + sizeof(CloudCancelPayload);
        h.message_type = MSG_CLOUD_CANCEL;
        CHECK(computePayloadSize(&h) == sizeof(CloudCancelPayload));
    }
    SUBCASE("CLOUD_ERROR")
    {
        FrameHeader h{};
        h.magic        = TCP_DEPTH_MAGIC;
        h.frame_length = sizeof(FrameHeader) + sizeof(CloudErrorPayload);
        h.message_type = MSG_CLOUD_ERROR;
        CHECK(computePayloadSize(&h) == sizeof(CloudErrorPayload));
    }
    SUBCASE("null returns 0")  { CHECK(computePayloadSize(nullptr) == 0); }
    SUBCASE("malformed returns 0")
    {
        FrameHeader h{};
        h.magic        = TCP_DEPTH_MAGIC;
        h.frame_length = 0;
        h.message_type = MSG_CLOUD_REQUEST;
        CHECK(computePayloadSize(&h) == 0);
    }
}

TEST_CASE("cloud_protocol struct sizes match static_assert expectations")
{
    CHECK(sizeof(CloudRequestOptions)  == 64);
    CHECK(sizeof(CloudRequestPayload)  == 12);
    CHECK(sizeof(CloudPollPayload)     == 4);
    CHECK(sizeof(CloudCancelPayload)   == 4);
    CHECK(sizeof(CloudProgressPayload) == 16);
    CHECK(sizeof(CloudResponsePayload) == 12);
    CHECK(sizeof(CloudErrorPayload)    == 260);
}

// T4: Parser serialization round-trip tests

TEST_CASE("cloud_protocol_parser MSG_CLOUD_POLL serializer round-trip")
{
    const auto frame = TcpProtocolSerializer::serializeCloudPoll(42);

    REQUIRE(frame.size() == sizeof(FrameHeader) + sizeof(CloudPollPayload));

    FrameHeader header{};
    std::memcpy(&header, frame.data(), sizeof(header));
    CHECK(header.magic == TCP_DEPTH_MAGIC);
    CHECK(header.message_type == MSG_CLOUD_POLL);
    CHECK(header.frame_length == frame.size());

    CloudPollPayload payload{};
    std::memcpy(&payload, frame.data() + sizeof(FrameHeader), sizeof(payload));
    CHECK(payload.job_id == 42);
}

TEST_CASE("cloud_protocol_parser MSG_CLOUD_CANCEL serializer round-trip")
{
    const auto frame = TcpProtocolSerializer::serializeCloudCancel(7);

    REQUIRE(frame.size() == sizeof(FrameHeader) + sizeof(CloudCancelPayload));

    FrameHeader header{};
    std::memcpy(&header, frame.data(), sizeof(header));
    CHECK(header.magic == TCP_DEPTH_MAGIC);
    CHECK(header.message_type == MSG_CLOUD_CANCEL);

    CloudCancelPayload payload{};
    std::memcpy(&payload, frame.data() + sizeof(FrameHeader), sizeof(payload));
    CHECK(payload.job_id == 7);
}

TEST_CASE("cloud_protocol_parser MSG_CLOUD_PROGRESS serializer round-trip")
{
    CloudProgressPayload src{};
    src.job_id = 7;
    src.stage = 2;
    src.percent = 50;
    src.windows_done = 3;
    src.windows_total = 8;
    src.eta_ms = 45000;

    const auto frame = TcpProtocolSerializer::serializeCloudProgress(src);

    REQUIRE(frame.size() == sizeof(FrameHeader) + sizeof(CloudProgressPayload));

    FrameHeader header{};
    std::memcpy(&header, frame.data(), sizeof(header));
    CHECK(header.magic == TCP_DEPTH_MAGIC);
    CHECK(header.message_type == MSG_CLOUD_PROGRESS);

    CloudProgressPayload dst{};
    std::memcpy(&dst, frame.data() + sizeof(FrameHeader), sizeof(dst));
    CHECK(dst.job_id == 7);
    CHECK(dst.stage == 2);
    CHECK(dst.percent == 50);
    CHECK(dst.windows_done == 3);
    CHECK(dst.windows_total == 8);
    CHECK(dst.eta_ms == 45000);
}

TEST_CASE("cloud_protocol_parser MSG_CLOUD_RESPONSE with path serializer round-trip")
{
    CloudResponsePayload src{};
    src.job_id = 7;
    src.n_points = 5000;
    src.path_length = 0;

    const std::string path = "/tmp/clouds/job_7.splat";
    const auto frame = TcpProtocolSerializer::serializeCloudResponse(src, path);

    REQUIRE_FALSE(frame.empty());

    FrameHeader header{};
    std::memcpy(&header, frame.data(), sizeof(header));
    CHECK(header.magic == TCP_DEPTH_MAGIC);
    CHECK(header.message_type == MSG_CLOUD_RESPONSE);

    const uint32_t expected_path_len = static_cast<uint32_t>(path.size() + 1);
    const size_t expected_size = sizeof(FrameHeader) + sizeof(CloudResponsePayload) + expected_path_len;
    CHECK(header.frame_length == expected_size);
    CHECK(frame.size() == expected_size);

    CloudResponsePayload dst{};
    std::memcpy(&dst, frame.data() + sizeof(FrameHeader), sizeof(dst));
    CHECK(dst.job_id == 7);
    CHECK(dst.n_points == 5000);
    CHECK(dst.path_length == expected_path_len);

    const char* path_data = reinterpret_cast<const char*>(
        frame.data() + sizeof(FrameHeader) + sizeof(CloudResponsePayload));
    CHECK(std::string(path_data) == path);
}

TEST_CASE("cloud_protocol_parser MSG_CLOUD_ERROR with 250-char message")
{
    CloudErrorPayload src{};
    src.job_id = 7;
    src.error_code = 5;
    std::string msg(250, 'A');
    std::memcpy(src.error_msg, msg.c_str(), msg.size());
    src.error_msg[msg.size()] = '\0';

    const auto frame = TcpProtocolSerializer::serializeCloudError(src);

    REQUIRE(frame.size() == sizeof(FrameHeader) + sizeof(CloudErrorPayload));

    FrameHeader header{};
    std::memcpy(&header, frame.data(), sizeof(header));
    CHECK(header.magic == TCP_DEPTH_MAGIC);
    CHECK(header.message_type == MSG_CLOUD_ERROR);

    CloudErrorPayload dst{};
    std::memcpy(&dst, frame.data() + sizeof(FrameHeader), sizeof(dst));
    CHECK(dst.job_id == 7);
    CHECK(dst.error_code == 5);
    CHECK(std::string(dst.error_msg) == msg);
    CHECK(std::strlen(dst.error_msg) == 250);
}

TEST_CASE("cloud_protocol_parser MSG_CLOUD_REQUEST with 8 frame paths")
{
    CloudRequestPayload payload{};
    payload.n_frames = 8;
    payload.options_size = sizeof(CloudRequestOptions);
    payload.frame_data_size = 0;

    CloudRequestOptions opts{};
    opts.fuse = 1;
    opts.icp_refine = 0;
    opts.metric = 0;
    opts.chunk_size = 12;
    opts.overlap = 3;
    opts.conf_pct = 55.0f;
    opts.point_size = 1.2f;

    std::vector<std::string> paths;
    for (int i = 1; i <= 8; ++i) {
        paths.push_back("/tmp/frame_0" + std::to_string(i) + ".jpg");
    }

    const auto frame = TcpProtocolSerializer::serializeCloudRequest(payload, opts, paths);
    REQUIRE_FALSE(frame.empty());

    CloudRequestPayload out_payload{};
    CloudRequestOptions out_opts{};
    std::vector<std::string> out_paths;

    bool ok = decodeCloudRequest(frame.data(), frame.size(), out_payload, out_opts, out_paths);
    REQUIRE(ok);

    CHECK(out_payload.n_frames == 8);
    CHECK(out_payload.options_size == sizeof(CloudRequestOptions));

    CHECK(out_opts.fuse == 1);
    CHECK(out_opts.icp_refine == 0);
    CHECK(out_opts.metric == 0);
    CHECK(out_opts.chunk_size == 12);
    CHECK(out_opts.overlap == 3);
    CHECK(out_opts.conf_pct == doctest::Approx(55.0f));
    CHECK(out_opts.point_size == doctest::Approx(1.2f));

    REQUIRE(out_paths.size() == 8);
    for (int i = 0; i < 8; ++i) {
        CHECK(out_paths[i] == paths[i]);
    }
}

TEST_CASE("cloud_protocol_parser Truncated frame rejected")
{
    CloudRequestPayload payload{};
    payload.n_frames = 2;
    payload.options_size = sizeof(CloudRequestOptions);

    CloudRequestOptions opts{};
    opts.chunk_size = 4;
    opts.overlap = 1;
    opts.conf_pct = 50.0f;
    opts.point_size = 1.0f;

    std::vector<std::string> paths = {"/tmp/a.jpg", "/tmp/b.jpg"};

    auto frame = TcpProtocolSerializer::serializeCloudRequest(payload, opts, paths);
    REQUIRE(frame.size() > sizeof(FrameHeader) + sizeof(CloudRequestPayload) + sizeof(CloudRequestOptions));

    FrameHeader hdr{};
    std::memcpy(&hdr, frame.data(), sizeof(hdr));
    hdr.frame_length = 1024 * 1024;
    std::memcpy(frame.data(), &hdr, sizeof(hdr));

    frame.resize(100);

    CloudRequestPayload out_payload{};
    CloudRequestOptions out_opts{};
    std::vector<std::string> out_paths;

    bool ok = decodeCloudRequest(frame.data(), frame.size(), out_payload, out_opts, out_paths);
    CHECK_FALSE(ok);
}

TEST_CASE("cloud_protocol_parser Empty frame paths rejected")
{
    // Manually construct a frame with n_frames=0 (bypass serializer to avoid LOG SIGTRAP)
    const uint32_t total_size = sizeof(FrameHeader) + sizeof(CloudRequestPayload) + sizeof(CloudRequestOptions);
    std::vector<uint8_t> frame(total_size);

    FrameHeader header{};
    header.magic = TCP_DEPTH_MAGIC;
    header.frame_length = total_size;
    header.message_type = MSG_CLOUD_REQUEST;
    std::memcpy(frame.data(), &header, sizeof(header));

    CloudRequestPayload payload{};
    payload.n_frames = 0;
    payload.options_size = sizeof(CloudRequestOptions);
    payload.frame_data_size = 0;
    std::memcpy(frame.data() + sizeof(FrameHeader), &payload, sizeof(payload));

    CloudRequestOptions opts{};
    std::memcpy(frame.data() + sizeof(FrameHeader) + sizeof(CloudRequestPayload), &opts, sizeof(opts));

    CloudRequestPayload out_payload{};
    CloudRequestOptions out_opts{};
    std::vector<std::string> out_paths;

    bool ok = decodeCloudRequest(frame.data(), frame.size(), out_payload, out_opts, out_paths);
    CHECK_FALSE(ok);
}
