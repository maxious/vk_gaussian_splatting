// Copyright (c) 2025, vk_gaussian_splatting contributors
// Licensed under Apache 2.0 - see LICENSE for details

#ifndef WITH_TCP_DEPTH
#define WITH_TCP_DEPTH
#endif

#include "doctest.h"

#include "../src/tcp_depth_client.h"
#include "../src/tcp_server_manager.h"
#include "../depth_server/protocol.h"
#include "../depth_server/protocol_parser.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <poll.h>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace std::chrono_literals;

constexpr uint32_t kWidth = 2;
constexpr uint32_t kHeight = 2;
constexpr uint32_t kTimestamp = 1234;
constexpr int kTestTimeoutMs = 3000;

std::vector<uint8_t> makeRgb()
{
    return {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
}

std::vector<float> makeDepth()
{
    return {0.5f, 1.25f, 2.5f, 3.75f};
}

bool waitUntil(const std::function<bool()>& predicate, int timeout_ms = kTestTimeoutMs)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while(std::chrono::steady_clock::now() < deadline)
    {
        if(predicate())
        {
            return true;
        }
        std::this_thread::sleep_for(10ms);
    }
    return predicate();
}

void sendAll(int fd, const uint8_t* data, size_t size)
{
    while(size > 0)
    {
        const ssize_t written = ::send(fd, data, size, MSG_NOSIGNAL);
        REQUIRE(written > 0);
        data += static_cast<size_t>(written);
        size -= static_cast<size_t>(written);
    }
}

class MockDepthServer {
public:
    enum class Mode { RESPOND, CLOSE_AFTER_REQUEST, HOLD_OPEN };

    explicit MockDepthServer(Mode mode = Mode::RESPOND)
        : m_mode(mode)
    {
    }

    ~MockDepthServer()
    {
        stop();
    }

    void start()
    {
        REQUIRE(m_listenFd < 0);

        m_listenFd = ::socket(AF_INET, SOCK_STREAM, 0);
        REQUIRE(m_listenFd >= 0);

        int reuse = 1;
        REQUIRE(::setsockopt(m_listenFd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) == 0);

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = 0;
        REQUIRE(::bind(m_listenFd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0);

        socklen_t addr_len = sizeof(addr);
        REQUIRE(::getsockname(m_listenFd, reinterpret_cast<sockaddr*>(&addr), &addr_len) == 0);
        m_port = ntohs(addr.sin_port);

        REQUIRE(::listen(m_listenFd, 8) == 0);
        int flags = ::fcntl(m_listenFd, F_GETFL, 0);
        REQUIRE(flags >= 0);
        REQUIRE(::fcntl(m_listenFd, F_SETFL, flags | O_NONBLOCK) == 0);

        m_running.store(true);
        m_thread = std::thread([this] { run(); });
    }

    void stop()
    {
        m_running.store(false);
        if(m_listenFd >= 0)
        {
            ::shutdown(m_listenFd, SHUT_RDWR);
            ::close(m_listenFd);
            m_listenFd = -1;
        }
        if(m_thread.joinable())
        {
            m_thread.join();
        }
    }

    int port() const { return m_port; }
    size_t requestCount() const { return m_requestCount.load(); }
    size_t connectionCount() const { return m_connectionCount.load(); }

private:
    void run()
    {
        while(m_running.load())
        {
            pollfd pfd{};
            pfd.fd = m_listenFd;
            pfd.events = POLLIN;
            const int rc = ::poll(&pfd, 1, 100);
            if(!m_running.load())
            {
                break;
            }
            if(rc <= 0)
            {
                continue;
            }

            sockaddr_in client_addr{};
            socklen_t client_len = sizeof(client_addr);
            const int client_fd = ::accept(m_listenFd, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
            if(client_fd < 0)
            {
                if(errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)
                {
                    continue;
                }
                break;
            }

            ++m_connectionCount;
            handleConnection(client_fd);
        }
    }

    void handleConnection(int client_fd)
    {
        TcpProtocolParser parser;
        std::array<uint8_t, 4096> buffer{};
        bool keep_running = true;

        while(m_running.load() && keep_running)
        {
            pollfd pfd{};
            pfd.fd = client_fd;
            pfd.events = POLLIN | POLLHUP | POLLERR;
            const int rc = ::poll(&pfd, 1, 100);
            if(rc == 0)
            {
                if(m_mode == Mode::HOLD_OPEN)
                {
                    continue;
                }
                continue;
            }
            if(rc < 0)
            {
                if(errno == EINTR)
                {
                    continue;
                }
                break;
            }

            if((pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0)
            {
                break;
            }

            const ssize_t received = ::recv(client_fd, buffer.data(), buffer.size(), 0);
            if(received <= 0)
            {
                break;
            }

            parser.feed(buffer.data(), static_cast<size_t>(received));
            while(parser.hasMessage())
            {
                size_t message_len = 0;
                uint32_t message_type = 0;
                const uint8_t* message = parser.nextMessage(message_len, message_type);
                if(message == nullptr)
                {
                    break;
                }

                if(message_type != MSG_FRAME_REQUEST)
                {
                    continue;
                }

                ++m_requestCount;

                FrameRequestPayload request{};
                REQUIRE(message_len >= sizeof(FrameHeader) + sizeof(request));
                std::memcpy(&request, message + sizeof(FrameHeader), sizeof(request));

                if(m_mode == Mode::CLOSE_AFTER_REQUEST)
                {
                    keep_running = false;
                    break;
                }

                if(m_mode == Mode::HOLD_OPEN)
                {
                    continue;
                }

                const size_t depth_count = static_cast<size_t>(request.width) * static_cast<size_t>(request.height);
                std::vector<float> depth(depth_count, 0.0f);
                const std::vector<float> base = makeDepth();
                for(size_t i = 0; i < depth.size(); ++i)
                {
                    depth[i] = base[i % base.size()];
                }

                auto response = TcpProtocolSerializer::serializeDepthResponse(0,
                                                                              request.timestamp_ms,
                                                                              request.width,
                                                                              request.height,
                                                                              0.000001f,
                                                                              0.646956f,
                                                                              0.734110f,
                                                                              depth.data());
                REQUIRE_FALSE(response.empty());
                sendAll(client_fd, response.data(), response.size());
            }
        }

        ::close(client_fd);
    }

    Mode m_mode;
    int m_listenFd = -1;
    int m_port = 0;
    std::atomic<bool> m_running{false};
    std::atomic<size_t> m_requestCount{0};
    std::atomic<size_t> m_connectionCount{0};
    std::thread m_thread;
};

std::array<uint8_t, 12> rgbArray()
{
    std::array<uint8_t, 12> rgb{};
    auto rgb_vec = makeRgb();
    std::memcpy(rgb.data(), rgb_vec.data(), rgb.size());
    return rgb;
}

}  // namespace

TEST_CASE("tcp_depth_client round-robin dispatch")
{
    SUBCASE("3 servers, 9 frames -> 3 each")
    {
        MockDepthServer s1;
        MockDepthServer s2;
        MockDepthServer s3;
        s1.start();
        s2.start();
        s3.start();

        TcpServerManager manager;
        manager.addServer("127.0.0.1", s1.port());
        manager.addServer("127.0.0.1", s2.port());
        manager.addServer("127.0.0.1", s3.port());
        manager.connectAll();

        REQUIRE(waitUntil([&] {
            const auto& servers = manager.getServers();
            return servers.size() == 3 && servers[0].client.isConnected() && servers[1].client.isConnected()
                   && servers[2].client.isConnected();
        }));

        const auto rgb = rgbArray();
        for(uint32_t i = 0; i < 9; ++i)
        {
            manager.sendFrame(i, kTimestamp + i, rgb.data(), kWidth, kHeight);
        }

        REQUIRE(waitUntil([&] { return s1.requestCount() == 3 && s2.requestCount() == 3 && s3.requestCount() == 3; }));

        const auto& servers = manager.getServers();
        CHECK(servers[0].frames_sent == 3);
        CHECK(servers[1].frames_sent == 3);
        CHECK(servers[2].frames_sent == 3);
    }
}

TEST_CASE("tcp_depth_client re-queues failed frames")
{
    SUBCASE("server A fails, frame moves to B")
    {
        MockDepthServer server_a(MockDepthServer::Mode::CLOSE_AFTER_REQUEST);
        MockDepthServer server_b;
        server_a.start();
        server_b.start();

        TcpServerManager manager;
        manager.addServer("127.0.0.1", server_a.port());
        manager.addServer("127.0.0.1", server_b.port());
        manager.connectAll();

        REQUIRE(waitUntil([&] {
            const auto& servers = manager.getServers();
            return servers[0].client.isConnected() && servers[1].client.isConnected();
        }));

        const auto rgb = rgbArray();
        manager.sendFrame(1, kTimestamp, rgb.data(), kWidth, kHeight);

        REQUIRE(waitUntil([&] { return server_a.requestCount() == 1; }));
        REQUIRE(waitUntil([&] { return !manager.getServers()[0].client.isConnected(); }));

        for(int i = 0; i < 50 && server_b.requestCount() == 0; ++i)
        {
            manager.update();
            std::this_thread::sleep_for(20ms);
        }

        REQUIRE(waitUntil([&] { return server_b.requestCount() == 1; }));
        manager.update();

        const auto& servers = manager.getServers();
        CHECK(servers[0].frames_failed == 1);
        CHECK(servers[0].frames_sent == 1);
        CHECK(servers[1].frames_sent == 1);
    }
}

TEST_CASE("tcp_depth_client drops frames after max retries")
{
    SUBCASE("frame is dropped after three retries")
    {
        MockDepthServer server(MockDepthServer::Mode::CLOSE_AFTER_REQUEST);
        server.start();

        TcpServerManager manager;
        manager.addServer("127.0.0.1", server.port());
        manager.connectAll();

        REQUIRE(waitUntil([&] { return manager.getServers()[0].client.isConnected(); }));

        const auto rgb = rgbArray();
        manager.sendFrame(7, kTimestamp, rgb.data(), kWidth, kHeight);

        REQUIRE(waitUntil([&] {
            manager.update();
            return server.requestCount() >= 4;
        }, 15000));

        const size_t before = server.requestCount();
        for(int i = 0; i < 20; ++i)
        {
            manager.update();
            std::this_thread::sleep_for(20ms);
        }

        CHECK(server.requestCount() == before);
        const auto& servers = manager.getServers();
        CHECK(servers[0].frames_failed >= 4);
        CHECK(servers[0].frames_sent >= 4);
    }
}

TEST_CASE("tcp_depth_client connection backoff")
{
    SUBCASE("retry intervals double until capped")
    {
        ServerConnection server{};
        server.retry_count = 1;
        CHECK(server.getBackoffMs() == 100);
        server.retry_count = 2;
        CHECK(server.getBackoffMs() == 200);
        server.retry_count = 3;
        CHECK(server.getBackoffMs() == 400);
        server.retry_count = 5;
        CHECK(server.getBackoffMs() == 1600);
        server.retry_count = 20;
        CHECK(server.getBackoffMs() == ServerConnection::MAX_BACKOFF_MS);
    }
}

TEST_CASE("tcp_depth_client DepthFrame construction")
{
    SUBCASE("DEPTH_RESPONSE fields match the callback frame")
    {
        MockDepthServer server;
        server.start();

        TcpDepthClient client;
        std::mutex mutex;
        std::condition_variable cv;
        bool got_frame = false;
        DepthFrame frame{};

        client.setDepthFrameCallback([&](const DepthFrame& received) {
            std::scoped_lock lock(mutex);
            frame = received;
            got_frame = true;
            cv.notify_one();
        });

        REQUIRE(client.connect("127.0.0.1", server.port(), 500));
        const auto rgb = rgbArray();
        client.sendFrameRequest(9, kTimestamp, rgb.data(), kWidth, kHeight);

        {
            std::unique_lock lock(mutex);
            REQUIRE(cv.wait_for(lock, std::chrono::milliseconds(kTestTimeoutMs), [&] { return got_frame; }));
        }

        CHECK(frame.timestampMs == kTimestamp);
        CHECK(frame.width == kWidth);
        CHECK(frame.height == kHeight);
        CHECK(frame.scale == doctest::Approx(0.000001f));
        CHECK(frame.bias == doctest::Approx(0.646956f));
        CHECK(frame.zMax == doctest::Approx(0.734110f));
        CHECK(frame.data == makeDepth());
        client.disconnect();
    }
}

TEST_CASE("tcp_depth_client frame skip interval")
{
    SUBCASE("30 fps video and 3 fps server skips 10 frames")
    {
        MockDepthServer server;
        server.start();

        TcpServerManager manager;
        manager.addServer("127.0.0.1", server.port());
        manager.connectAll();
        manager.setFrameSkip(30, 3);

        REQUIRE(waitUntil([&] { return manager.getServers()[0].client.isConnected(); }));

        const auto rgb = rgbArray();
        for(uint32_t i = 0; i < 30; ++i)
        {
            manager.sendFrame(i, kTimestamp + i, rgb.data(), kWidth, kHeight);
        }

        REQUIRE(waitUntil([&] { return server.requestCount() == 3; }));
        CHECK(server.requestCount() == 3);
        CHECK(manager.getServers()[0].frames_sent == 3);
    }
}

TEST_CASE("tcp_depth_client pending request timeout")
{
    SUBCASE("server holds the socket open without replying")
    {
        MockDepthServer server(MockDepthServer::Mode::HOLD_OPEN);
        server.start();

        TcpDepthClient client;
        REQUIRE(client.connect("127.0.0.1", server.port(), 200));

        const auto rgb = rgbArray();
        client.sendFrameRequest(3, kTimestamp, rgb.data(), kWidth, kHeight);

        REQUIRE(waitUntil([&] { return !client.isConnected(); }, 10000));
        CHECK_FALSE(client.isConnected());
        client.disconnect();
    }
}

TEST_CASE("tcp_depth_client mock server integration")
{
    SUBCASE("minimal loopback server handles multiple requests")
    {
        MockDepthServer server;
        server.start();

        TcpServerManager manager;
        manager.addServer("127.0.0.1", server.port());
        manager.connectAll();

        REQUIRE(waitUntil([&] { return manager.getServers()[0].client.isConnected(); }));

        std::mutex mutex;
        std::condition_variable cv;
        size_t callback_count = 0;
        manager.setDepthFrameCallback([&](const DepthFrame&) {
            std::scoped_lock lock(mutex);
            ++callback_count;
            cv.notify_one();
        });

        const auto rgb = rgbArray();
        manager.sendFrame(1, kTimestamp, rgb.data(), kWidth, kHeight);
        manager.sendFrame(2, kTimestamp + 1, rgb.data(), kWidth, kHeight);

        REQUIRE(waitUntil([&] {
            manager.update();
            std::scoped_lock lock(mutex);
            return callback_count == 2;
        }, 10000));

        CHECK(server.requestCount() == 2);
        CHECK(callback_count == 2);
        CHECK(manager.getServers()[0].frames_sent == 2);
    }
}
