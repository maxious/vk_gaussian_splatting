#include "tcp_server.h"

#include "protocol.h"

#include <nvutils/logger.hpp>

#include <algorithm>
#include <arpa/inet.h>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <limits>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

extern std::atomic<bool> g_shutdownRequested;

namespace {

constexpr size_t kIoBufferSize = 64 * 1024;

int64_t nowMs()
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

bool checkedMul(size_t a, size_t b, size_t& out)
{
    if(a != 0 && b > (std::numeric_limits<size_t>::max() / a))
    {
        return false;
    }
    out = a * b;
    return true;
}

void resetClient(ClientConnection& client)
{
    if(client.fd >= 0)
    {
        close(client.fd);
    }
    client = ClientConnection{};
}

bool sendAll(int fd, const uint8_t* data, size_t size, uint64_t& bytes_sent)
{
    size_t offset = 0;
    while(offset < size)
    {
        const ssize_t rc = send(fd, data + offset, size - offset, MSG_NOSIGNAL);
        if(rc < 0)
        {
            if(errno == EINTR)
            {
                continue;
            }
            return false;
        }
        if(rc == 0)
        {
            return false;
        }

        offset += static_cast<size_t>(rc);
        bytes_sent += static_cast<uint64_t>(rc);
    }

    return true;
}

bool decodeFrameRequest(const uint8_t* message,
                        size_t message_len,
                        FrameRequestPayload& payload,
                        const uint8_t*& rgb_data,
                        size_t& rgb_size)
{
    if(message == nullptr || message_len < sizeof(FrameHeader) + sizeof(FrameRequestPayload))
    {
        return false;
    }

    std::memcpy(&payload, message + sizeof(FrameHeader), sizeof(payload));

    size_t pixel_count = 0;
    if(!checkedMul(static_cast<size_t>(payload.width), static_cast<size_t>(payload.height), pixel_count)
       || !checkedMul(pixel_count, static_cast<size_t>(3), rgb_size))
    {
        return false;
    }

    const size_t expected_size = sizeof(FrameHeader) + sizeof(FrameRequestPayload) + rgb_size;
    if(expected_size != message_len)
    {
        return false;
    }

    rgb_data = message + sizeof(FrameHeader) + sizeof(FrameRequestPayload);
    return true;
}

bool sendSplatError(int fd, uint32_t job_id, uint32_t error_code, const char* error_msg, uint64_t& bytes_tx)
{
    auto frame = TcpProtocolSerializer::serializeSplatError(job_id, error_code, error_msg);
    if(frame.empty())
    {
        return false;
    }
    return sendAll(fd, frame.data(), frame.size(), bytes_tx);
}

bool sendCloudError(int fd, uint32_t job_id, uint32_t error_code, const char* error_msg, uint64_t& bytes_tx)
{
    CloudErrorPayload payload{};
    payload.job_id = job_id;
    payload.error_code = error_code;
    std::strncpy(payload.error_msg, error_msg != nullptr ? error_msg : "unknown error", sizeof(payload.error_msg) - 1);
    payload.error_msg[sizeof(payload.error_msg) - 1] = '\0';
    auto frame = TcpProtocolSerializer::serializeCloudError(payload);
    if(frame.empty())
    {
        return false;
    }
    return sendAll(fd, frame.data(), frame.size(), bytes_tx);
}

}  // namespace

TcpServer::TcpServer(WorkerPool& pool, SplatWorkerPool* splatPool) : m_pool(pool), m_splatPool(splatPool) {}

TcpServer::~TcpServer()
{
    stop();
}

bool TcpServer::start(int port)
{
    stop();

    const int sock = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
    if(sock < 0)
    {
        LOGE("TcpServer: socket() failed: %s\n", std::strerror(errno));
        return false;
    }

    int reuse_addr = 1;
    if(setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &reuse_addr, sizeof(reuse_addr)) != 0)
    {
        LOGE("TcpServer: setsockopt(SO_REUSEADDR) failed: %s\n", std::strerror(errno));
        close(sock);
        return false;
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(static_cast<uint16_t>(port));

    if(bind(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0)
    {
        LOGE("TcpServer: bind(port=%d) failed: %s\n", port, std::strerror(errno));
        close(sock);
        return false;
    }

    if(listen(sock, SOMAXCONN) != 0)
    {
        LOGE("TcpServer: listen() failed: %s\n", std::strerror(errno));
        close(sock);
        return false;
    }

    m_listenFd = sock;
    m_port = port;
    m_clients.assign(MAX_CLIENTS, ClientConnection{});
    m_clientGenerations.assign(MAX_CLIENTS, 0);
    m_inFlightFrames.clear();
    m_lastCleanupMs = nowMs();
    m_totalFramesProcessed = 0;
    m_nextFrameIndex = 1;
    m_running.store(true);

    LOGI("TCP server listening on port %d\n", port);
    return true;
}

void TcpServer::stop()
{
    if(m_listenFd >= 0)
    {
        close(m_listenFd);
        m_listenFd = -1;
    }

    for(ClientConnection& client : m_clients)
    {
        resetClient(client);
    }

    m_inFlightFrames.clear();
    m_inFlightSplatJobs.clear();
    m_running.store(false);
}

bool TcpServer::isRunning() const
{
    return m_running.load();
}

void TcpServer::update()
{
    if(!m_running.load())
    {
        return;
    }

    if(g_shutdownRequested.load())
    {
        stop();
        return;
    }

    const int64_t current_time_ms = nowMs();

    for(;;)
    {
        sockaddr_in client_addr{};
        socklen_t client_addr_len = sizeof(client_addr);
        const int client_fd = accept4(m_listenFd, reinterpret_cast<sockaddr*>(&client_addr), &client_addr_len, SOCK_NONBLOCK);
        if(client_fd < 0)
        {
            if(errno == EAGAIN || errno == EWOULDBLOCK)
            {
                break;
            }
            if(errno == EINTR)
            {
                continue;
            }

            LOGE("TcpServer: accept4() failed: %s\n", std::strerror(errno));
            break;
        }

        int free_slot = -1;
        for(size_t i = 0; i < m_clients.size(); ++i)
        {
            if(m_clients[i].fd < 0)
            {
                free_slot = static_cast<int>(i);
                break;
            }
        }

        if(free_slot < 0)
        {
            LOGE("Max clients reached, rejecting connection\n");
            close(client_fd);
            continue;
        }

        ClientConnection& client = m_clients[static_cast<size_t>(free_slot)];
        client = ClientConnection{};
        client.fd = client_fd;
        client.last_activity_ms = current_time_ms;
        ++m_clientGenerations[static_cast<size_t>(free_slot)];

        char addr_buffer[INET_ADDRSTRLEN] = {};
        const char* printable_addr = inet_ntop(AF_INET, &client_addr.sin_addr, addr_buffer, sizeof(addr_buffer));
        LOGD("TcpServer: accepted client %d from %s:%u\n", free_slot,
             printable_addr != nullptr ? printable_addr : "unknown",
             static_cast<unsigned int>(ntohs(client_addr.sin_port)));
    }

    std::array<uint8_t, kIoBufferSize> recv_buffer{};
    for(size_t client_idx = 0; client_idx < m_clients.size(); ++client_idx)
    {
        ClientConnection& client = m_clients[client_idx];
        if(client.fd < 0)
        {
            continue;
        }

        bool close_client = false;
        pollfd pfd{};
        pfd.fd = client.fd;
        pfd.events = POLLIN;
        const int poll_rc = poll(&pfd, 1, 0);
        if(poll_rc < 0)
        {
            if(errno != EINTR)
            {
                LOGE("TcpServer: poll(client=%zu) failed: %s\n", client_idx, std::strerror(errno));
                close_client = true;
            }
        }
        else if(poll_rc > 0)
        {
            if((pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0)
            {
                close_client = true;
            }

            while(!close_client && (pfd.revents & POLLIN) != 0)
            {
                const ssize_t received = recv(client.fd, recv_buffer.data(), recv_buffer.size(), 0);
                if(received < 0)
                {
                    if(errno == EAGAIN || errno == EWOULDBLOCK)
                    {
                        break;
                    }
                    if(errno == EINTR)
                    {
                        continue;
                    }

                    LOGE("TcpServer: recv(client=%zu) failed: %s\n", client_idx, std::strerror(errno));
                    close_client = true;
                    break;
                }

                if(received == 0)
                {
                    close_client = true;
                    break;
                }

                client.bytes_rx += static_cast<uint64_t>(received);
                client.last_activity_ms = nowMs();
                client.parser.feed(recv_buffer.data(), static_cast<size_t>(received));

                while(!close_client && client.parser.hasMessage())
                {
                    size_t message_len = 0;
                    uint32_t message_type = 0;
                    const uint8_t* message = client.parser.nextMessage(message_len, message_type);
                    if(message == nullptr)
                    {
                        break;
                    }

                    if(message_type == MSG_FRAME_REQUEST)
                    {
                        FrameRequestPayload payload{};
                        const uint8_t* rgb_data = nullptr;
                        size_t rgb_size = 0;
                        if(!decodeFrameRequest(message, message_len, payload, rgb_data, rgb_size))
                        {
                            LOGE("TcpServer: invalid FRAME_REQUEST from client %zu\n", client_idx);
                            close_client = true;
                            break;
                        }

                        const uint32_t frame_index = m_nextFrameIndex++;
                        const int submit_rc = m_pool.submitFrame(frame_index, payload.timestamp_ms, rgb_data, payload.width,
                                                                payload.height);
                        if(submit_rc < 0)
                        {
                            LOGE("TcpServer: failed to submit frame %u for client %zu\n", frame_index, client_idx);
                            auto error_frame = TcpProtocolSerializer::serializeError(1, "failed to queue frame");
                            if(!error_frame.empty())
                            {
                                if(!sendAll(client.fd, error_frame.data(), error_frame.size(), client.bytes_tx))
                                {
                                    close_client = true;
                                    break;
                                }
                                client.last_activity_ms = nowMs();
                            }
                            continue;
                        }

                        ++client.frames_received;
                        m_inFlightFrames.emplace(frame_index,
                                                 InFlightFrame{static_cast<int>(client_idx),
                                                               m_clientGenerations[client_idx],
                                                               payload.timestamp_ms});
                        LOGD("TcpServer: queued frame %u from client %zu (%ux%u)\n", frame_index, client_idx,
                             payload.width, payload.height);
                    }
                    else if(message_type == MSG_SHUTDOWN)
                    {
                        auto ack = TcpProtocolSerializer::serializeShutdownAck();
                        if(!ack.empty())
                        {
                            if(sendAll(client.fd, ack.data(), ack.size(), client.bytes_tx))
                            {
                                client.last_activity_ms = nowMs();
                            }
                        }
                        close_client = true;
                    }
                    else if(message_type == MSG_SPLAT_REQUEST)
                    {
                        if(m_splatPool == nullptr)
                        {
                            sendSplatError(client.fd, 0, 0, "splat server not running", client.bytes_tx);
                            client.last_activity_ms = nowMs();
                            continue;
                        }

                        SplatRequestPayload splat_payload{};
                        const uint8_t* options_ptr = nullptr;
                        const uint8_t* images_ptr  = nullptr;
                        size_t images_size = 0;
                        if(!decodeSplatRequest(message, message_len, splat_payload, options_ptr, images_ptr, images_size))
                        {
                            LOGE("TcpServer: invalid SPLAT_REQUEST from client %zu\n", client_idx);
                            sendSplatError(client.fd, 0, 2, "invalid splat request", client.bytes_tx);
                            client.last_activity_ms = nowMs();
                            continue;
                        }

                        if(splat_payload.n_views < 2 || splat_payload.n_views > 4)
                        {
                            sendSplatError(client.fd, 0, 2, "n_views must be 2-4", client.bytes_tx);
                            client.last_activity_ms = nowMs();
                            continue;
                        }

                        if(images_size > static_cast<size_t>(MAX_IMAGE_SIZE) * splat_payload.n_views)
                        {
                            sendSplatError(client.fd, 0, 2, "image too large", client.bytes_tx);
                            client.last_activity_ms = nowMs();
                            continue;
                        }

                        std::vector<std::vector<uint8_t>> image_bytes;
                        image_bytes.reserve(splat_payload.n_views);
                        {
                            const uint8_t* cursor = images_ptr;
                            size_t remaining = images_size;
                            bool parse_ok = true;
                            for(uint32_t i = 0; i < splat_payload.n_views; ++i)
                            {
                                if(remaining < sizeof(uint32_t))
                                {
                                    parse_ok = false;
                                    break;
                                }
                                uint32_t len = 0;
                                std::memcpy(&len, cursor, sizeof(len));
                                cursor += sizeof(uint32_t);
                                remaining -= sizeof(uint32_t);

                                if(len > MAX_IMAGE_SIZE || remaining < len)
                                {
                                    parse_ok = false;
                                    break;
                                }
                                image_bytes.emplace_back(cursor, cursor + len);
                                cursor += len;
                                remaining -= len;
                            }
                            if(!parse_ok)
                            {
                                sendSplatError(client.fd, 0, 2, "malformed image data", client.bytes_tx);
                                client.last_activity_ms = nowMs();
                                continue;
                            }
                        }

                        SplatRequestOptions opts{};
                        if(options_ptr != nullptr)
                        {
                            std::memcpy(&opts, options_ptr, sizeof(opts));
                        }

                        const uint32_t job_id = m_splatPool->submitJob(image_bytes, opts);
                        if(job_id == 0)
                        {
                            sendSplatError(client.fd, 0, 1, "queue full", client.bytes_tx);
                            client.last_activity_ms = nowMs();
                            continue;
                        }

                        m_inFlightSplatJobs.emplace(job_id,
                                                   InFlightFrame{static_cast<int>(client_idx),
                                                                 m_clientGenerations[client_idx],
                                                                 0});
                        LOGD("TcpServer: queued splat job %u from client %zu (%u views)\n",
                             job_id, client_idx, splat_payload.n_views);

                        auto progress = TcpProtocolSerializer::serializeSplatProgress(job_id, 0, 0, 0);
                        if(!progress.empty())
                        {
                            sendAll(client.fd, progress.data(), progress.size(), client.bytes_tx);
                            client.last_activity_ms = nowMs();
                        }
                    }
                    else if(message_type == MSG_SPLAT_POLL)
                    {
                        if(m_splatPool == nullptr)
                        {
                            sendSplatError(client.fd, 0, 0, "splat server not running", client.bytes_tx);
                            client.last_activity_ms = nowMs();
                            continue;
                        }

                        if(message_len < sizeof(FrameHeader) + sizeof(SplatPollPayload))
                        {
                            continue;
                        }

                        SplatPollPayload poll{};
                        std::memcpy(&poll, message + sizeof(FrameHeader), sizeof(poll));

                        // Check if job is still in-flight. The main result polling loop
                        // (lines 680+) delivers results asynchronously — we only report
                        // progress here, never consume results (avoids race with main loop).
                        auto inflight_it = m_inFlightSplatJobs.find(poll.job_id);
                        if(inflight_it == m_inFlightSplatJobs.end())
                        {
                            sendSplatError(client.fd, poll.job_id, 0, "unknown job_id", client.bytes_tx);
                            client.last_activity_ms = nowMs();
                            continue;
                        }

                        uint8_t percent = static_cast<uint8_t>(
                            std::min(100, m_splatPool->queueDepth() * 10));
                        auto progress = TcpProtocolSerializer::serializeSplatProgress(poll.job_id, 1, percent, 0);
                        if(!progress.empty())
                        {
                            sendAll(client.fd, progress.data(), progress.size(), client.bytes_tx);
                        }
                        client.last_activity_ms = nowMs();
                    }
                    else if(message_type == MSG_SPLAT_CANCEL)
                    {
                        if(m_splatPool == nullptr)
                        {
                            sendSplatError(client.fd, 0, 0, "splat server not running", client.bytes_tx);
                            client.last_activity_ms = nowMs();
                            continue;
                        }

                        if(message_len < sizeof(FrameHeader) + sizeof(SplatCancelPayload))
                        {
                            continue;
                        }

                        SplatCancelPayload cancel{};
                        std::memcpy(&cancel, message + sizeof(FrameHeader), sizeof(cancel));

                        m_splatPool->cancelJob(cancel.job_id);
                        m_inFlightSplatJobs.erase(cancel.job_id);
                        LOGD("TcpServer: cancelled splat job %u from client %zu\n", cancel.job_id, client_idx);
                    }
                    else if(message_type == MSG_CLOUD_REQUEST)
                    {
                        if(m_cloudPool == nullptr)
                        {
                            sendCloudError(client.fd, 0, 0, "cloud server not running", client.bytes_tx);
                            client.last_activity_ms = nowMs();
                            continue;
                        }

                        CloudRequestPayload cloud_payload{};
                        CloudRequestOptions cloud_opts{};
                        std::vector<std::string> frame_paths;
                        if(!decodeCloudRequest(message, message_len, cloud_payload, cloud_opts, frame_paths))
                        {
                            LOGE("TcpServer: invalid CLOUD_REQUEST from client %zu\n", client_idx);
                            sendCloudError(client.fd, 0, 2, "invalid cloud request", client.bytes_tx);
                            client.last_activity_ms = nowMs();
                            continue;
                        }

                        if(cloud_payload.n_frames < 2 || cloud_payload.n_frames > MAX_CLOUD_FRAMES)
                        {
                            sendCloudError(client.fd, 0, 2, "n_frames must be 2-200", client.bytes_tx);
                            client.last_activity_ms = nowMs();
                            continue;
                        }

                        const uint32_t job_id = m_cloudPool->submitJob(frame_paths, cloud_opts);
                        if(job_id == 0)
                        {
                            sendCloudError(client.fd, 0, 1, "queue full", client.bytes_tx);
                            client.last_activity_ms = nowMs();
                            continue;
                        }

                        m_inFlightCloudJobs.emplace(job_id,
                                                    InFlightFrame{static_cast<int>(client_idx),
                                                                  m_clientGenerations[client_idx],
                                                                  0});
                        LOGD("TcpServer: queued cloud job %u from client %zu (%u frames)\n",
                             job_id, client_idx, cloud_payload.n_frames);

                        CloudProgressPayload progress{};
                        progress.job_id = job_id;
                        progress.stage = 0;
                        progress.percent = 0;
                        progress.windows_done = 0;
                        progress.windows_total = 0;
                        progress.eta_ms = 0;
                        auto progress_msg = TcpProtocolSerializer::serializeCloudProgress(progress);
                        if(!progress_msg.empty())
                        {
                            sendAll(client.fd, progress_msg.data(), progress_msg.size(), client.bytes_tx);
                            client.last_activity_ms = nowMs();
                        }
                    }
                    else if(message_type == MSG_CLOUD_POLL)
                    {
                        if(m_cloudPool == nullptr)
                        {
                            sendCloudError(client.fd, 0, 0, "cloud server not running", client.bytes_tx);
                            client.last_activity_ms = nowMs();
                            continue;
                        }

                        if(message_len < sizeof(FrameHeader) + sizeof(CloudPollPayload))
                        {
                            continue;
                        }

                        CloudPollPayload poll{};
                        std::memcpy(&poll, message + sizeof(FrameHeader), sizeof(poll));

                        auto inflight_it = m_inFlightCloudJobs.find(poll.job_id);
                        if(inflight_it == m_inFlightCloudJobs.end())
                        {
                            sendCloudError(client.fd, poll.job_id, 0, "unknown job_id", client.bytes_tx);
                            client.last_activity_ms = nowMs();
                            continue;
                        }

                        inflight_it->second.client_slot         = static_cast<int>(client_idx);
                        inflight_it->second.client_generation   = m_clientGenerations[client_idx];

                        auto pending = m_pendingCloudResults.find(poll.job_id);
                        if(pending != m_pendingCloudResults.end())
                        {
                            CloudJobResult cloud_res = std::move(pending->second);
                            m_pendingCloudResults.erase(pending);

                            if(cloud_res.error.empty())
                            {
                                CloudResponsePayload resp_payload{};
                                resp_payload.job_id = cloud_res.job_id;
                                resp_payload.n_points = cloud_res.n_points;
                                resp_payload.path_length = static_cast<uint32_t>(cloud_res.output_path.size());
                                auto resp = TcpProtocolSerializer::serializeCloudResponse(resp_payload, cloud_res.output_path);
                                if(!resp.empty())
                                {
                                    sendAll(client.fd, resp.data(), resp.size(), client.bytes_tx);
                                }
                            }
                            else
                            {
                                sendCloudError(client.fd, cloud_res.job_id, 3, cloud_res.error.c_str(), client.bytes_tx);
                            }
                            m_inFlightCloudJobs.erase(poll.job_id);
                            client.last_activity_ms = nowMs();
                            continue;
                        }

                        uint8_t percent = static_cast<uint8_t>(
                            std::min(100, m_cloudPool->queueDepth() * 10));
                        CloudProgressPayload progress{};
                        progress.job_id = poll.job_id;
                        progress.stage = 1;
                        progress.percent = percent;
                        progress.windows_done = 0;
                        progress.windows_total = 0;
                        progress.eta_ms = 0;
                        auto progress_msg = TcpProtocolSerializer::serializeCloudProgress(progress);
                        if(!progress_msg.empty())
                        {
                            sendAll(client.fd, progress_msg.data(), progress_msg.size(), client.bytes_tx);
                        }
                        client.last_activity_ms = nowMs();
                    }
                    else if(message_type == MSG_CLOUD_CANCEL)
                    {
                        if(m_cloudPool == nullptr)
                        {
                            sendCloudError(client.fd, 0, 0, "cloud server not running", client.bytes_tx);
                            client.last_activity_ms = nowMs();
                            continue;
                        }

                        if(message_len < sizeof(FrameHeader) + sizeof(CloudCancelPayload))
                        {
                            continue;
                        }

                        CloudCancelPayload cancel{};
                        std::memcpy(&cancel, message + sizeof(FrameHeader), sizeof(cancel));

                        m_cloudPool->cancelJob(cancel.job_id);
                        m_inFlightCloudJobs.erase(cancel.job_id);
                        LOGD("TcpServer: cancelled cloud job %u from client %zu\n", cancel.job_id, client_idx);
                    }
                    else
                    {
                        LOGW("TcpServer: ignoring unsupported message type %s from client %zu\n",
                             getMessageTypeName(message_type), client_idx);
                    }
                }

                if(received < static_cast<ssize_t>(recv_buffer.size()))
                {
                    break;
                }
            }
        }

        if(close_client)
        {
            LOGD("TcpServer: closing client %zu\n", client_idx);
            resetClient(client);
            ++m_clientGenerations[client_idx];
        }
    }

    for(;;)
    {
        int frame_index = -1;
        std::vector<float> depth_data;
        int out_w = 0;
        int out_h = 0;
        float scale = 0.0f;
        float bias = 0.0f;
        float z_max = 0.0f;
        if(!m_pool.pollResult(frame_index, depth_data, out_w, out_h, scale, bias, z_max))
        {
            break;
        }

        auto inflight_it = m_inFlightFrames.find(static_cast<uint32_t>(frame_index));
        if(inflight_it == m_inFlightFrames.end())
        {
            LOGW("TcpServer: no in-flight routing entry for frame %d\n", frame_index);
            ++m_totalFramesProcessed;
            continue;
        }

        const InFlightFrame inflight = inflight_it->second;
        m_inFlightFrames.erase(inflight_it);

        if(inflight.client_slot < 0 || inflight.client_slot >= static_cast<int>(m_clients.size()))
        {
            ++m_totalFramesProcessed;
            continue;
        }

        ClientConnection& client = m_clients[static_cast<size_t>(inflight.client_slot)];
        if(client.fd < 0 || m_clientGenerations[static_cast<size_t>(inflight.client_slot)] != inflight.client_generation)
        {
            LOGW("TcpServer: dropping frame %d because client %d is no longer active\n", frame_index,
                 inflight.client_slot);
            ++m_totalFramesProcessed;
            continue;
        }

        auto response = TcpProtocolSerializer::serializeDepthResponse(static_cast<uint32_t>(frame_index),
                                                                      inflight.timestamp_ms,
                                                                      static_cast<uint32_t>(out_w),
                                                                      static_cast<uint32_t>(out_h),
                                                                      scale,
                                                                      bias,
                                                                      z_max,
                                                                      depth_data.data());
        if(response.empty() || !sendAll(client.fd, response.data(), response.size(), client.bytes_tx))
        {
            LOGE("TcpServer: failed to send DEPTH_RESPONSE for frame %d to client %d: %s\n", frame_index,
                 inflight.client_slot, std::strerror(errno));
            resetClient(client);
            ++m_clientGenerations[static_cast<size_t>(inflight.client_slot)];
            ++m_totalFramesProcessed;
            continue;
        }

        client.last_activity_ms = nowMs();
        ++client.frames_processed;
        ++m_totalFramesProcessed;
    }

    if(m_splatPool != nullptr)
    {
        for(;;)
        {
            SplatJobResult splat_res;
            if(!m_splatPool->pollResult(splat_res))
            {
                break;
            }

            auto it = m_inFlightSplatJobs.find(splat_res.job_id);
            if(it == m_inFlightSplatJobs.end())
            {
                LOGW("TcpServer: no in-flight routing entry for splat job %u\n", splat_res.job_id);
                continue;
            }

            const InFlightFrame inflight = it->second;

            if(inflight.client_slot < 0 || inflight.client_slot >= static_cast<int>(m_clients.size()))
            {
                m_inFlightSplatJobs.erase(it);
                continue;
            }

            ClientConnection& client = m_clients[static_cast<size_t>(inflight.client_slot)];
            if(client.fd < 0 || m_clientGenerations[static_cast<size_t>(inflight.client_slot)] != inflight.client_generation)
            {
                LOGW("TcpServer: dropping splat job %u because client %d is no longer active\n",
                     splat_res.job_id, inflight.client_slot);
                m_inFlightSplatJobs.erase(it);
                continue;
            }

            if(splat_res.error.empty())
            {
                auto resp = TcpProtocolSerializer::serializeSplatResponse(
                    splat_res.job_id, splat_res.n_gaussians, splat_res.output_path.c_str());
                if(resp.empty() || !sendAll(client.fd, resp.data(), resp.size(), client.bytes_tx))
                {
                    LOGE("TcpServer: failed to send SPLAT_RESPONSE for job %u to client %d: %s\n",
                         splat_res.job_id, inflight.client_slot, std::strerror(errno));
                    resetClient(client);
                    ++m_clientGenerations[static_cast<size_t>(inflight.client_slot)];
                }
                else
                {
                    client.last_activity_ms = nowMs();
                    LOGD("TcpServer: delivered splat job %u to client %d (%u gaussians)\n",
                         splat_res.job_id, inflight.client_slot, splat_res.n_gaussians);
                }
            }
            else
            {
                if(!sendSplatError(client.fd, splat_res.job_id, 3, splat_res.error.c_str(), client.bytes_tx))
                {
                    LOGE("TcpServer: failed to send SPLAT_ERROR for job %u to client %d: %s\n",
                         splat_res.job_id, inflight.client_slot, std::strerror(errno));
                    resetClient(client);
                    ++m_clientGenerations[static_cast<size_t>(inflight.client_slot)];
                }
                else
                {
                    client.last_activity_ms = nowMs();
                }
            }

            m_inFlightSplatJobs.erase(it);
        }
    }

    // Poll cloud pool for completed jobs
    if(m_cloudPool != nullptr)
    {
        for(;;)
        {
            CloudJobResult cloud_res;
            if(!m_cloudPool->pollResult(cloud_res))
            {
                break;
            }

            auto it = m_inFlightCloudJobs.find(cloud_res.job_id);
            if(it == m_inFlightCloudJobs.end())
            {
                LOGW("TcpServer: no in-flight routing entry for cloud job %u\n", cloud_res.job_id);
                continue;
            }

            const InFlightFrame inflight = it->second;

            if(inflight.client_slot < 0 || inflight.client_slot >= static_cast<int>(m_clients.size()))
            {
                m_inFlightCloudJobs.erase(it);
                continue;
            }

            ClientConnection& client = m_clients[static_cast<size_t>(inflight.client_slot)];
            if(client.fd < 0 || m_clientGenerations[static_cast<size_t>(inflight.client_slot)] != inflight.client_generation)
            {
                LOGD("TcpServer: deferring cloud job %u result (client %d reconnecting)\n",
                     cloud_res.job_id, inflight.client_slot);
                m_pendingCloudResults[cloud_res.job_id] = std::move(cloud_res);
                continue;
            }

            if(cloud_res.error.empty())
            {
                CloudResponsePayload resp_payload{};
                resp_payload.job_id = cloud_res.job_id;
                resp_payload.n_points = cloud_res.n_points;
                resp_payload.path_length = static_cast<uint32_t>(cloud_res.output_path.size());
                auto resp = TcpProtocolSerializer::serializeCloudResponse(resp_payload, cloud_res.output_path);
                if(resp.empty() || !sendAll(client.fd, resp.data(), resp.size(), client.bytes_tx))
                {
                    LOGE("TcpServer: failed to send CLOUD_RESPONSE for job %u to client %d: %s\n",
                         cloud_res.job_id, inflight.client_slot, std::strerror(errno));
                    resetClient(client);
                    ++m_clientGenerations[static_cast<size_t>(inflight.client_slot)];
                }
                else
                {
                    client.last_activity_ms = nowMs();
                    LOGD("TcpServer: delivered cloud job %u to client %d (%u points)\n",
                         cloud_res.job_id, inflight.client_slot, cloud_res.n_points);
                }
            }
            else
            {
                if(!sendCloudError(client.fd, cloud_res.job_id, 3, cloud_res.error.c_str(), client.bytes_tx))
                {
                    LOGE("TcpServer: failed to send CLOUD_ERROR for job %u to client %d: %s\n",
                         cloud_res.job_id, inflight.client_slot, std::strerror(errno));
                    resetClient(client);
                    ++m_clientGenerations[static_cast<size_t>(inflight.client_slot)];
                }
                else
                {
                    client.last_activity_ms = nowMs();
                }
            }

            m_inFlightCloudJobs.erase(it);
        }
    }

    if(current_time_ms - m_lastCleanupMs >= CLEANUP_INTERVAL_MS)
    {
        for(size_t client_idx = 0; client_idx < m_clients.size(); ++client_idx)
        {
            ClientConnection& client = m_clients[client_idx];
            if(client.fd >= 0 && (current_time_ms - client.last_activity_ms) > IDLE_TIMEOUT_MS)
            {
                LOGI("TcpServer: disconnecting idle client %zu after %d ms\n", client_idx, IDLE_TIMEOUT_MS);
                resetClient(client);
                ++m_clientGenerations[client_idx];
            }
        }
        m_lastCleanupMs = current_time_ms;
    }

    if(g_shutdownRequested.load())
    {
        stop();
    }
}

int TcpServer::activeConnections() const
{
    int count = 0;
    for(const ClientConnection& client : m_clients)
    {
        if(client.fd >= 0)
        {
            ++count;
        }
    }
    return count;
}

uint64_t TcpServer::totalFramesProcessed() const
{
    return m_totalFramesProcessed;
}
