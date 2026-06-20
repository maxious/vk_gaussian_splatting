#include "tcp_server_manager.h"

#ifdef WITH_TCP_DEPTH

#include <nvutils/logger.hpp>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <exception>
#include <string_view>
#include <utility>

namespace {

constexpr int kDefaultTcpDepthPort = 9000;
constexpr int kConnectTimeoutMs = 100;
constexpr int kInitialBackoffMs = 100;

int64_t nowMs()
{
    return static_cast<int64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch())
                                    .count());
}

std::string trimCopy(std::string_view text)
{
    size_t begin = 0;
    while(begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin])) != 0)
    {
        ++begin;
    }

    size_t end = text.size();
    while(end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0)
    {
        --end;
    }

    return std::string(text.substr(begin, end - begin));
}

}  // namespace

int ServerConnection::getBackoffMs() const
{
    if(retry_count <= 1)
    {
        return kInitialBackoffMs;
    }

    int64_t backoff_ms = kInitialBackoffMs;
    for(int attempt = 1; attempt < retry_count; ++attempt)
    {
        backoff_ms = std::min<int64_t>(backoff_ms * 2, MAX_BACKOFF_MS);
    }

    return static_cast<int>(backoff_ms);
}

ServerConnection::ServerConnection(ServerConnection&& other) noexcept
    : host(std::move(other.host))
    , port(other.port)
    , frames_sent(other.frames_sent)
    , frames_completed(other.frames_completed)
    , frames_failed(other.frames_failed)
    , avg_latency_ms(other.avg_latency_ms)
    , is_connected(false)
    , last_connect_attempt(other.last_connect_attempt)
    , retry_count(other.retry_count)
{
}

ServerConnection& ServerConnection::operator=(ServerConnection&& other) noexcept
{
    if(this == &other)
    {
        return *this;
    }

    client.disconnect();
    host = std::move(other.host);
    port = other.port;
    frames_sent = other.frames_sent;
    frames_completed = other.frames_completed;
    frames_failed = other.frames_failed;
    avg_latency_ms = other.avg_latency_ms;
    is_connected = false;
    last_connect_attempt = other.last_connect_attempt;
    retry_count = other.retry_count;
    return *this;
}

TcpServerManager::TcpServerManager() = default;

TcpServerManager::~TcpServerManager()
{
    disconnectAll();
}

void TcpServerManager::addServer(const std::string& host, int port)
{
    const std::string trimmed_host = trimCopy(host);
    if(trimmed_host.empty())
    {
        LOGW("TcpServerManager: ignoring empty server host\n");
        return;
    }

    ServerConnection connection{};
    connection.host = trimmed_host;
    connection.port = port > 0 ? port : kDefaultTcpDepthPort;

    std::scoped_lock lock(m_mutex);
    const bool has_active_servers = std::any_of(m_servers.begin(), m_servers.end(), [](const ServerConnection& server) {
        return server.is_connected || server.client.isConnected();
    });
    if(has_active_servers)
    {
        LOGE("TcpServerManager: addServer() is only supported before connectAll()\n");
        return;
    }

    m_servers.push_back(std::move(connection));
    m_inFlightFrames.emplace_back();

    for(size_t server_idx = 0; server_idx < m_servers.size(); ++server_idx)
    {
        m_servers[server_idx].client.setDepthFrameCallback([this, server_idx](const DepthFrame& frame) {
            DepthBuffer* depth_buffer = nullptr;
            {
                std::scoped_lock lock(this->m_mutex);
                depth_buffer = this->m_depthBuffer;

                this->m_completedFrames.push_back({server_idx, frame});
            }

            if(depth_buffer)
            {
                depth_buffer->addFrame(frame);
            }
        });
    }
}

void TcpServerManager::setDepthBuffer(DepthBuffer* buffer)
{
    std::scoped_lock lock(m_mutex);
    m_depthBuffer = buffer;
}

void TcpServerManager::setFrameSkip(int video_fps, int server_fps_estimate)
{
    m_frameSkip = std::max(1, video_fps / std::max(1, server_fps_estimate));
    LOGI("TcpServerManager: frame skip set to %d (video %dfps, server ~%dfps)\n",
         m_frameSkip,
         video_fps,
         server_fps_estimate);
}

void TcpServerManager::connectAll()
{
    const int64_t current_ms = nowMs();
    for(ServerConnection& server : m_servers)
    {
        server.last_connect_attempt = current_ms;
        server.is_connected = server.client.connect(server.host, server.port, kConnectTimeoutMs);
        if(server.is_connected)
        {
            server.retry_count = 0;
        }
        else
        {
            server.retry_count = std::max(server.retry_count + 1, 1);
        }
    }
}

void TcpServerManager::disconnectAll()
{
    for(ServerConnection& server : m_servers)
    {
        server.client.disconnect();
        server.is_connected = false;
    }

    std::scoped_lock lock(m_mutex);
    for(std::vector<InFlightFrame>& frames : m_inFlightFrames)
    {
        frames.clear();
    }
    m_pendingFrames.clear();
    m_completedFrames.clear();
    m_nextServer = 0;
}

int TcpServerManager::sendFrame(uint32_t frame_index,
                                uint32_t timestamp_ms,
                                const uint8_t* rgb_data,
                                uint32_t width,
                                uint32_t height)
{
    if(m_frameSkip > 1 && (frame_index % static_cast<uint32_t>(m_frameSkip)) != 0)
    {
        return static_cast<int>(frame_index);
    }

    PendingFrame frame{};
    frame.frame_index = frame_index;
    frame.timestamp_ms = timestamp_ms;
    frame.width = width;
    frame.height = height;

    const size_t rgb_size = static_cast<size_t>(width) * static_cast<size_t>(height) * 3U;
    if(rgb_size > 0)
    {
        if(rgb_data == nullptr)
        {
            LOGE("TcpServerManager: null RGB data for frame %u\n", frame_index);
            return static_cast<int>(frame_index);
        }
        frame.rgb.assign(rgb_data, rgb_data + rgb_size);
    }

    if(m_servers.empty())
    {
        std::scoped_lock lock(m_mutex);
        m_pendingFrames.push_back(std::move(frame));
        LOGW("TcpServerManager: no servers configured, queued frame %u\n", frame_index);
        return static_cast<int>(frame_index);
    }

    const size_t server_count = m_servers.size();
    for(size_t offset = 0; offset < server_count; ++offset)
    {
        const size_t server_idx = (m_nextServer + offset) % server_count;
        if(dispatchToServer(frame, server_idx) >= 0)
        {
            m_nextServer = (server_idx + 1) % server_count;
            return static_cast<int>(frame_index);
        }
    }

    std::scoped_lock lock(m_mutex);
    m_pendingFrames.push_back(std::move(frame));
    return static_cast<int>(frame_index);
}

void TcpServerManager::setDepthFrameCallback(DepthFrameCallback cb)
{
    std::scoped_lock lock(m_mutex);
    m_callback = std::move(cb);
}

void TcpServerManager::update()
{
    const int64_t current_ms = nowMs();

    for(ServerConnection& server : m_servers)
    {
        server.client.update();
    }

    {
        std::scoped_lock lock(m_mutex);
        for(size_t server_idx = 0; server_idx < m_servers.size(); ++server_idx)
        {
            ServerConnection& server = m_servers[server_idx];
            const bool connected = server.client.isConnected();
            if(server.is_connected && !connected)
            {
                server.is_connected = false;
                server.last_connect_attempt = current_ms;
                server.retry_count = std::max(server.retry_count + 1, 1);
                requeueDisconnectedFrames(server_idx, current_ms);
            }
            else if(!server.is_connected && connected)
            {
                server.is_connected = true;
                server.retry_count = 0;
            }
        }
    }

    processCompletedFrames();
    reconnectFailedServers();

    std::vector<PendingFrame> pending_frames;
    {
        std::scoped_lock lock(m_mutex);
        pending_frames.swap(m_pendingFrames);
    }

    std::vector<PendingFrame> remaining_frames;
    remaining_frames.reserve(pending_frames.size());

    for(PendingFrame& frame : pending_frames)
    {
        bool dispatched = false;
        const size_t server_count = m_servers.size();
        for(size_t offset = 0; offset < server_count; ++offset)
        {
            const size_t server_idx = (m_nextServer + offset) % server_count;
            if(dispatchToServer(frame, server_idx) >= 0)
            {
                m_nextServer = (server_idx + 1) % server_count;
                dispatched = true;
                break;
            }
        }

        if(!dispatched)
        {
            remaining_frames.push_back(std::move(frame));
        }
    }

    if(!remaining_frames.empty())
    {
        std::scoped_lock lock(m_mutex);
        m_pendingFrames.insert(m_pendingFrames.end(),
                               std::make_move_iterator(remaining_frames.begin()),
                               std::make_move_iterator(remaining_frames.end()));
    }
}

size_t TcpServerManager::serverCount() const
{
    return m_servers.size();
}

const std::vector<ServerConnection>& TcpServerManager::getServers() const
{
    return m_servers;
}

void TcpServerManager::parseServerList(const std::string& list)
{
    size_t start = 0;
    while(start <= list.size())
    {
        const size_t end = list.find(',', start);
        const std::string entry = trimCopy(std::string_view(list).substr(start, end - start));
        if(!entry.empty())
        {
            const size_t colon = entry.rfind(':');
            if(colon == std::string::npos)
            {
                addServer(entry, kDefaultTcpDepthPort);
            }
            else
            {
                const std::string host = trimCopy(std::string_view(entry).substr(0, colon));
                const std::string port_text = trimCopy(std::string_view(entry).substr(colon + 1));
                int port = kDefaultTcpDepthPort;
                if(!port_text.empty())
                {
                    try
                    {
                        port = std::stoi(port_text);
                    }
                    catch(const std::exception&)
                    {
                        LOGW("TcpServerManager: invalid port '%s', using %d\n", port_text.c_str(), kDefaultTcpDepthPort);
                        port = kDefaultTcpDepthPort;
                    }
                }
                addServer(host, port);
            }
        }

        if(end == std::string::npos)
        {
            break;
        }
        start = end + 1;
    }
}

int TcpServerManager::dispatchToServer(PendingFrame& frame, size_t server_idx)
{
    if(server_idx >= m_servers.size())
    {
        return -1;
    }

    ServerConnection* server = nullptr;
    {
        std::scoped_lock lock(m_mutex);
        ServerConnection& candidate = m_servers[server_idx];
        if(!candidate.is_connected || !candidate.client.isConnected())
        {
            return -1;
        }
        server = &candidate;
    }

    server->client.sendFrameRequest(frame.frame_index,
                                    frame.timestamp_ms,
                                    frame.rgb.empty() ? nullptr : frame.rgb.data(),
                                    frame.width,
                                    frame.height);

    {
        std::scoped_lock lock(m_mutex);
        ServerConnection& connected_server = m_servers[server_idx];
        ++connected_server.frames_sent;
        m_inFlightFrames[server_idx].push_back({std::move(frame), nowMs()});
    }

    return static_cast<int>(server_idx);
}

void TcpServerManager::reconnectFailedServers()
{
    const int64_t current_ms = nowMs();
    for(ServerConnection& server : m_servers)
    {
        if(server.is_connected || server.client.isConnected())
        {
            server.is_connected = true;
            server.retry_count = 0;
            continue;
        }

        if(server.last_connect_attempt != 0 && (current_ms - server.last_connect_attempt) < server.getBackoffMs())
        {
            continue;
        }

        server.last_connect_attempt = current_ms;
        if(server.client.connect(server.host, server.port, kConnectTimeoutMs))
        {
            server.is_connected = true;
            server.retry_count = 0;
            LOGI("TcpServerManager: reconnected %s:%d\n", server.host.c_str(), server.port);
        }
        else
        {
            server.is_connected = false;
            server.retry_count = std::max(server.retry_count + 1, 1);
        }
    }
}

void TcpServerManager::processCompletedFrames()
{
    std::vector<CompletedFrame> completed_frames;
    DepthFrameCallback callback;
    {
        std::scoped_lock lock(m_mutex);
        completed_frames.swap(m_completedFrames);
        callback = m_callback;
    }

    const int64_t current_ms = nowMs();
    for(CompletedFrame& completed : completed_frames)
    {
        bool matched_in_flight = false;
        {
            std::scoped_lock lock(m_mutex);
            if(completed.server_idx < m_servers.size())
            {
                ServerConnection& server = m_servers[completed.server_idx];
                std::vector<InFlightFrame>& in_flight = m_inFlightFrames[completed.server_idx];
                auto it = std::find_if(in_flight.begin(), in_flight.end(), [&](const InFlightFrame& in_flight_frame) {
                    return in_flight_frame.frame.timestamp_ms == completed.frame.timestampMs;
                });

        if(it != in_flight.end())
        {
            const double latency_ms = static_cast<double>(current_ms - it->dispatch_time_ms);
            ++server.frames_completed;
                    if(server.frames_completed == 1)
                    {
                        server.avg_latency_ms = latency_ms;
                    }
                    else
                    {
                        server.avg_latency_ms += (latency_ms - server.avg_latency_ms)
                                                 / static_cast<double>(server.frames_completed);
            }
            in_flight.erase(it);
            matched_in_flight = true;
        }
    }
        }

        if(!matched_in_flight)
        {
            LOGD("TcpServerManager: completed frame timestamp %u did not match in-flight work\n",
                 completed.frame.timestampMs);
        }

        if(callback)
        {
            callback(completed.frame);
        }
    }
}

void TcpServerManager::requeueDisconnectedFrames(size_t server_idx, int64_t now_ms)
{
    if(server_idx >= m_servers.size())
    {
        return;
    }

    ServerConnection& server = m_servers[server_idx];
    std::vector<InFlightFrame>& in_flight = m_inFlightFrames[server_idx];
    for(InFlightFrame& frame : in_flight)
    {
        ++server.frames_failed;
        ++frame.frame.retry_count;
        if(frame.frame.retry_count <= PendingFrame::MAX_RETRIES)
        {
            m_pendingFrames.push_back(std::move(frame.frame));
        }
        else
        {
            LOGW("TcpServerManager: dropping frame %u after %d retries on %s:%d\n",
                 frame.frame.frame_index,
                 frame.frame.retry_count,
                 server.host.c_str(),
                 server.port);
        }
    }
    in_flight.clear();
    server.last_connect_attempt = now_ms;
}

#endif
