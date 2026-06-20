#include "tcp_depth_client.h"

#ifdef WITH_TCP_DEPTH

#include "tcp_depth_protocol.h"

#include "../depth_server/protocol_parser.h"

#include <nvutils/logger.hpp>

#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <netdb.h>
#include <poll.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <unistd.h>

namespace {

constexpr int kIoPollTimeoutMs = 100;
constexpr int kMaxEpollEvents = 4;

uint64_t nowMs()
{
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                     std::chrono::steady_clock::now().time_since_epoch())
                                     .count());
}

const char* toString(TcpDepthClient::State state)
{
    switch(state)
    {
    case TcpDepthClient::State::DISCONNECTED: return "DISCONNECTED";
    case TcpDepthClient::State::CONNECTING: return "CONNECTING";
    case TcpDepthClient::State::CONNECTED: return "CONNECTED";
    case TcpDepthClient::State::DISCONNECTING: return "DISCONNECTING";
    }

    return "UNKNOWN";
}

void closeFd(int& fd)
{
    if(fd >= 0)
    {
        close(fd);
        fd = -1;
    }
}

bool checkedMul(size_t a, size_t b, size_t& out)
{
    if(a != 0 && b > std::numeric_limits<size_t>::max() / a)
    {
        return false;
    }

    out = a * b;
    return true;
}

bool waitForSocketEvent(int fd, short events, int timeout_ms)
{
    pollfd pfd{};
    pfd.fd = fd;
    pfd.events = events;

    for(;;)
    {
        const int rc = poll(&pfd, 1, timeout_ms);
        if(rc > 0)
        {
            if((pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0)
            {
                return false;
            }
            return (pfd.revents & events) != 0;
        }
        if(rc == 0)
        {
            return false;
        }
        if(errno != EINTR)
        {
            return false;
        }
    }
}

bool writeAll(int fd, const uint8_t* data, size_t size, int timeout_ms)
{
    const uint64_t deadline_ms = nowMs() + static_cast<uint64_t>(std::max(timeout_ms, 0));
    while(size > 0)
    {
        const ssize_t n = send(fd, data, size, MSG_NOSIGNAL);
        if(n > 0)
        {
            data += n;
            size -= static_cast<size_t>(n);
            continue;
        }
        if(n == 0)
        {
            return false;
        }

        if(errno == EINTR)
        {
            continue;
        }
        if(errno != EAGAIN && errno != EWOULDBLOCK)
        {
            return false;
        }

        const uint64_t current_ms = nowMs();
        if(current_ms >= deadline_ms)
        {
            errno = ETIMEDOUT;
            return false;
        }

        const int wait_ms = static_cast<int>(deadline_ms - current_ms);
        if(!waitForSocketEvent(fd, POLLOUT, wait_ms))
        {
            errno = ETIMEDOUT;
            return false;
        }
    }

    return true;
}

bool readSocketData(int fd, TcpProtocolParser& parser, bool& peer_closed)
{
    uint8_t buffer[64 * 1024];
    peer_closed = false;

    for(;;)
    {
        const ssize_t received = recv(fd, buffer, sizeof(buffer), 0);
        if(received > 0)
        {
            parser.feed(buffer, static_cast<size_t>(received));
            if(received < static_cast<ssize_t>(sizeof(buffer)))
            {
                return true;
            }
            continue;
        }

        if(received == 0)
        {
            peer_closed = true;
            return false;
        }

        if(errno == EINTR)
        {
            continue;
        }
        if(errno == EAGAIN || errno == EWOULDBLOCK)
        {
            return true;
        }

        return false;
    }
}

bool decodeDepthResponse(const uint8_t* message, size_t message_len, DepthFrame& out_frame)
{
    if(message == nullptr || message_len < sizeof(FrameHeader) + sizeof(DepthResponsePayload))
    {
        return false;
    }

    FrameHeader header{};
    std::memcpy(&header, message, sizeof(header));
    if(!validateHeader(&header, message_len) || header.message_type != MSG_DEPTH_RESPONSE || header.frame_length != message_len)
    {
        return false;
    }

    DepthResponsePayload payload{};
    std::memcpy(&payload, message + sizeof(FrameHeader), sizeof(payload));

    size_t depth_count = 0;
    size_t depth_bytes = 0;
    if(!checkedMul(static_cast<size_t>(payload.width), static_cast<size_t>(payload.height), depth_count)
       || !checkedMul(depth_count, sizeof(float), depth_bytes))
    {
        return false;
    }

    const size_t expected_size = sizeof(FrameHeader) + sizeof(DepthResponsePayload) + depth_bytes;
    if(expected_size != message_len)
    {
        return false;
    }

    out_frame.timestampMs = payload.timestamp_ms;
    out_frame.width = payload.width;
    out_frame.height = payload.height;
    out_frame.scale = payload.scale;
    out_frame.bias = payload.bias;
    out_frame.zMax = payload.z_max;
    out_frame.data.resize(depth_count);
    if(depth_bytes != 0)
    {
        std::memcpy(out_frame.data.data(),
                    message + sizeof(FrameHeader) + sizeof(DepthResponsePayload),
                    depth_bytes);
    }

    return true;
}

void drainWakeupFd(int wakeup_fd)
{
    if(wakeup_fd < 0)
    {
        return;
    }

    eventfd_t counter = 0;
    while(eventfd_read(wakeup_fd, &counter) == 0)
    {
    }
}

}  // namespace

TcpDepthClient::TcpDepthClient() = default;

TcpDepthClient::~TcpDepthClient()
{
    disconnect();
}

bool TcpDepthClient::connect(const std::string& host, int port, int timeout_ms)
{
    disconnect();

    m_host = host;
    m_port = port;
    m_connectTimeoutMs = timeout_ms;
    m_responseTimeoutMs = timeout_ms > 0 ? timeout_ms : 5000;
    m_state.store(State::CONNECTING);

    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;

    addrinfo* results = nullptr;
    const std::string port_string = std::to_string(port);
    const int gai_rc = getaddrinfo(host.c_str(), port_string.c_str(), &hints, &results);
    if(gai_rc != 0)
    {
        LOGE("TcpDepthClient: getaddrinfo(%s:%d) failed: %s\n", host.c_str(), port, gai_strerror(gai_rc));
        m_state.store(State::DISCONNECTED);
        return false;
    }

    bool connected = false;
    int socket_fd = -1;
    for(addrinfo* it = results; it != nullptr; it = it->ai_next)
    {
        socket_fd = socket(it->ai_family, it->ai_socktype, it->ai_protocol);
        if(socket_fd < 0)
        {
            continue;
        }

        const int flags = fcntl(socket_fd, F_GETFL, 0);
        if(flags < 0 || fcntl(socket_fd, F_SETFL, flags | O_NONBLOCK) != 0)
        {
            LOGE("TcpDepthClient: failed to configure non-blocking socket: %s\n", std::strerror(errno));
            closeFd(socket_fd);
            continue;
        }

        const int connect_rc = ::connect(socket_fd, it->ai_addr, it->ai_addrlen);
        if(connect_rc == 0)
        {
            connected = true;
            break;
        }

        if(errno != EINPROGRESS)
        {
            LOGW("TcpDepthClient: connect(%s:%d) failed: %s\n", host.c_str(), port, std::strerror(errno));
            closeFd(socket_fd);
            continue;
        }

        pollfd pfd{};
        pfd.fd = socket_fd;
        pfd.events = POLLOUT;
        const int poll_rc = poll(&pfd, 1, timeout_ms);
        if(poll_rc <= 0)
        {
            const char* reason = poll_rc == 0 ? "timed out" : std::strerror(errno);
            LOGW("TcpDepthClient: connect(%s:%d) %s\n", host.c_str(), port, reason);
            closeFd(socket_fd);
            continue;
        }

        int socket_error = 0;
        socklen_t socket_error_len = sizeof(socket_error);
        if(getsockopt(socket_fd, SOL_SOCKET, SO_ERROR, &socket_error, &socket_error_len) != 0)
        {
            LOGE("TcpDepthClient: getsockopt(SO_ERROR) failed: %s\n", std::strerror(errno));
            closeFd(socket_fd);
            continue;
        }

        if(socket_error != 0)
        {
            LOGW("TcpDepthClient: connect(%s:%d) refused/failed: %s\n", host.c_str(), port, std::strerror(socket_error));
            closeFd(socket_fd);
            continue;
        }

        connected = true;
        break;
    }

    freeaddrinfo(results);

    if(!connected || socket_fd < 0)
    {
        m_state.store(State::DISCONNECTED);
        return false;
    }

    m_epollFd = epoll_create1(EPOLL_CLOEXEC);
    if(m_epollFd < 0)
    {
        LOGE("TcpDepthClient: epoll_create1() failed: %s\n", std::strerror(errno));
        closeFd(socket_fd);
        m_state.store(State::DISCONNECTED);
        return false;
    }

    m_wakeupFd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    if(m_wakeupFd < 0)
    {
        LOGE("TcpDepthClient: eventfd() failed: %s\n", std::strerror(errno));
        closeFd(m_epollFd);
        closeFd(socket_fd);
        m_state.store(State::DISCONNECTED);
        return false;
    }

    epoll_event socket_event{};
    socket_event.events = EPOLLIN | EPOLLRDHUP | EPOLLERR;
    socket_event.data.fd = socket_fd;
    if(epoll_ctl(m_epollFd, EPOLL_CTL_ADD, socket_fd, &socket_event) != 0)
    {
        LOGE("TcpDepthClient: epoll_ctl(socket) failed: %s\n", std::strerror(errno));
        closeFd(m_wakeupFd);
        closeFd(m_epollFd);
        closeFd(socket_fd);
        m_state.store(State::DISCONNECTED);
        return false;
    }

    epoll_event wake_event{};
    wake_event.events = EPOLLIN | EPOLLERR;
    wake_event.data.fd = m_wakeupFd;
    if(epoll_ctl(m_epollFd, EPOLL_CTL_ADD, m_wakeupFd, &wake_event) != 0)
    {
        LOGE("TcpDepthClient: epoll_ctl(wakeup) failed: %s\n", std::strerror(errno));
        closeFd(m_wakeupFd);
        closeFd(m_epollFd);
        closeFd(socket_fd);
        m_state.store(State::DISCONNECTED);
        return false;
    }

    m_socketFd = socket_fd;
    m_lastIoActivityMs = nowMs();
    m_inFlightRequests = 0;
    {
        std::scoped_lock lock(m_mutex);
        m_lastError.clear();
    }
    m_state.store(State::CONNECTED);
    m_ioThread = std::thread(&TcpDepthClient::ioThreadFunc, this);
    LOGI("TcpDepthClient: connected to %s:%d\n", host.c_str(), port);
    return true;
}

void TcpDepthClient::disconnect()
{
    const bool had_thread = m_ioThread.joinable();
    State previous_state = m_state.exchange(State::DISCONNECTING);
    if(previous_state != State::DISCONNECTED || had_thread)
    {
        if(m_wakeupFd >= 0)
        {
            const eventfd_t wake_value = 1;
            const ssize_t rc = write(m_wakeupFd, &wake_value, sizeof(wake_value));
            (void)rc;
        }
    }

    if(m_ioThread.joinable())
    {
        m_ioThread.join();
    }

    closeFd(m_socketFd);
    closeFd(m_wakeupFd);
    closeFd(m_epollFd);

    {
        std::scoped_lock lock(m_mutex);
        m_pendingRequests.clear();
    }

    m_inFlightRequests = 0;
    m_lastIoActivityMs = 0;
    m_state.store(State::DISCONNECTED);
}

void TcpDepthClient::sendFrameRequest(uint32_t frame_index,
                                      uint32_t timestamp_ms,
                                      const uint8_t* rgb_data,
                                      uint32_t width,
                                      uint32_t height)
{
    if(m_state.load() != State::CONNECTED)
    {
        LOGD("TcpDepthClient: dropping frame request while not connected\n");
        return;
    }

    auto request = TcpProtocolSerializer::serializeFrameRequest(frame_index, timestamp_ms, width, height, rgb_data);
    if(request.empty())
    {
        LOGE("TcpDepthClient: failed to serialize frame request %u\n", frame_index);
        return;
    }

    {
        std::scoped_lock lock(m_mutex);
        m_pendingRequests.push_back(std::move(request));
    }

    if(m_wakeupFd >= 0)
    {
        const eventfd_t wake_value = 1;
        const ssize_t rc = write(m_wakeupFd, &wake_value, sizeof(wake_value));
        if(rc < 0 && errno != EAGAIN)
        {
            LOGW("TcpDepthClient: failed to wake I/O thread: %s\n", std::strerror(errno));
        }
    }
}

void TcpDepthClient::setDepthFrameCallback(DepthFrameCallback cb)
{
    std::scoped_lock lock(m_mutex);
    m_callback = std::move(cb);
}

void TcpDepthClient::update()
{
    const State current_state = m_state.load();
    if(current_state != m_lastLoggedState)
    {
        LOGI("TcpDepthClient: state %s -> %s\n", toString(m_lastLoggedState), toString(current_state));
        m_lastLoggedState = current_state;
    }
}

TcpDepthClient::State TcpDepthClient::getState() const
{
    return m_state.load();
}

bool TcpDepthClient::isConnected() const
{
    return m_state.load() == State::CONNECTED;
}

void TcpDepthClient::ioThreadFunc()
{
    TcpProtocolParser parser;
    epoll_event events[kMaxEpollEvents]{};

    auto disconnectWithError = [this](const char* context, int error_code) {
        const char* error_text = error_code != 0 ? std::strerror(error_code) : "connection closed";
        {
            std::scoped_lock lock(m_mutex);
            m_lastError = std::string(context) + ": " + error_text;
        }
        LOGW("TcpDepthClient: %s: %s\n", context, error_text);
        m_state.store(State::DISCONNECTED);
        closeFd(m_socketFd);
    };

    while(m_state.load() == State::CONNECTED || m_state.load() == State::DISCONNECTING)
    {
        if(m_state.load() == State::DISCONNECTING)
        {
            break;
        }

        std::deque<std::vector<uint8_t>> local_requests;
        {
            std::scoped_lock lock(m_mutex);
            local_requests.swap(m_pendingRequests);
        }

        while(!local_requests.empty())
        {
            const std::vector<uint8_t>& request = local_requests.front();
            if(!writeAll(m_socketFd, request.data(), request.size(), m_responseTimeoutMs))
            {
                disconnectWithError("send failed", errno);
                break;
            }

            local_requests.pop_front();
            ++m_inFlightRequests;
            m_lastIoActivityMs = nowMs();
        }

        if(m_state.load() != State::CONNECTED)
        {
            break;
        }

        const int event_count = epoll_wait(m_epollFd, events, kMaxEpollEvents, kIoPollTimeoutMs);
        if(event_count < 0)
        {
            if(errno == EINTR)
            {
                continue;
            }
            disconnectWithError("epoll_wait failed", errno);
            break;
        }

        if(event_count == 0)
        {
            if(m_inFlightRequests > 0 && m_responseTimeoutMs > 0 && nowMs() - m_lastIoActivityMs > static_cast<uint64_t>(m_responseTimeoutMs))
            {
                disconnectWithError("response timeout", ETIMEDOUT);
                break;
            }
            continue;
        }

        for(int i = 0; i < event_count; ++i)
        {
            const epoll_event& event = events[i];
            if(event.data.fd == m_wakeupFd)
            {
                drainWakeupFd(m_wakeupFd);
                continue;
            }

            if(event.data.fd != m_socketFd)
            {
                continue;
            }

            if((event.events & (EPOLLERR | EPOLLHUP | EPOLLRDHUP)) != 0)
            {
                disconnectWithError("server disconnected", 0);
                break;
            }

            if((event.events & EPOLLIN) == 0)
            {
                continue;
            }

            bool peer_closed = false;
            if(!readSocketData(m_socketFd, parser, peer_closed))
            {
                disconnectWithError(peer_closed ? "server disconnected" : "recv failed", peer_closed ? 0 : errno);
                break;
            }

            m_lastIoActivityMs = nowMs();

            while(parser.hasMessage())
            {
                size_t message_len = 0;
                uint32_t message_type = 0;
                const uint8_t* message = parser.nextMessage(message_len, message_type);
                if(message == nullptr)
                {
                    break;
                }

                if(message_type == MSG_DEPTH_RESPONSE)
                {
                    DepthFrame frame;
                    if(!decodeDepthResponse(message, message_len, frame))
                    {
                        disconnectWithError("invalid depth response", EPROTO);
                        break;
                    }

                    if(m_inFlightRequests > 0)
                    {
                        --m_inFlightRequests;
                    }

                    DepthFrameCallback callback;
                    {
                        std::scoped_lock lock(m_mutex);
                        callback = m_callback;
                    }
                    if(callback)
                    {
                        callback(frame);
                    }
                }
                else if(message_type == MSG_ERROR)
                {
                    ErrorPayload payload{};
                    if(message_len >= sizeof(FrameHeader) + sizeof(ErrorPayload))
                    {
                        std::memcpy(&payload, message + sizeof(FrameHeader), sizeof(payload));
                        LOGW("TcpDepthClient: server error %u: %s\n", payload.error_code, payload.error_msg);
                    }
                }
                else if(message_type == MSG_SERVER_STATUS)
                {
                    LOGD("TcpDepthClient: received SERVER_STATUS\n");
                }
                else if(message_type == MSG_SHUTDOWN_ACK)
                {
                    LOGI("TcpDepthClient: received SHUTDOWN_ACK\n");
                }
                else
                {
                    LOGW("TcpDepthClient: ignoring unexpected message type %s\n", getMessageTypeName(message_type));
                }
            }

            if(m_state.load() != State::CONNECTED)
            {
                break;
            }
        }
    }

    closeFd(m_socketFd);
}

#endif  // WITH_TCP_DEPTH
