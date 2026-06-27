#include "free_splatter_client.h"

#ifdef WITH_FREE_SPLATTER

#include <nvutils/logger.hpp>

#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <netdb.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>

namespace vk_viewer {

namespace {

// Protocol constants — replicated locally to avoid a depth_server header
// dependency in vk_viewer. Must stay in sync with depth_server/protocol.h.
constexpr uint32_t TCP_DEPTH_MAGIC = 0x44455054u;
constexpr uint32_t MAX_FRAME_SIZE  = 100u * 1024u * 1024u;
constexpr uint32_t MAX_IMAGE_SIZE  = 16u * 1024u * 1024u;

constexpr uint32_t MSG_SPLAT_REQUEST  = 0x10u;
constexpr uint32_t MSG_SPLAT_POLL     = 0x11u;
constexpr uint32_t MSG_SPLAT_PROGRESS = 0x12u;
constexpr uint32_t MSG_SPLAT_RESPONSE = 0x13u;
constexpr uint32_t MSG_SPLAT_CANCEL   = 0x14u;
constexpr uint32_t MSG_SPLAT_ERROR    = 0x15u;
constexpr uint32_t MSG_ERROR          = 0xFFu;

#pragma pack(push, 1)
struct FrameHeader
{
  uint32_t magic;
  uint32_t frame_length;
  uint32_t message_type;
};

struct SplatRequestOptions
{
  float    downsample;
  uint8_t  sh_degree;
  uint8_t  variant;
  uint8_t  _pad[2];
  uint32_t timeout_ms;
  uint32_t flags;
};

struct SplatRequestPayload
{
  uint32_t n_views;
  uint32_t width;
  uint32_t height;
  uint32_t options_size;
  uint32_t image_data_size;
};

struct SplatPollPayload
{
  uint32_t job_id;
};

struct SplatProgressPayload
{
  uint32_t job_id;
  uint8_t  stage;
  uint8_t  percent;
  uint16_t _pad;
  uint32_t eta_ms;
};

struct SplatResponsePayload
{
  uint32_t job_id;
  uint32_t n_gaussians;
  uint32_t path_length;
};

struct SplatCancelPayload
{
  uint32_t job_id;
};

struct SplatErrorPayload
{
  uint32_t job_id;
  uint32_t error_code;
  char     error_msg[252];
};

struct ErrorPayload
{
  uint32_t error_code;
  char     error_msg[256];
};
#pragma pack(pop)

static_assert(sizeof(FrameHeader) == 12, "FrameHeader must be 12 bytes");
static_assert(sizeof(SplatRequestOptions) == 16, "SplatRequestOptions must be 16 bytes");
static_assert(sizeof(SplatRequestPayload) == 20, "SplatRequestPayload must be 20 bytes");
static_assert(sizeof(SplatPollPayload) == 4, "SplatPollPayload must be 4 bytes");
static_assert(sizeof(SplatProgressPayload) == 12, "SplatProgressPayload must be 12 bytes");
static_assert(sizeof(SplatResponsePayload) == 12, "SplatResponsePayload must be 12 bytes");
static_assert(sizeof(SplatCancelPayload) == 4, "SplatCancelPayload must be 4 bytes");
static_assert(sizeof(SplatErrorPayload) == 260, "SplatErrorPayload must be 260 bytes");

uint64_t nowMs()
{
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

bool waitForEvent(int fd, short events, int timeout_ms)
{
  pollfd pfd{};
  pfd.fd     = fd;
  pfd.events = events;
  for(;;)
  {
    const int rc = poll(&pfd, 1, timeout_ms);
    if(rc > 0)
    {
      if((pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0)
        return false;
      return (pfd.revents & events) != 0;
    }
    if(rc == 0)
      return false;
    if(errno != EINTR)
      return false;
  }
}

}  // namespace

FreeSplatterClient::TcpSocket::~TcpSocket()
{
  close();
}

bool FreeSplatterClient::TcpSocket::connect(const std::string& host, int port, int timeout_ms)
{
  close();

  addrinfo hints{};
  hints.ai_family   = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_protocol = IPPROTO_TCP;

  addrinfo* results = nullptr;
  const std::string port_string = std::to_string(port);
  const int         gai_rc      = getaddrinfo(host.c_str(), port_string.c_str(), &hints, &results);
  if(gai_rc != 0)
  {
    LOGE("FreeSplatterClient: getaddrinfo(%s:%d) failed: %s\n", host.c_str(), port, gai_strerror(gai_rc));
    return false;
  }

  bool connected = false;
  for(addrinfo* it = results; it != nullptr; it = it->ai_next)
  {
    fd = socket(it->ai_family, it->ai_socktype, it->ai_protocol);
    if(fd < 0)
      continue;

    const int flags = fcntl(fd, F_GETFL, 0);
    if(flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) != 0)
    {
      LOGE("FreeSplatterClient: fcntl(O_NONBLOCK) failed: %s\n", std::strerror(errno));
      close();
      continue;
    }

    const int connect_rc = ::connect(fd, it->ai_addr, it->ai_addrlen);
    if(connect_rc == 0)
    {
      connected = true;
      break;
    }

    if(errno != EINPROGRESS)
    {
      LOGW("FreeSplatterClient: connect(%s:%d) failed: %s\n", host.c_str(), port, std::strerror(errno));
      close();
      continue;
    }

    if(!waitForEvent(fd, POLLOUT, timeout_ms))
    {
      LOGW("FreeSplatterClient: connect(%s:%d) timed out\n", host.c_str(), port);
      close();
      continue;
    }

    int socket_error = 0;
    socklen_t err_len = sizeof(socket_error);
    if(getsockopt(fd, SOL_SOCKET, SO_ERROR, &socket_error, &err_len) != 0 || socket_error != 0)
    {
      LOGW("FreeSplatterClient: connect(%s:%d) refused: %s\n", host.c_str(), port,
           socket_error != 0 ? std::strerror(socket_error) : std::strerror(errno));
      close();
      continue;
    }

    connected = true;
    break;
  }

  freeaddrinfo(results);

  if(!connected || fd < 0)
    return false;

  int yes = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));
  return true;
}

bool FreeSplatterClient::TcpSocket::sendAll(const uint8_t* data, size_t len, int timeout_ms)
{
  if(fd < 0)
    return false;

  const uint64_t deadline = nowMs() + static_cast<uint64_t>(std::max(timeout_ms, 0));
  while(len > 0)
  {
    const ssize_t n = send(fd, data, len, MSG_NOSIGNAL);
    if(n > 0)
    {
      data += n;
      len -= static_cast<size_t>(n);
      continue;
    }
    if(n == 0)
      return false;

    if(errno == EINTR)
      continue;

    if(errno != EAGAIN && errno != EWOULDBLOCK)
    {
      LOGE("FreeSplatterClient: send() failed: %s\n", std::strerror(errno));
      return false;
    }

    const uint64_t current = nowMs();
    if(current >= deadline)
    {
      LOGE("FreeSplatterClient: send() timed out\n");
      return false;
    }
    if(!waitForEvent(fd, POLLOUT, static_cast<int>(deadline - current)))
    {
      LOGE("FreeSplatterClient: send() poll timed out\n");
      return false;
    }
  }
  return true;
}

bool FreeSplatterClient::TcpSocket::recvAll(uint8_t* buf, size_t len, int timeout_ms)
{
  if(fd < 0)
    return false;

  const uint64_t deadline = nowMs() + static_cast<uint64_t>(std::max(timeout_ms, 0));
  while(len > 0)
  {
    const uint64_t current = nowMs();
    if(current >= deadline)
    {
      LOGE("FreeSplatterClient: recv() timed out\n");
      return false;
    }
    if(!waitForEvent(fd, POLLIN, static_cast<int>(deadline - current)))
    {
      LOGE("FreeSplatterClient: recv() poll timed out\n");
      return false;
    }

    const ssize_t n = recv(fd, buf, len, 0);
    if(n > 0)
    {
      buf += n;
      len -= static_cast<size_t>(n);
      continue;
    }
    if(n == 0)
    {
      LOGE("FreeSplatterClient: recv() peer closed connection\n");
      return false;
    }
    if(errno == EINTR)
      continue;
    if(errno != EAGAIN && errno != EWOULDBLOCK)
    {
      LOGE("FreeSplatterClient: recv() failed: %s\n", std::strerror(errno));
      return false;
    }
  }
  return true;
}

void FreeSplatterClient::TcpSocket::close()
{
  if(fd >= 0)
  {
    ::close(fd);
    fd = -1;
  }
}

bool FreeSplatterClient::sendFrame(TcpSocket& sock, uint32_t type, const uint8_t* payload, size_t payload_size)
{
  const size_t total = sizeof(FrameHeader) + payload_size;
  if(total > MAX_FRAME_SIZE || total > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
  {
    LOGE("FreeSplatterClient: frame too large (%zu bytes)\n", total);
    return false;
  }

  FrameHeader header{};
  header.magic         = TCP_DEPTH_MAGIC;
  header.frame_length  = static_cast<uint32_t>(total);
  header.message_type  = type;

  if(!sock.sendAll(reinterpret_cast<const uint8_t*>(&header), sizeof(header)))
    return false;

  if(payload_size != 0 && payload != nullptr)
  {
    if(!sock.sendAll(payload, payload_size))
      return false;
  }
  return true;
}

bool FreeSplatterClient::recvFrame(TcpSocket& sock, uint32_t& out_type, std::vector<uint8_t>& out_payload, int timeout_ms)
{
  FrameHeader header{};
  if(!sock.recvAll(reinterpret_cast<uint8_t*>(&header), sizeof(header), timeout_ms))
    return false;

  if(header.magic != TCP_DEPTH_MAGIC)
  {
    LOGE("FreeSplatterClient: bad magic 0x%08x\n", header.magic);
    return false;
  }
  if(header.frame_length < sizeof(FrameHeader) || header.frame_length > MAX_FRAME_SIZE)
  {
    LOGE("FreeSplatterClient: bad frame_length %u\n", header.frame_length);
    return false;
  }

  const size_t payload_size = header.frame_length - sizeof(FrameHeader);
  out_payload.resize(payload_size);
  if(payload_size != 0)
  {
    if(!sock.recvAll(out_payload.data(), payload_size, timeout_ms))
      return false;
  }
  out_type = header.message_type;
  return true;
}

FreeSplatterClient::FreeSplatterClient() = default;

FreeSplatterClient::~FreeSplatterClient()
{
  close();
}

void FreeSplatterClient::setServerAddress(const std::string& host, int port)
{
  m_host = host;
  m_port = port;
}

void FreeSplatterClient::setLocalDownloadDir(const std::filesystem::path& dir)
{
  m_downloadDir = dir;
}

void FreeSplatterClient::testConnection(ConnectionCallback callback)
{
  std::thread([this, callback = std::move(callback)]() {
    ConnectionStatus status;
    TcpSocket        sock;
    if(sock.connect(m_host, m_port, 5000))
    {
      status.reachable = true;
    }
    else
    {
      status.reachable = false;
      status.error     = "Connection to " + m_host + ":" + std::to_string(m_port) + " failed";
    }
    sock.close();
    if(callback)
      callback(status);
  }).detach();
}

uint32_t FreeSplatterClient::submitJob(const std::vector<std::filesystem::path>& image_paths,
                                       StatusCallback                            on_update,
                                       CompleteCallback                          on_complete)
{
  if(image_paths.size() < 2 || image_paths.size() > 4)
  {
    LOGE("FreeSplatterClient: submitJob requires 2-4 images, got %zu\n", image_paths.size());
    return 0;
  }

  std::vector<std::vector<uint8_t>> images;
  images.reserve(image_paths.size());
  size_t image_data_size = 0;
  for(const auto& path : image_paths)
  {
    std::ifstream file(path, std::ios::binary);
    if(!file)
    {
      LOGE("FreeSplatterClient: cannot open image %s\n", path.string().c_str());
      return 0;
    }
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if(bytes.empty())
    {
      LOGE("FreeSplatterClient: empty image %s\n", path.string().c_str());
      return 0;
    }
    if(bytes.size() > MAX_IMAGE_SIZE)
    {
      LOGE("FreeSplatterClient: image %s exceeds %u bytes\n", path.string().c_str(), MAX_IMAGE_SIZE);
      return 0;
    }
    image_data_size += sizeof(uint32_t) + bytes.size();
    images.emplace_back(std::move(bytes));
  }

  TcpSocket sock;
  if(!sock.connect(m_host, m_port, 10000))
  {
    LOGE("FreeSplatterClient: cannot connect to %s:%d\n", m_host.c_str(), m_port);
    return 0;
  }

  SplatRequestOptions options{};
  options.downsample  = 1.0f;
  options.sh_degree   = 0;
  options.variant     = 0;
  options.timeout_ms  = 0;
  options.flags       = 0;

  const uint32_t n_views = static_cast<uint32_t>(images.size());

  std::vector<uint8_t> payload;
  payload.reserve(sizeof(SplatRequestPayload) + sizeof(SplatRequestOptions) + image_data_size);

  SplatRequestPayload req_header{};
  req_header.n_views         = n_views;
  req_header.width           = 0;
  req_header.height          = 0;
  req_header.options_size    = sizeof(SplatRequestOptions);
  req_header.image_data_size = static_cast<uint32_t>(image_data_size);

  payload.insert(payload.end(),
                 reinterpret_cast<const uint8_t*>(&req_header),
                 reinterpret_cast<const uint8_t*>(&req_header) + sizeof(req_header));
  payload.insert(payload.end(),
                 reinterpret_cast<const uint8_t*>(&options),
                 reinterpret_cast<const uint8_t*>(&options) + sizeof(options));

  for(const auto& img : images)
  {
    uint32_t len = static_cast<uint32_t>(img.size());
    payload.insert(payload.end(),
                   reinterpret_cast<const uint8_t*>(&len),
                   reinterpret_cast<const uint8_t*>(&len) + sizeof(len));
    payload.insert(payload.end(), img.begin(), img.end());
  }

  if(!sendFrame(sock, MSG_SPLAT_REQUEST, payload.data(), payload.size()))
  {
    LOGE("FreeSplatterClient: failed to send MSG_SPLAT_REQUEST\n");
    return 0;
  }

  uint32_t               resp_type = 0;
  std::vector<uint8_t>   resp_payload;
  if(!recvFrame(sock, resp_type, resp_payload, 30000))
  {
    LOGE("FreeSplatterClient: no response to MSG_SPLAT_REQUEST\n");
    return 0;
  }

  uint32_t job_id = 0;
  if(resp_type == MSG_SPLAT_PROGRESS && resp_payload.size() >= sizeof(SplatProgressPayload))
  {
    SplatProgressPayload prog{};
    std::memcpy(&prog, resp_payload.data(), sizeof(prog));
    job_id = prog.job_id;
  }
  else if(resp_type == MSG_SPLAT_ERROR && resp_payload.size() >= sizeof(SplatErrorPayload))
  {
    SplatErrorPayload err{};
    std::memcpy(&err, resp_payload.data(), sizeof(err));
    LOGE("FreeSplatterClient: server rejected job: code=%u msg=%s\n", err.error_code, err.error_msg);
    return 0;
  }
  else if(resp_type == MSG_ERROR && resp_payload.size() >= sizeof(ErrorPayload))
  {
    ErrorPayload err{};
    std::memcpy(&err, resp_payload.data(), sizeof(err));
    LOGE("FreeSplatterClient: server error: code=%u msg=%s\n", err.error_code, err.error_msg);
    return 0;
  }
  else
  {
    LOGE("FreeSplatterClient: unexpected response type 0x%08x to REQUEST\n", resp_type);
    return 0;
  }

  if(job_id == 0)
  {
    LOGE("FreeSplatterClient: server returned job_id=0\n");
    return 0;
  }

  sock.close();

  {
    std::lock_guard<std::mutex> lock(m_jobsMutex);
    auto& job = m_activeJobs[job_id];
    job.on_update   = std::move(on_update);
    job.on_complete = std::move(on_complete);
    job.poll_thread = std::thread(&FreeSplatterClient::pollLoop, this, job_id, job.on_update, job.on_complete);
  }

  return job_id;
}

void FreeSplatterClient::pollLoop(uint32_t job_id, StatusCallback on_update, CompleteCallback on_complete)
{
  constexpr int kPollIntervalMs = 500;
  constexpr int kRecvTimeoutMs  = 30000;

  std::string error_msg;
  std::string output_path;
  int         n_gaussians = 0;
  bool        done        = false;
  bool        errored     = false;

  while(m_shouldRun.load() && !done && !errored)
  {
    bool cancel_flag = false;
    {
      std::lock_guard<std::mutex> lock(m_jobsMutex);
      auto it = m_activeJobs.find(job_id);
      if(it == m_activeJobs.end())
        return;
      cancel_flag = it->second.cancel_requested.load();
    }

    TcpSocket sock;
    if(!sock.connect(m_host, m_port, 5000))
    {
      LOGW("FreeSplatterClient: poll connect failed, retrying\n");
      if(!m_shouldRun.load())
        break;
      std::this_thread::sleep_for(std::chrono::milliseconds(kPollIntervalMs));
      continue;
    }

    if(cancel_flag)
    {
      SplatCancelPayload cancel{};
      cancel.job_id = job_id;
      if(!sendFrame(sock, MSG_SPLAT_CANCEL, reinterpret_cast<const uint8_t*>(&cancel), sizeof(cancel)))
      {
        LOGW("FreeSplatterClient: failed to send CANCEL\n");
      }
    }
    else
    {
      SplatPollPayload poll{};
      poll.job_id = job_id;
      if(!sendFrame(sock, MSG_SPLAT_POLL, reinterpret_cast<const uint8_t*>(&poll), sizeof(poll)))
      {
        LOGW("FreeSplatterClient: failed to send POLL\n");
        sock.close();
        if(!m_shouldRun.load())
          break;
        std::this_thread::sleep_for(std::chrono::milliseconds(kPollIntervalMs));
        continue;
      }
    }

    uint32_t             resp_type = 0;
    std::vector<uint8_t> resp_payload;
    if(!recvFrame(sock, resp_type, resp_payload, kRecvTimeoutMs))
    {
      LOGW("FreeSplatterClient: no response to POLL\n");
      sock.close();
      if(!m_shouldRun.load())
        break;
      std::this_thread::sleep_for(std::chrono::milliseconds(kPollIntervalMs));
      continue;
    }
    sock.close();

    switch(resp_type)
    {
    case MSG_SPLAT_PROGRESS:
      if(resp_payload.size() >= sizeof(SplatProgressPayload))
      {
        SplatProgressPayload prog{};
        std::memcpy(&prog, resp_payload.data(), sizeof(prog));
        JobStatus status;
        switch(prog.stage)
        {
        case 0: status.state = JobStatus::State::Queued; break;
        case 1: status.state = JobStatus::State::Preprocessing; break;
        case 2: status.state = JobStatus::State::Inference; break;
        case 3: status.state = JobStatus::State::Writing; break;
        case 4: status.state = JobStatus::State::Done; break;
        default: status.state = JobStatus::State::Queued; break;
        }
        status.percent = prog.percent;
        status.eta_ms  = static_cast<int>(prog.eta_ms);
        if(on_update)
          on_update(job_id, status);
      }
      break;

    case MSG_SPLAT_RESPONSE:
      if(resp_payload.size() >= sizeof(SplatResponsePayload))
      {
        SplatResponsePayload resp{};
        std::memcpy(&resp, resp_payload.data(), sizeof(resp));
        n_gaussians = static_cast<int>(resp.n_gaussians);
        if(resp.path_length > 0 && sizeof(SplatResponsePayload) + resp.path_length <= resp_payload.size())
        {
          output_path.assign(reinterpret_cast<const char*>(resp_payload.data() + sizeof(SplatResponsePayload)),
                             resp.path_length);
          if(!output_path.empty() && output_path.back() == '\0')
            output_path.pop_back();
        }
        done = true;
      }
      break;

    case MSG_SPLAT_ERROR:
      if(resp_payload.size() >= sizeof(SplatErrorPayload))
      {
        SplatErrorPayload err{};
        std::memcpy(&err, resp_payload.data(), sizeof(err));
        error_msg.assign(err.error_msg, strnlen(err.error_msg, sizeof(err.error_msg)));
        errored = true;
      }
      break;

    case MSG_ERROR:
      if(resp_payload.size() >= sizeof(ErrorPayload))
      {
        ErrorPayload err{};
        std::memcpy(&err, resp_payload.data(), sizeof(err));
        error_msg.assign(err.error_msg, strnlen(err.error_msg, sizeof(err.error_msg)));
        errored = true;
      }
      break;

    default:
      LOGW("FreeSplatterClient: unexpected poll response 0x%08x\n", resp_type);
      break;
    }

    if(!done && !errored && m_shouldRun.load())
      std::this_thread::sleep_for(std::chrono::milliseconds(kPollIntervalMs));
  }

  if(on_complete)
  {
    std::string local_path;
    std::string complete_error;
    if(errored)
    {
      complete_error = error_msg.empty() ? "Job failed" : error_msg;
    }
    else if(done)
    {
      if(output_path.empty())
      {
        complete_error = "Server returned empty output path";
      }
      else if(std::filesystem::exists(output_path))
      {
        local_path = output_path;
      }
      else if(!m_downloadDir.empty())
      {
        std::error_code ec;
        auto dest = m_downloadDir / std::filesystem::path(output_path).filename();
        std::filesystem::copy_file(output_path, dest, std::filesystem::copy_options::overwrite_existing, ec);
        if(ec)
        {
          complete_error = "Cannot copy " + output_path + " to " + dest.string() + ": " + ec.message();
        }
        else
        {
          local_path = dest.string();
        }
      }
      else
      {
        complete_error = "Output path not accessible locally: " + output_path;
      }
    }
    on_complete(job_id, local_path, complete_error);
  }

  {
    std::lock_guard<std::mutex> lock(m_jobsMutex);
    auto it = m_activeJobs.find(job_id);
    if(it != m_activeJobs.end())
    {
      if(it->second.poll_thread.joinable())
      {
        if(it->second.poll_thread.get_id() != std::this_thread::get_id())
          it->second.poll_thread.join();
        else
          it->second.poll_thread.detach();
      }
      m_activeJobs.erase(it);
    }
  }
}

void FreeSplatterClient::cancelJob(uint32_t job_id)
{
  {
    std::lock_guard<std::mutex> lock(m_jobsMutex);
    auto it = m_activeJobs.find(job_id);
    if(it != m_activeJobs.end())
      it->second.cancel_requested.store(true);
  }

  TcpSocket sock;
  if(!sock.connect(m_host, m_port, 5000))
  {
    LOGW("FreeSplatterClient: cancelJob connect failed\n");
    return;
  }
  SplatCancelPayload cancel{};
  cancel.job_id = job_id;
  sendFrame(sock, MSG_SPLAT_CANCEL, reinterpret_cast<const uint8_t*>(&cancel), sizeof(cancel));
}

void FreeSplatterClient::close()
{
  m_shouldRun.store(false);

  std::vector<std::thread> threads_to_join;
  {
    std::lock_guard<std::mutex> lock(m_jobsMutex);
    for(auto& [id, job] : m_activeJobs)
    {
      if(job.poll_thread.joinable())
        threads_to_join.emplace_back(std::move(job.poll_thread));
    }
    m_activeJobs.clear();
  }

  for(auto& t : threads_to_join)
  {
    if(t.joinable())
      t.join();
  }
}

}  // namespace vk_viewer

#endif  // WITH_FREE_SPLATTER
