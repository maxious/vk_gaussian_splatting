#include "cloud_client.h"

#ifdef WITH_TCP_DEPTH

#include <nvutils/logger.hpp>

#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>

namespace vk_viewer {

namespace {

constexpr uint32_t TCP_DEPTH_MAGIC = 0x44455054u;
constexpr uint32_t MAX_FRAME_SIZE  = 100u * 1024u * 1024u;
constexpr uint32_t MAX_CLOUD_FRAMES = 200u;
constexpr uint32_t MAX_CLOUD_PATH_LEN = 4096u;

constexpr uint32_t MSG_CLOUD_REQUEST  = 0x06u;
constexpr uint32_t MSG_CLOUD_POLL     = 0x07u;
constexpr uint32_t MSG_CLOUD_PROGRESS = 0x08u;
constexpr uint32_t MSG_CLOUD_RESPONSE = 0x09u;
constexpr uint32_t MSG_CLOUD_CANCEL   = 0x0Au;
constexpr uint32_t MSG_CLOUD_ERROR    = 0x0Bu;
constexpr uint32_t MSG_ERROR          = 0xFFu;

constexpr uint32_t kDefaultFps          = 6u;
constexpr uint32_t kDefaultMaxDimension = 640u;

#pragma pack(push, 1)
struct FrameHeader
{
  uint32_t magic;
  uint32_t frame_length;
  uint32_t message_type;
};

struct CloudRequestPayload
{
  uint32_t n_frames;
  uint32_t options_size;
  uint32_t frame_data_size;
};

struct CloudPollPayload
{
  uint32_t job_id;
};

struct CloudProgressPayload
{
  uint32_t job_id;
  uint8_t  stage;
  uint8_t  percent;
  uint16_t windows_done;
  uint16_t windows_total;
  uint16_t _pad;
  uint32_t eta_ms;
};

struct CloudResponsePayload
{
  uint32_t job_id;
  uint32_t n_points;
  uint32_t path_length;
};

struct CloudCancelPayload
{
  uint32_t job_id;
};

struct CloudErrorPayload
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
static_assert(sizeof(CloudRequestOptions) == 64, "CloudRequestOptions must be 64 bytes");
static_assert(sizeof(CloudRequestPayload) == 12, "CloudRequestPayload must be 12 bytes");
static_assert(sizeof(CloudPollPayload) == 4, "CloudPollPayload must be 4 bytes");
static_assert(sizeof(CloudProgressPayload) == 16, "CloudProgressPayload must be 16 bytes");
static_assert(sizeof(CloudResponsePayload) == 12, "CloudResponsePayload must be 12 bytes");
static_assert(sizeof(CloudCancelPayload) == 4, "CloudCancelPayload must be 4 bytes");
static_assert(sizeof(CloudErrorPayload) == 260, "CloudErrorPayload must be 260 bytes");

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

CloudClient::TcpSocket::~TcpSocket()
{
  close();
}

bool CloudClient::TcpSocket::connect(const std::string& host, int port, int timeout_ms)
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
    LOGE("CloudClient: getaddrinfo(%s:%d) failed: %s\n", host.c_str(), port, gai_strerror(gai_rc));
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
      LOGE("CloudClient: fcntl(O_NONBLOCK) failed: %s\n", std::strerror(errno));
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
      LOGW("CloudClient: connect(%s:%d) failed: %s\n", host.c_str(), port, std::strerror(errno));
      close();
      continue;
    }

    if(!waitForEvent(fd, POLLOUT, timeout_ms))
    {
      LOGW("CloudClient: connect(%s:%d) timed out\n", host.c_str(), port);
      close();
      continue;
    }

    int socket_error = 0;
    socklen_t err_len = sizeof(socket_error);
    if(getsockopt(fd, SOL_SOCKET, SO_ERROR, &socket_error, &err_len) != 0 || socket_error != 0)
    {
      LOGW("CloudClient: connect(%s:%d) refused: %s\n", host.c_str(), port,
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

bool CloudClient::TcpSocket::sendAll(const uint8_t* data, size_t len, int timeout_ms)
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
      LOGE("CloudClient: send() failed: %s\n", std::strerror(errno));
      return false;
    }

    const uint64_t current = nowMs();
    if(current >= deadline)
    {
      LOGE("CloudClient: send() timed out\n");
      return false;
    }
    if(!waitForEvent(fd, POLLOUT, static_cast<int>(deadline - current)))
    {
      LOGE("CloudClient: send() poll timed out\n");
      return false;
    }
  }
  return true;
}

bool CloudClient::TcpSocket::recvAll(uint8_t* buf, size_t len, int timeout_ms)
{
  if(fd < 0)
    return false;

  const uint64_t deadline = nowMs() + static_cast<uint64_t>(std::max(timeout_ms, 0));
  while(len > 0)
  {
    const uint64_t current = nowMs();
    if(current >= deadline)
    {
      LOGE("CloudClient: recv() timed out\n");
      return false;
    }
    if(!waitForEvent(fd, POLLIN, static_cast<int>(deadline - current)))
    {
      LOGE("CloudClient: recv() poll timed out\n");
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
      LOGE("CloudClient: recv() peer closed connection\n");
      return false;
    }
    if(errno == EINTR)
      continue;
    if(errno != EAGAIN && errno != EWOULDBLOCK)
    {
      LOGE("CloudClient: recv() failed: %s\n", std::strerror(errno));
      return false;
    }
  }
  return true;
}

void CloudClient::TcpSocket::close()
{
  if(fd >= 0)
  {
    ::close(fd);
    fd = -1;
  }
}

bool CloudClient::sendFrame(TcpSocket& sock, uint32_t type, const uint8_t* payload, size_t payload_size)
{
  const size_t total = sizeof(FrameHeader) + payload_size;
  if(total > MAX_FRAME_SIZE || total > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
  {
    LOGE("CloudClient: frame too large (%zu bytes)\n", total);
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

bool CloudClient::recvFrame(TcpSocket& sock, uint32_t& out_type, std::vector<uint8_t>& out_payload, int timeout_ms)
{
  FrameHeader header{};
  if(!sock.recvAll(reinterpret_cast<uint8_t*>(&header), sizeof(header), timeout_ms))
    return false;

  if(header.magic != TCP_DEPTH_MAGIC)
  {
    LOGE("CloudClient: bad magic 0x%08x\n", header.magic);
    return false;
  }
  if(header.frame_length < sizeof(FrameHeader) || header.frame_length > MAX_FRAME_SIZE)
  {
    LOGE("CloudClient: bad frame_length %u\n", header.frame_length);
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

bool CloudClient::extractFrames(const std::string& video_path,
                                const std::string& tmpdir,
                                uint32_t max_frames,
                                uint32_t fps,
                                uint32_t max_dimension,
                                std::vector<std::string>& out_paths)
{
  out_paths.clear();

  char cmd[2048];
  std::snprintf(cmd, sizeof(cmd),
                "ffmpeg -y -loglevel error -i \"%s\" -vf \"fps=%u,scale=%u:-2\" -frames:v %u \"%s/frame_%%05d.jpg\" 2>&1",
                video_path.c_str(), fps, max_dimension, max_frames, tmpdir.c_str());

  LOGI("CloudClient: extracting frames: %s\n", cmd);

  FILE* fp = popen(cmd, "r");
  if(!fp)
  {
    LOGE("CloudClient: popen failed: %s\n", std::strerror(errno));
    return false;
  }

  char buf[256];
  std::string err_output;
  while(fgets(buf, sizeof(buf), fp) != nullptr)
  {
    err_output += buf;
  }

  const int rc = pclose(fp);
  if(rc != 0)
  {
    if(!err_output.empty())
    {
      LOGE("CloudClient: ffmpeg failed (exit=%d): %s\n", rc, err_output.c_str());
    }
    else
    {
      LOGE("CloudClient: ffmpeg failed (exit=%d)\n", rc);
    }
    return false;
  }

  for(uint32_t i = 1; i <= max_frames; ++i)
  {
    char path[512];
    std::snprintf(path, sizeof(path), "%s/frame_%05d.jpg", tmpdir.c_str(), i);
    if(std::filesystem::exists(path))
    {
      out_paths.push_back(path);
    }
    else
    {
      break;
    }
  }

  if(out_paths.empty())
  {
    LOGE("CloudClient: no frames extracted\n");
    return false;
  }

  if(out_paths.size() < 2)
  {
    LOGW("CloudClient: only %zu frame(s) extracted, need at least 2\n", out_paths.size());
    return false;
  }

  LOGI("CloudClient: extracted %zu frames\n", out_paths.size());
  return true;
}

CloudClient::CloudClient() = default;

CloudClient::~CloudClient()
{
  close();
}

void CloudClient::setServerAddress(const std::string& host, int port)
{
  m_host = host;
  m_port = port;
}

void CloudClient::setLocalDownloadDir(const std::filesystem::path& dir)
{
  m_downloadDir = dir;
}

void CloudClient::testConnection(ConnectionCallback callback)
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

uint32_t CloudClient::submitVideo(const std::string& video_path,
                                  const CloudRequestOptions& options,
                                  StatusCallback on_update,
                                  CompleteCallback on_complete)
{
  if(!std::filesystem::exists(video_path))
  {
    LOGE("CloudClient: video file not found: %s\n", video_path.c_str());
    return 0;
  }

  const uint32_t n_frames = (options.n_frames > 0 && options.n_frames <= MAX_CLOUD_FRAMES)
                              ? options.n_frames
                              : MAX_CLOUD_FRAMES;

  char tmpdir_template[] = "/tmp/cloud_client_XXXXXX";
  if(!mkdtemp(tmpdir_template))
  {
    LOGE("CloudClient: mkdtemp failed: %s\n", std::strerror(errno));
    return 0;
  }
  const std::string tmpdir(tmpdir_template);

  std::vector<std::string> frame_paths;
  const uint32_t fps = kDefaultFps;
  const uint32_t max_dim = kDefaultMaxDimension;
  if(!extractFrames(video_path, tmpdir, n_frames, fps, max_dim, frame_paths))
  {
    std::error_code ec;
    std::filesystem::remove_all(tmpdir, ec);
    return 0;
  }

  TcpSocket sock;
  if(!sock.connect(m_host, m_port, 10000))
  {
    LOGE("CloudClient: cannot connect to %s:%d\n", m_host.c_str(), m_port);
    std::error_code ec;
    std::filesystem::remove_all(tmpdir, ec);
    return 0;
  }

  CloudRequestOptions opts = options;
  opts.n_frames = static_cast<uint32_t>(frame_paths.size());

  size_t frame_data_size = 0;
  for(const auto& p : frame_paths)
  {
    frame_data_size += sizeof(uint32_t) + p.size();
  }

  CloudRequestPayload payload{};
  payload.n_frames        = opts.n_frames;
  payload.options_size    = sizeof(CloudRequestOptions);
  payload.frame_data_size = static_cast<uint32_t>(frame_data_size);

  std::vector<uint8_t> msg;
  msg.reserve(sizeof(CloudRequestPayload) + sizeof(CloudRequestOptions) + frame_data_size);

  msg.insert(msg.end(),
             reinterpret_cast<const uint8_t*>(&payload),
             reinterpret_cast<const uint8_t*>(&payload) + sizeof(payload));
  msg.insert(msg.end(),
             reinterpret_cast<const uint8_t*>(&opts),
             reinterpret_cast<const uint8_t*>(&opts) + sizeof(opts));

  for(const auto& p : frame_paths)
  {
    uint32_t len = static_cast<uint32_t>(p.size());
    msg.insert(msg.end(),
               reinterpret_cast<const uint8_t*>(&len),
               reinterpret_cast<const uint8_t*>(&len) + sizeof(len));
    msg.insert(msg.end(),
               reinterpret_cast<const uint8_t*>(p.data()),
               reinterpret_cast<const uint8_t*>(p.data()) + p.size());
  }

  LOGI("CloudClient: sending MSG_CLOUD_REQUEST with %u frames (%zu bytes)\n",
       opts.n_frames, msg.size());

  if(!sendFrame(sock, MSG_CLOUD_REQUEST, msg.data(), msg.size()))
  {
    LOGE("CloudClient: failed to send MSG_CLOUD_REQUEST\n");
    std::error_code ec;
    std::filesystem::remove_all(tmpdir, ec);
    return 0;
  }

  uint32_t             resp_type = 0;
  std::vector<uint8_t> resp_payload;
  if(!recvFrame(sock, resp_type, resp_payload, 30000))
  {
    LOGE("CloudClient: no response to MSG_CLOUD_REQUEST\n");
    std::error_code ec;
    std::filesystem::remove_all(tmpdir, ec);
    return 0;
  }

  uint32_t job_id = 0;
  if(resp_type == MSG_CLOUD_PROGRESS && resp_payload.size() >= sizeof(CloudProgressPayload))
  {
    CloudProgressPayload prog{};
    std::memcpy(&prog, resp_payload.data(), sizeof(prog));
    job_id = prog.job_id;
  }
  else if(resp_type == MSG_CLOUD_ERROR && resp_payload.size() >= sizeof(CloudErrorPayload))
  {
    CloudErrorPayload err{};
    std::memcpy(&err, resp_payload.data(), sizeof(err));
    LOGE("CloudClient: server rejected job: code=%u msg=%s\n", err.error_code, err.error_msg);
    std::error_code ec;
    std::filesystem::remove_all(tmpdir, ec);
    return 0;
  }
  else if(resp_type == MSG_ERROR && resp_payload.size() >= sizeof(ErrorPayload))
  {
    ErrorPayload err{};
    std::memcpy(&err, resp_payload.data(), sizeof(err));
    LOGE("CloudClient: server error: code=%u msg=%s\n", err.error_code, err.error_msg);
    std::error_code ec;
    std::filesystem::remove_all(tmpdir, ec);
    return 0;
  }
  else
  {
    LOGE("CloudClient: unexpected response type 0x%08x to REQUEST\n", resp_type);
    std::error_code ec;
    std::filesystem::remove_all(tmpdir, ec);
    return 0;
  }

  if(job_id == 0)
  {
    LOGE("CloudClient: server returned job_id=0\n");
    std::error_code ec;
    std::filesystem::remove_all(tmpdir, ec);
    return 0;
  }

  sock.close();
  std::error_code ec;
  std::filesystem::remove_all(tmpdir, ec);

  {
    std::lock_guard<std::mutex> lock(m_jobsMutex);
    auto& job = m_activeJobs[job_id];
    job.on_update   = std::move(on_update);
    job.on_complete = std::move(on_complete);
    job.poll_thread = std::thread(&CloudClient::pollLoop, this, job_id, job.on_update, job.on_complete);
  }

  return job_id;
}

void CloudClient::pollLoop(uint32_t job_id, StatusCallback on_update, CompleteCallback on_complete)
{
  constexpr int kPollIntervalMs = 250;
  constexpr int kRecvTimeoutMs  = 30000;

  std::string error_msg;
  std::string output_path;
  bool        done    = false;
  bool        errored = false;

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
      LOGW("CloudClient: poll connect failed, retrying\n");
      if(!m_shouldRun.load())
        break;
      std::this_thread::sleep_for(std::chrono::milliseconds(kPollIntervalMs));
      continue;
    }

    if(cancel_flag)
    {
      CloudCancelPayload cancel{};
      cancel.job_id = job_id;
      if(!sendFrame(sock, MSG_CLOUD_CANCEL, reinterpret_cast<const uint8_t*>(&cancel), sizeof(cancel)))
      {
        LOGW("CloudClient: failed to send CANCEL\n");
      }
    }
    else
    {
      CloudPollPayload poll{};
      poll.job_id = job_id;
      if(!sendFrame(sock, MSG_CLOUD_POLL, reinterpret_cast<const uint8_t*>(&poll), sizeof(poll)))
      {
        LOGW("CloudClient: failed to send POLL\n");
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
      LOGW("CloudClient: no response to POLL\n");
      sock.close();
      if(!m_shouldRun.load())
        break;
      std::this_thread::sleep_for(std::chrono::milliseconds(kPollIntervalMs));
      continue;
    }
    sock.close();

    switch(resp_type)
    {
    case MSG_CLOUD_PROGRESS:
      if(resp_payload.size() >= sizeof(CloudProgressPayload))
      {
        CloudProgressPayload prog{};
        std::memcpy(&prog, resp_payload.data(), sizeof(prog));
        JobStatus status;
        switch(prog.stage)
        {
        case 0: status.state = JobStatus::State::Queued; break;
        case 1: status.state = JobStatus::State::Extracting; break;
        case 2: status.state = JobStatus::State::Inference; break;
        case 3: status.state = JobStatus::State::Writing; break;
        case 4: status.state = JobStatus::State::Done; break;
        default: status.state = JobStatus::State::Queued; break;
        }
        status.percent       = prog.percent;
        status.windows_done  = static_cast<int>(prog.windows_done);
        status.windows_total = static_cast<int>(prog.windows_total);
        status.eta_ms        = static_cast<int>(prog.eta_ms);
        if(on_update)
          on_update(job_id, status);
      }
      break;

    case MSG_CLOUD_RESPONSE:
      if(resp_payload.size() >= sizeof(CloudResponsePayload))
      {
        CloudResponsePayload resp{};
        std::memcpy(&resp, resp_payload.data(), sizeof(resp));
        if(resp.path_length > 0 && sizeof(CloudResponsePayload) + resp.path_length <= resp_payload.size())
        {
          output_path.assign(reinterpret_cast<const char*>(resp_payload.data() + sizeof(CloudResponsePayload)),
                             resp.path_length);
          if(!output_path.empty() && output_path.back() == '\0')
            output_path.pop_back();
        }
        done = true;
      }
      break;

    case MSG_CLOUD_ERROR:
      if(resp_payload.size() >= sizeof(CloudErrorPayload))
      {
        CloudErrorPayload err{};
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
      LOGW("CloudClient: unexpected poll response 0x%08x\n", resp_type);
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

void CloudClient::cancelJob(uint32_t job_id)
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
    LOGW("CloudClient: cancelJob connect failed\n");
    return;
  }
  CloudCancelPayload cancel{};
  cancel.job_id = job_id;
  sendFrame(sock, MSG_CLOUD_CANCEL, reinterpret_cast<const uint8_t*>(&cancel), sizeof(cancel));
}

void CloudClient::close()
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

#endif  // WITH_TCP_DEPTH
