// Minimal CLI test client for depth_server cloud mode.
// Submits JPEG frame paths via raw TCP, polls for completion,
// saves the resulting .splat to --output.
//
// Uses raw POSIX sockets + depth_server/protocol.h directly
// (no dependency on cloud_client.cpp).
#include "../depth_server/protocol.h"

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

namespace {

// ---- TCP helpers -----------------------------------------------------------

int tcpConnect(const std::string& host, int port, int timeout_ms)
{
  addrinfo hints{};
  hints.ai_family   = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_protocol = IPPROTO_TCP;

  addrinfo*          results = nullptr;
  const std::string  port_str = std::to_string(port);
  const int          gai_rc   = getaddrinfo(host.c_str(), port_str.c_str(), &hints, &results);
  if(gai_rc != 0)
    return -1;

  int fd = -1;
  for(addrinfo* it = results; it != nullptr; it = it->ai_next)
  {
    fd = ::socket(it->ai_family, it->ai_socktype, it->ai_protocol);
    if(fd < 0)
      continue;

    // Non-blocking connect for timeout support
    const int flags = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, flags | O_NONBLOCK);

    const int conn_rc = ::connect(fd, it->ai_addr, it->ai_addrlen);
    if(conn_rc == 0)
      break;

    if(errno != EINPROGRESS)
    {
      ::close(fd);
      fd = -1;
      continue;
    }

    {
      pollfd pfd{};
      pfd.fd     = fd;
      pfd.events = POLLOUT;
      if(::poll(&pfd, 1, timeout_ms) <= 0)
      {
        ::close(fd);
        fd = -1;
        continue;
      }

      int       so_error = 0;
      socklen_t err_len  = sizeof(so_error);
      getsockopt(fd, SOL_SOCKET, SO_ERROR, &so_error, &err_len);
      if(so_error != 0)
      {
        ::close(fd);
        fd = -1;
        continue;
      }
    }
    break;
  }

  freeaddrinfo(results);
  if(fd < 0)
    return -1;

  // Restore blocking
  fcntl(fd, F_SETFL, fcntl(fd, F_GETFL, 0) & ~O_NONBLOCK);
  int yes = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));
  return fd;
}

bool sendAll(int fd, const uint8_t* data, size_t len)
{
  while(len > 0)
  {
    const ssize_t n = ::send(fd, data, len, MSG_NOSIGNAL);
    if(n <= 0)
      return false;
    data += n;
    len -= static_cast<size_t>(n);
  }
  return true;
}

bool recvAll(int fd, uint8_t* buf, size_t len)
{
  while(len > 0)
  {
    const ssize_t n = ::recv(fd, buf, len, 0);
    if(n <= 0)
      return false;
    buf += n;
    len -= static_cast<size_t>(n);
  }
  return true;
}

bool sendFrame(int fd, uint32_t type, const uint8_t* payload, size_t payload_size)
{
  const size_t total = sizeof(FrameHeader) + payload_size;
  if(total > MAX_FRAME_SIZE)
    return false;

  FrameHeader hdr{};
  hdr.magic        = TCP_DEPTH_MAGIC;
  hdr.frame_length = static_cast<uint32_t>(total);
  hdr.message_type = type;

  if(!sendAll(fd, reinterpret_cast<const uint8_t*>(&hdr), sizeof(hdr)))
    return false;
  if(payload_size != 0 && !sendAll(fd, payload, payload_size))
    return false;
  return true;
}

bool recvFrame(int fd, uint32_t& out_type, std::vector<uint8_t>& out_payload, int timeout_ms)
{
  FrameHeader hdr{};
  if(!recvAll(fd, reinterpret_cast<uint8_t*>(&hdr), sizeof(hdr)))
    return false;

  if(hdr.magic != TCP_DEPTH_MAGIC)
    return false;
  if(hdr.frame_length < sizeof(FrameHeader) || hdr.frame_length > MAX_FRAME_SIZE)
    return false;

  const size_t payload_size = hdr.frame_length - sizeof(FrameHeader);
  out_payload.resize(payload_size);
  if(payload_size != 0 && !recvAll(fd, out_payload.data(), payload_size))
    return false;
  out_type = hdr.message_type;
  return true;
}

// ---- CLI help --------------------------------------------------------------

void printHelp(const char* a0)
{
  // clang-format off
  std::cout
    << "Usage: " << a0 << " [options]\n"
    << "Submit JPEG frame paths to depth_server cloud mode and save the .splat.\n"
    << "\n"
    << "Required:\n"
    << "  --port PORT         TCP port of depth_server (default: 9002)\n"
    << "  --frames PATHS      Comma-separated JPEG frame paths (2.." << MAX_CLOUD_FRAMES << ")\n"
    << "  --output PATH       Output .splat file path\n"
    << "\n"
    << "Cloud reconstruction options:\n"
    << "  --chunk-size N       Chunk size [2..24] (default: 8)\n"
    << "  --overlap N          Overlap [0..chunk_size-1] (default: 2)\n"
    << "  --conf-pct F         Confidence threshold percentage 0..100 (default: 0.0)\n"
    << "  --point-size F       Point size > 0 (default: 0.01)\n"
    << "  --fuse / --no-fuse   Enable/disable TSDF fusion (default: enabled)\n"
    << "  --metric / --no-metric Enable/disable metric depth (default: enabled)\n"
    << "  --icp / --no-icp     Enable/disable ICP refinement (default: enabled)\n"
    << "  --loop-close / --no-loop-close  Enable/disable loop closure (default: enabled)\n"
    << "  --fuse-voxel-frac F  TSDF voxel fraction (default: 0.004 if fuse enabled)\n"
    << "  --fuse-trunc-mult F  TSDF truncation multiplier (default: 4.0 if fuse enabled)\n"
    << "  --timeout-ms N       Server-side timeout in ms, 0 = no timeout (default: 0)\n"
    << "  --help               Print this help and exit\n"
    << "\n"
    << "Example:\n"
    << "  " << a0 << " --port 9002 --frames frame1.jpg,frame2.jpg,frame3.jpg\n"
    << "    --output scene.splat --chunk-size 8 --overlap 2 --icp --loop-close\n";
  // clang-format on
}

}  // anonymous namespace

// ---- main ------------------------------------------------------------------

int main(int argc, char** argv)
{
  // ---- parse arguments -----------------------------------------------------
  int         port           = 9002;
  std::string frames_str;
  std::string output_path;
  uint32_t    chunk_size     = 8;
  uint32_t    overlap        = 2;
  float       conf_pct       = 0.0f;
  float       point_size     = 0.01f;
  bool        fuse           = true;
  bool        metric_enabled = true;  // "metric" collides with std::metric
  bool        icp            = true;
  bool        loop_close     = true;
  float       fuse_voxel_frac = 0.0f;
  float       fuse_trunc_mult = 0.0f;
  uint32_t    timeout_ms     = 0;

  for(int i = 1; i < argc; ++i)
  {
    if(std::strcmp(argv[i], "--help") == 0)
    {
      printHelp(argv[0]);
      return 0;
    }
    else if(std::strcmp(argv[i], "--port") == 0 && i + 1 < argc)
      port = std::stoi(argv[++i]);
    else if(std::strcmp(argv[i], "--frames") == 0 && i + 1 < argc)
      frames_str = argv[++i];
    else if(std::strcmp(argv[i], "--output") == 0 && i + 1 < argc)
      output_path = argv[++i];
    else if(std::strcmp(argv[i], "--chunk-size") == 0 && i + 1 < argc)
      chunk_size = static_cast<uint32_t>(std::stoul(argv[++i]));
    else if(std::strcmp(argv[i], "--overlap") == 0 && i + 1 < argc)
      overlap = static_cast<uint32_t>(std::stoul(argv[++i]));
    else if(std::strcmp(argv[i], "--conf-pct") == 0 && i + 1 < argc)
      conf_pct = std::stof(argv[++i]);
    else if(std::strcmp(argv[i], "--point-size") == 0 && i + 1 < argc)
      point_size = std::stof(argv[++i]);
    else if(std::strcmp(argv[i], "--fuse") == 0)
      fuse = true;
    else if(std::strcmp(argv[i], "--no-fuse") == 0)
      fuse = false;
    else if(std::strcmp(argv[i], "--metric") == 0)
      metric_enabled = true;
    else if(std::strcmp(argv[i], "--no-metric") == 0)
      metric_enabled = false;
    else if(std::strcmp(argv[i], "--icp") == 0)
      icp = true;
    else if(std::strcmp(argv[i], "--no-icp") == 0)
      icp = false;
    else if(std::strcmp(argv[i], "--loop-close") == 0)
      loop_close = true;
    else if(std::strcmp(argv[i], "--no-loop-close") == 0)
      loop_close = false;
    else if(std::strcmp(argv[i], "--fuse-voxel-frac") == 0 && i + 1 < argc)
      fuse_voxel_frac = std::stof(argv[++i]);
    else if(std::strcmp(argv[i], "--fuse-trunc-mult") == 0 && i + 1 < argc)
      fuse_trunc_mult = std::stof(argv[++i]);
    else if(std::strcmp(argv[i], "--timeout-ms") == 0 && i + 1 < argc)
      timeout_ms = static_cast<uint32_t>(std::stoul(argv[++i]));
    else
    {
      std::cerr << "Unknown option: " << argv[i] << "\n\n";
      printHelp(argv[0]);
      return 2;
    }
  }

  // Frames and output are required
  if(frames_str.empty() || output_path.empty())
  {
    printHelp(argv[0]);
    return 2;
  }

  // ---- parse frame paths ---------------------------------------------------
  std::vector<std::string> frame_paths;
  {
    std::string remaining = frames_str;
    size_t      pos       = 0;
    while((pos = remaining.find(',')) != std::string::npos)
    {
      const std::string token = remaining.substr(0, pos);
      if(!token.empty())
        frame_paths.push_back(token);
      remaining.erase(0, pos + 1);
    }
    if(!remaining.empty())
      frame_paths.push_back(remaining);
  }

  if(frame_paths.size() < 2 || frame_paths.size() > MAX_CLOUD_FRAMES)
  {
    std::cerr << "ERROR: need 2.." << MAX_CLOUD_FRAMES << " frames, got " << frame_paths.size() << "\n";
    return 1;
  }

  // Verify each frame exists and is readable
  for(const auto& p : frame_paths)
  {
    if(access(p.c_str(), R_OK) != 0)
    {
      std::cerr << "ERROR: frame not found or unreadable: " << p << "\n";
      return 1;
    }
  }

  // ---- build CloudRequestOptions -------------------------------------------
  CloudRequestOptions opts{};
  opts.n_frames       = static_cast<uint32_t>(frame_paths.size());
  opts.chunk_size     = chunk_size;
  opts.overlap        = overlap;
  opts.conf_pct       = conf_pct;
  opts.point_size     = point_size;
  opts.global_budget  = 0;
  opts.icp_refine     = icp ? 1 : 0;
  opts.loop_close     = loop_close ? 1 : 0;
  opts.fuse           = fuse ? 1 : 0;
  opts.metric         = metric_enabled ? 1 : 0;
  opts.fuse_voxel_frac = fuse_voxel_frac;
  opts.fuse_trunc_mult = fuse_trunc_mult;
  opts.timeout_ms     = timeout_ms;
  opts.flags          = 0;

  // ---- build request payload -----------------------------------------------
  size_t frame_data_size = 0;
  for(const auto& p : frame_paths)
    frame_data_size += sizeof(uint32_t) + p.size();

  CloudRequestPayload req{};
  req.n_frames        = static_cast<uint32_t>(frame_paths.size());
  req.options_size    = sizeof(CloudRequestOptions);
  req.frame_data_size = static_cast<uint32_t>(frame_data_size);

  std::vector<uint8_t> msg;
  msg.reserve(sizeof(req) + sizeof(opts) + frame_data_size);
  msg.insert(msg.end(), reinterpret_cast<const uint8_t*>(&req),
             reinterpret_cast<const uint8_t*>(&req) + sizeof(req));
  msg.insert(msg.end(), reinterpret_cast<const uint8_t*>(&opts),
             reinterpret_cast<const uint8_t*>(&opts) + sizeof(opts));
  for(const auto& p : frame_paths)
  {
    const uint32_t len = static_cast<uint32_t>(p.size());
    msg.insert(msg.end(), reinterpret_cast<const uint8_t*>(&len),
               reinterpret_cast<const uint8_t*>(&len) + sizeof(len));
    msg.insert(msg.end(), reinterpret_cast<const uint8_t*>(p.data()),
               reinterpret_cast<const uint8_t*>(p.data()) + p.size());
  }

  // ---- connect and send request --------------------------------------------
  int fd = tcpConnect("127.0.0.1", port, 10000);
  if(fd < 0)
  {
    std::cerr << "ERROR: cannot connect to 127.0.0.1:" << port << "\n";
    return 1;
  }

  if(!sendFrame(fd, MSG_CLOUD_REQUEST, msg.data(), msg.size()))
  {
    std::cerr << "ERROR: failed to send MSG_CLOUD_REQUEST\n";
    ::close(fd);
    return 1;
  }

  // ---- receive initial response (job_id) -----------------------------------
  uint32_t             resp_type = 0;
  std::vector<uint8_t> resp_payload;
  if(!recvFrame(fd, resp_type, resp_payload, 30000))
  {
    std::cerr << "ERROR: no response to MSG_CLOUD_REQUEST\n";
    ::close(fd);
    return 1;
  }
  ::close(fd);

  uint32_t job_id = 0;
  if(resp_type == MSG_CLOUD_PROGRESS && resp_payload.size() >= sizeof(CloudProgressPayload))
  {
    CloudProgressPayload prog{};
    std::memcpy(&prog, resp_payload.data(), sizeof(prog));
    job_id = prog.job_id;
    std::cout << "Initial progress: stage=" << static_cast<int>(prog.stage)
              << " percent=" << static_cast<int>(prog.percent) << "\n";
  }
  else if(resp_type == MSG_CLOUD_ERROR && resp_payload.size() >= sizeof(CloudErrorPayload))
  {
    CloudErrorPayload err{};
    std::memcpy(&err, resp_payload.data(), sizeof(err));
    std::cerr << "ERROR: server rejected job: code=" << err.error_code << " msg=" << err.error_msg
              << "\n";
    return 1;
  }
  else if(resp_type == MSG_ERROR && resp_payload.size() >= sizeof(ErrorPayload))
  {
    ErrorPayload err{};
    std::memcpy(&err, resp_payload.data(), sizeof(err));
    std::cerr << "ERROR: server error: code=" << err.error_code << " msg=" << err.error_msg
              << "\n";
    return 1;
  }
  else
  {
    std::cerr << "ERROR: unexpected response type 0x" << std::hex << resp_type << std::dec
              << " to REQUEST\n";
    return 1;
  }

  if(job_id == 0)
  {
    std::cerr << "ERROR: server returned job_id=0\n";
    return 1;
  }

  std::cout << "Job submitted, id=" << job_id << "\n";

  // ---- poll loop -----------------------------------------------------------
  const int kPollIntervalMs = 250;
  const int kPollTimeoutMs  = 600000;  // 10 minutes
  int       waited_ms       = 0;

  std::string server_output_path;
  bool        done    = false;
  bool        errored = false;
  std::string error_msg;

  while(!done && !errored && waited_ms < kPollTimeoutMs)
  {
    std::this_thread::sleep_for(std::chrono::milliseconds(kPollIntervalMs));
    waited_ms += kPollIntervalMs;

    fd = tcpConnect("127.0.0.1", port, 5000);
    if(fd < 0)
      continue;

    CloudPollPayload poll{};
    poll.job_id = job_id;
    if(!sendFrame(fd, MSG_CLOUD_POLL, reinterpret_cast<const uint8_t*>(&poll), sizeof(poll)))
    {
      ::close(fd);
      continue;
    }

    if(!recvFrame(fd, resp_type, resp_payload, 30000))
    {
      ::close(fd);
      continue;
    }
    ::close(fd);

    switch(resp_type)
    {
    case MSG_CLOUD_PROGRESS:
      if(resp_payload.size() >= sizeof(CloudProgressPayload))
      {
        CloudProgressPayload prog{};
        std::memcpy(&prog, resp_payload.data(), sizeof(prog));
        std::cout << "  progress: " << static_cast<int>(prog.percent)
                  << "% stage=" << static_cast<int>(prog.stage)
                  << " windows=" << prog.windows_done << "/" << prog.windows_total
                  << " eta=" << prog.eta_ms << "ms\n";
      }
      break;

    case MSG_CLOUD_RESPONSE:
      if(resp_payload.size() >= sizeof(CloudResponsePayload))
      {
        CloudResponsePayload resp{};
        std::memcpy(&resp, resp_payload.data(), sizeof(resp));
        if(resp.path_length > 0
           && sizeof(CloudResponsePayload) + resp.path_length <= resp_payload.size())
        {
          server_output_path.assign(
              reinterpret_cast<const char*>(resp_payload.data() + sizeof(CloudResponsePayload)),
              resp.path_length);
          // Strip trailing null if present
          if(!server_output_path.empty() && server_output_path.back() == '\0')
            server_output_path.pop_back();
        }
        std::cout << "  done: " << resp.n_points << " points\n";
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
      std::cout << "  unexpected poll response type 0x" << std::hex << resp_type << std::dec
                << "\n";
      break;
    }
  }

  if(!done && !errored)
  {
    std::cerr << "TIMEOUT: job did not complete in " << kPollTimeoutMs / 1000 << "s\n";
    return 3;
  }

  if(errored)
  {
    std::cerr << "ERROR: " << (error_msg.empty() ? "Job failed" : error_msg) << "\n";
    return 1;
  }

  if(server_output_path.empty())
  {
    std::cerr << "ERROR: server returned empty output path\n";
    return 1;
  }

  // ---- copy output .splat --------------------------------------------------
  {
    std::ifstream src(server_output_path, std::ios::binary);
    if(!src)
    {
      std::cerr << "ERROR: cannot open server output: " << server_output_path << "\n";
      return 1;
    }
    std::ofstream dst(output_path, std::ios::binary);
    dst << src.rdbuf();
    if(!dst)
    {
      std::cerr << "ERROR: failed to write " << output_path << "\n";
      return 1;
    }
  }

  std::cout << "OK: saved " << output_path << "\n";
  return 0;
}
