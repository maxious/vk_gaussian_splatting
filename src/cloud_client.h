#pragma once

#ifdef WITH_TCP_DEPTH

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "../depth_server/protocol.h"

namespace vk_viewer {

// Raw POSIX TCP client for the depth_server cloud mode. Communicates using
// the custom binary protocol (MSG_CLOUD_* family, TCP_DEPTH_MAGIC framing).
// Extracts frames from a video via ffmpeg, sends them as paths to the server,
// receives the resulting .splat, and copies it to the local filesystem.
// Mirrors FreeSplatterClient's async pattern (std::thread + callbacks, no
// ixwebsocket). Protocol constants are replicated locally in the .cpp so
// vk_viewer does not depend on depth_server/protocol.h headers.
class CloudClient
{
public:
  struct ConnectionStatus
  {
    bool        reachable = false;
    std::string error;
  };

  struct JobStatus
  {
    enum class State
    {
      Queued,
      Extracting,
      Inference,
      Writing,
      Done,
      Error,
      Cancelled
    };

    State   state = State::Queued;
    int     percent = 0;
    int     windows_done = 0;
    int     windows_total = 0;
    int     eta_ms = 0;
    std::string error_message;
  };

  using ConnectionCallback = std::function<void(const ConnectionStatus&)>;
  using StatusCallback     = std::function<void(uint32_t job_id, const JobStatus&)>;
  using CompleteCallback   = std::function<void(uint32_t job_id, const std::string& local_splat_path, const std::string& error)>;
  using ProgressCallback   = std::function<void(uint32_t job_id, const JobStatus&)>;

  CloudClient();
  ~CloudClient();

  CloudClient(const CloudClient&)            = delete;
  CloudClient& operator=(const CloudClient&) = delete;

  void setServerAddress(const std::string& host, int port);
  void setLocalDownloadDir(const std::filesystem::path& dir);

  void testConnection(ConnectionCallback callback);

  // Extracts frames from video_path using ffmpeg, sends them as a cloud job
  // to the server, and spawns a poll thread. Returns server-assigned job_id,
  // or 0 on failure.
  uint32_t submitVideo(const std::string& video_path,
                       const CloudRequestOptions& options,
                       StatusCallback on_update,
                       CompleteCallback on_complete);

  // Opens a fresh socket, sends MSG_CLOUD_CANCEL, closes.
  void cancelJob(uint32_t job_id);

  void close();

private:
  struct TcpSocket
  {
    int fd = -1;

    TcpSocket()           = default;
    TcpSocket(const TcpSocket&)            = delete;
    TcpSocket& operator=(const TcpSocket&) = delete;
    ~TcpSocket();

    bool connect(const std::string& host, int port, int timeout_ms = 5000);
    bool sendAll(const uint8_t* data, size_t len, int timeout_ms = 30000);
    bool recvAll(uint8_t* buf, size_t len, int timeout_ms = 30000);
    void close();
  };

  bool sendFrame(TcpSocket& sock, uint32_t type, const uint8_t* payload, size_t payload_size);
  bool recvFrame(TcpSocket& sock, uint32_t& out_type, std::vector<uint8_t>& out_payload, int timeout_ms);

  bool extractFrames(const std::string& video_path,
                     const std::string& tmpdir,
                     uint32_t max_frames,
                     uint32_t fps,
                     uint32_t max_dimension,
                     std::vector<std::string>& out_paths);

  void pollLoop(uint32_t job_id, StatusCallback on_update, CompleteCallback on_complete);

  struct ActiveJob
  {
    StatusCallback    on_update;
    CompleteCallback  on_complete;
    std::thread       poll_thread;
    std::atomic<bool> cancel_requested{false};
  };

  std::string   m_host = "127.0.0.1";
  int           m_port = 9002;
  std::filesystem::path m_downloadDir;
  std::mutex    m_jobsMutex;
  std::map<uint32_t, ActiveJob> m_activeJobs;
  std::atomic<bool> m_shouldRun{true};
};

}  // namespace vk_viewer

#endif  // WITH_TCP_DEPTH
