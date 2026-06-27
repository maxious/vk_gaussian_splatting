#pragma once

#ifdef WITH_FREE_SPLATTER

#include <string>
#include <vector>
#include <functional>
#include <thread>
#include <mutex>
#include <map>
#include <atomic>
#include <filesystem>
#include <cstdint>

namespace vk_viewer {

// Raw POSIX TCP client for the depth_server splat mode. Communicates using
// the custom binary protocol (MSG_SPLAT_* family, TCP_DEPTH_MAGIC framing).
// Mirrors SupersplatClient's async pattern (std::thread + callbacks, no
// ixwebsocket). Protocol constants are replicated locally in the .cpp so
// vk_viewer does not depend on depth_server/protocol.h headers.
class FreeSplatterClient
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
      Preprocessing,
      Inference,
      Writing,
      Done,
      Error,
      Cancelled
    };

    State       state = State::Queued;
    int         percent = 0;
    int         eta_ms = 0;
    std::string output_path;
    int         n_gaussians = 0;
    std::string error_message;
  };

  using ConnectionCallback =
      std::function<void(const ConnectionStatus&)>;
  using StatusCallback =
      std::function<void(uint32_t job_id, const JobStatus&)>;
  using CompleteCallback =
      std::function<void(uint32_t job_id, const std::string& local_splat_path, const std::string& error)>;

  FreeSplatterClient();
  ~FreeSplatterClient();

  FreeSplatterClient(const FreeSplatterClient&)            = delete;
  FreeSplatterClient& operator=(const FreeSplatterClient&) = delete;

  void setServerAddress(const std::string& host, int port);
  void setLocalDownloadDir(const std::filesystem::path& dir);

  void testConnection(ConnectionCallback callback);

  // Reads each image file from disk, sends MSG_SPLAT_REQUEST, then spawns a
  // poll thread that loops MSG_SPLAT_POLL until MSG_SPLAT_RESPONSE/ERROR.
  // Returns the server-assigned job_id, or 0 on failure. on_update and
  // on_complete are dispatched on the poll thread.
  uint32_t submitJob(const std::vector<std::filesystem::path>& image_paths,
                     StatusCallback                            on_update,
                     CompleteCallback                          on_complete);

  // Opens a fresh socket, sends MSG_SPLAT_CANCEL, closes. The poll thread
  // observes the cancellation via the server's ERROR/RESPONSE reply.
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

  void pollLoop(uint32_t job_id, StatusCallback on_update, CompleteCallback on_complete);

  struct ActiveJob
  {
    StatusCallback  on_update;
    CompleteCallback on_complete;
    std::thread     poll_thread;
    std::atomic<bool> cancel_requested{false};
  };

  std::string             m_host = "127.0.0.1";
  int                     m_port = 9001;
  std::filesystem::path   m_downloadDir;
  std::mutex              m_jobsMutex;
  std::map<uint32_t, ActiveJob> m_activeJobs;
  std::atomic<bool>       m_shouldRun{true};
};

}  // namespace vk_viewer

#endif  // WITH_FREE_SPLATTER
