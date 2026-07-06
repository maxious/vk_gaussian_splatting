// Minimal CLI client for the depth_server cloud mode.
// Submits a video file via CloudClient and saves the resulting .splat.
#include "../src/cloud_client.h"

#ifdef WITH_TCP_DEPTH

#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

namespace fs = std::filesystem;

static int usage(const char* a0)
{
  std::cerr << "Usage: " << a0
            << " --port PORT --video video.mp4 --output out.splat"
            << " [--fps FPS] [--max-frames N] [--max-dimension D]\n";
  return 2;
}

int main(int argc, char** argv)
{
  int         port         = 9002;
  std::string video_path;
  std::string output_path;
  uint32_t    fps          = 6;
  uint32_t    max_frames   = 200;
  uint32_t    max_dimension = 640;

  for(int i = 1; i < argc; ++i)
  {
    if(std::strcmp(argv[i], "--port") == 0 && i + 1 < argc)
      port = std::stoi(argv[++i]);
    else if(std::strcmp(argv[i], "--video") == 0 && i + 1 < argc)
      video_path = argv[++i];
    else if(std::strcmp(argv[i], "--output") == 0 && i + 1 < argc)
      output_path = argv[++i];
    else if(std::strcmp(argv[i], "--fps") == 0 && i + 1 < argc)
      fps = static_cast<uint32_t>(std::stoul(argv[++i]));
    else if(std::strcmp(argv[i], "--max-frames") == 0 && i + 1 < argc)
      max_frames = static_cast<uint32_t>(std::stoul(argv[++i]));
    else if(std::strcmp(argv[i], "--max-dimension") == 0 && i + 1 < argc)
      max_dimension = static_cast<uint32_t>(std::stoul(argv[++i]));
    else
      return usage(argv[0]);
  }

  if(video_path.empty() || output_path.empty())
    return usage(argv[0]);

  if(!fs::exists(video_path))
  {
    std::cerr << "ERROR: video file not found: " << video_path << "\n";
    return 1;
  }

  vk_viewer::CloudClient client;
  client.setServerAddress("127.0.0.1", port);

  CloudRequestOptions options{};
  options.n_frames          = max_frames;
  options.chunk_size        = 8;
  options.overlap           = 2;
  options.conf_pct          = 0.0f;
  options.point_size        = 0.01f;
  options.global_budget     = 0;
  options.icp_refine        = 1;
  options.loop_close        = 1;
  options.fuse              = 1;
  options.metric            = 1;
  options.fuse_voxel_frac   = 0.0f;
  options.fuse_trunc_mult   = 0.0f;
  options.timeout_ms        = 0;
  options.flags             = 0;

  std::atomic<bool> done{false};
  std::string       result_path;
  std::string       error_msg;

  uint32_t job_id = client.submitVideo(
      video_path,
      options,
      [](uint32_t, const vk_viewer::CloudClient::JobStatus& s) {
        std::cout << "Progress: " << s.percent
                  << "% stage=" << static_cast<int>(s.state)
                  << " windows=" << s.windows_done << "/" << s.windows_total
                  << " eta=" << s.eta_ms << "ms\n";
      },
      [&](uint32_t, const std::string& path, const std::string& err) {
        result_path = path;
        error_msg   = err;
        done        = true;
      });

  if(job_id == 0)
  {
    std::cerr << "ERROR: Failed to submit job (server full? ffmpeg missing?)\n";
    return 1;
  }

  std::cout << "Job submitted, id=" << job_id << "\n";

  const int timeout_ms = 600000;
  int       waited     = 0;
  while(!done && waited < timeout_ms)
  {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    waited += 100;
  }

  if(!done)
  {
    std::cerr << "TIMEOUT: job did not complete in " << timeout_ms / 1000 << "s\n";
    client.cancelJob(job_id);
    client.close();
    return 3;
  }

  if(!error_msg.empty())
  {
    std::cerr << "ERROR: " << error_msg << "\n";
    client.close();
    return 1;
  }

  if(result_path.empty())
  {
    std::cerr << "ERROR: no output\n";
    client.close();
    return 1;
  }

  std::ifstream src(result_path, std::ios::binary);
  std::ofstream dst(output_path, std::ios::binary);
  dst << src.rdbuf();
  if(!dst)
  {
    std::cerr << "ERROR: failed to write " << output_path << "\n";
    client.close();
    return 1;
  }
  std::cout << "OK: saved " << output_path << "\n";
  client.close();
  return 0;
}

#else
#include <iostream>
int main()
{
  std::cerr << "ERROR: not built with WITH_TCP_DEPTH\n";
  return 1;
}
#endif
