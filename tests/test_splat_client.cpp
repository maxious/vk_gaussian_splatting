// Minimal CLI client for the depth_server splat mode.
// Submits a 2-4 image job via FreeSplatterClient and saves the resulting .splat.
#include "../src/free_splatter_client.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

static int usage(const char* a0)
{
  std::cerr << "Usage: " << a0 << " --port PORT --images img1.png,img2.png --output out.splat\n";
  return 2;
}

int main(int argc, char** argv)
{
  int         port        = 9001;
  std::string images_str;
  std::string output_path;

  for(int i = 1; i < argc; ++i)
  {
    if(std::strcmp(argv[i], "--port") == 0 && i + 1 < argc)
      port = std::stoi(argv[++i]);
    else if(std::strcmp(argv[i], "--images") == 0 && i + 1 < argc)
      images_str = argv[++i];
    else if(std::strcmp(argv[i], "--output") == 0 && i + 1 < argc)
      output_path = argv[++i];
    else
      return usage(argv[0]);
  }
  if(images_str.empty() || output_path.empty())
    return usage(argv[0]);

  std::vector<fs::path> image_paths;
  {
    std::string token;
    size_t      pos = 0;
    while((pos = images_str.find(',')) != std::string::npos)
    {
      token = images_str.substr(0, pos);
      if(!token.empty())
        image_paths.push_back(token);
      images_str.erase(0, pos + 1);
    }
    if(!images_str.empty())
      image_paths.push_back(images_str);
  }

  if(image_paths.size() < 2 || image_paths.size() > 4)
  {
    std::cerr << "Need 2-4 images, got " << image_paths.size() << "\n";
    return 1;
  }

  vk_viewer::FreeSplatterClient client;
  client.setServerAddress("127.0.0.1", port);

  std::atomic<bool>   done{false};
  std::string         result_path;
  std::string         error_msg;

  uint32_t job_id = client.submitJob(
      image_paths,
      [](uint32_t, const vk_viewer::FreeSplatterClient::JobStatus& s) {
        std::cout << "Progress: " << s.percent << "% (stage=" << static_cast<int>(s.state) << ")\n";
      },
      [&](uint32_t, const std::string& path, const std::string& err) {
        result_path = path;
        error_msg   = err;
        done        = true;
      });

  if(job_id == 0)
  {
    std::cerr << "Failed to submit job (server full?)\n";
    return 1;
  }

  const int timeout_ms = 300000;
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