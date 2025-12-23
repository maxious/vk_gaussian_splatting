#pragma once

#include <vulkan/vulkan.h>
#include <filesystem>
#include <vector>
#include <array>
#include <future>
#include <mutex>
#include <queue>
#include <atomic>
#include <functional>
#include <condition_variable>

namespace vk_gaussian_splatting {

struct StagingBuffer
{
  std::vector<uint8_t> data;
  uint32_t             width  = 0;
  uint32_t             height = 0;
  bool                 isHDR  = false;
  bool                 inUse  = false;
};

class AsyncFrameSaver
{
public:
  static constexpr int DEFAULT_NUM_BUFFERS = 8;

  explicit AsyncFrameSaver(int numBuffers = DEFAULT_NUM_BUFFERS);
  ~AsyncFrameSaver() { waitForAll(); }

  void queueFrame(const void*                  pixelData,
                  uint32_t                     width,
                  uint32_t                     height,
                  bool                         isHDR,
                  const std::filesystem::path& outputPath,
                  int                          quality = 100);

  void waitForAll();

  bool hasPendingFrames() const { return m_pendingCount > 0; }
  int  getPendingCount() const { return m_pendingCount.load(); }
  int  getNumBuffers() const { return static_cast<int>(m_stagingBuffers.size()); }

  void setFrameSavedCallback(std::function<void(int)> callback) { m_frameSavedCallback = std::move(callback); }

private:
  StagingBuffer* acquireStagingBuffer(uint32_t width, uint32_t height, bool isHDR);
  void           releaseStagingBuffer(StagingBuffer* buffer);

  static void saveFrameToFile(StagingBuffer*               buffer,
                              const std::filesystem::path& path,
                              int                          quality,
                              int                          frameIndex,
                              std::function<void(int)>     callback,
                              AsyncFrameSaver*             self);

  std::vector<StagingBuffer>   m_stagingBuffers;
  std::mutex                   m_stagingMutex;
  std::condition_variable      m_bufferAvailable;

  std::vector<std::future<void>> m_pendingFutures;
  std::mutex                     m_futuresMutex;
  std::atomic<int>               m_pendingCount{0};
  std::atomic<int>               m_frameCounter{0};

  std::function<void(int)> m_frameSavedCallback;
};

}  // namespace vk_gaussian_splatting
