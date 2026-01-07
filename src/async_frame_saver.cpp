#include "async_frame_saver.h"
#include <nvutils/logger.hpp>
#include <nvutils/file_operations.hpp>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

#include <algorithm>
#include <fstream>

namespace vk_gaussian_splatting {

namespace {

void stbiWriteCallback(void* context, void* data, int size)
{
  auto* file = static_cast<std::ofstream*>(context);
  file->write(static_cast<const char*>(data), size);
}

}  // namespace

AsyncFrameSaver::AsyncFrameSaver(int numBuffers)
    : m_stagingBuffers(numBuffers)
{
}

void AsyncFrameSaver::queueFrame(const void*                  pixelData,
                                 uint32_t                     width,
                                 uint32_t                     height,
                                 bool                         isHDR,
                                 const std::filesystem::path& outputPath,
                                 int                          quality)
{
  StagingBuffer* buffer = acquireStagingBuffer(width, height, isHDR);

  size_t bytesPerPixel = isHDR ? sizeof(float) * 4 : sizeof(uint8_t) * 4;
  size_t dataSize      = width * height * bytesPerPixel;
  memcpy(buffer->data.data(), pixelData, dataSize);
  buffer->width  = width;
  buffer->height = height;
  buffer->isHDR  = isHDR;

  int frameIndex = m_frameCounter++;
  m_pendingCount++;

  auto future = std::async(std::launch::async, &AsyncFrameSaver::saveFrameToFile, buffer, outputPath, quality,
                           frameIndex, m_frameSavedCallback, this);

  {
    std::lock_guard<std::mutex> lock(m_futuresMutex);
    m_pendingFutures.erase(std::remove_if(m_pendingFutures.begin(), m_pendingFutures.end(),
                                          [](const std::future<void>& f) {
                                            return f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
                                          }),
                           m_pendingFutures.end());
    m_pendingFutures.push_back(std::move(future));
  }
}

void AsyncFrameSaver::waitForAll()
{
  std::lock_guard<std::mutex> lock(m_futuresMutex);
  for(auto& future : m_pendingFutures)
  {
    if(future.valid())
    {
      future.wait();
    }
  }
  m_pendingFutures.clear();
}

StagingBuffer* AsyncFrameSaver::acquireStagingBuffer(uint32_t width, uint32_t height, bool isHDR)
{
  std::unique_lock<std::mutex> lock(m_stagingMutex);

  size_t bytesPerPixel = isHDR ? sizeof(float) * 4 : sizeof(uint8_t) * 4;
  size_t requiredSize  = width * height * bytesPerPixel;

  while(true)
  {
    for(auto& buffer : m_stagingBuffers)
    {
      if(!buffer.inUse)
      {
        if(buffer.data.size() < requiredSize)
        {
          buffer.data.resize(requiredSize);
        }
        buffer.inUse = true;
        return &buffer;
      }
    }

    LOGW("All %d staging buffers in use, waiting...\n", static_cast<int>(m_stagingBuffers.size()));
    m_bufferAvailable.wait(lock);
  }
}

void AsyncFrameSaver::releaseStagingBuffer(StagingBuffer* buffer)
{
  {
    std::lock_guard<std::mutex> lock(m_stagingMutex);
    buffer->inUse = false;
  }
  m_bufferAvailable.notify_one();
}

void AsyncFrameSaver::saveFrameToFile(StagingBuffer*               buffer,
                                      const std::filesystem::path& path,
                                      int                          quality,
                                      int                          frameIndex,
                                      std::function<void(int)>     callback,
                                      AsyncFrameSaver*             self)
{
  std::string pathUtf8  = nvutils::utf8FromPath(path);
  std::string extension = path.extension().string();

  std::ofstream file(path, std::ios::binary);
  if(!file.is_open())
  {
    LOGE("Failed to open file for writing: %s\n", pathUtf8.c_str());
    self->releaseStagingBuffer(buffer);
    self->m_pendingCount--;
    return;
  }

  if(extension == ".hdr")
  {
    stbi_write_hdr_to_func(stbiWriteCallback, &file, buffer->width, buffer->height, 4,
                           reinterpret_cast<const float*>(buffer->data.data()));
  }
  else
  {
    stbi_write_tga_to_func(stbiWriteCallback, &file, buffer->width, buffer->height, 4, buffer->data.data());
  }

  file.close();

  self->releaseStagingBuffer(buffer);
  self->m_pendingCount--;

  if(callback)
  {
    callback(frameIndex);
  }
}

}  // namespace vk_gaussian_splatting
