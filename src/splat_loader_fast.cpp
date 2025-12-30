/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "splat_loader_fast.h"
#include <fstream>
#include <cstring>
#include <charconv>
#include <string_view>
#include <algorithm>
#include <immintrin.h>
#include <nvutils/logger.hpp>
#include <nvutils/file_mapping.hpp>
#include <nvutils/parallel_work.hpp>

#ifdef _WIN32
#include <intrin.h>
#else
#include <cpuid.h>
#endif

namespace vk_gaussian_splatting {

struct PropertyLayout
{
  size_t vertexCount  = 0;
  size_t vertexStride = 0;
  size_t xOffset      = static_cast<size_t>(-1);
  size_t yOffset      = static_cast<size_t>(-1);
  size_t zOffset      = static_cast<size_t>(-1);
  size_t opacityOffset = static_cast<size_t>(-1);
  size_t dcOffset[3]   = {static_cast<size_t>(-1), static_cast<size_t>(-1), static_cast<size_t>(-1)};
  size_t scaleOffset[3] = {static_cast<size_t>(-1), static_cast<size_t>(-1), static_cast<size_t>(-1)};
  size_t rotOffset[4]   = {static_cast<size_t>(-1), static_cast<size_t>(-1), static_cast<size_t>(-1), static_cast<size_t>(-1)};
  size_t restOffset[45];
  int    restCount = 0;
  size_t motionOffset[3] = {static_cast<size_t>(-1), static_cast<size_t>(-1), static_cast<size_t>(-1)};
  size_t timeOffset      = static_cast<size_t>(-1);
  size_t timeScaleOffset = static_cast<size_t>(-1);

  PropertyLayout()
  {
    for(int i = 0; i < 45; ++i)
      restOffset[i] = static_cast<size_t>(-1);
  }
};

static bool hasAVX2()
{
#ifdef _WIN32
  int cpuInfo[4];
  __cpuid(cpuInfo, 7);
  return (cpuInfo[1] & (1 << 5)) != 0;
#else
  unsigned int eax, ebx, ecx, edx;
  if(__get_cpuid_max(0, nullptr) < 7)
    return false;
  __cpuid_count(7, 0, eax, ebx, ecx, edx);
  return (ebx & (1 << 5)) != 0;
#endif
}

static const char* findLineEnd(const char* start, const char* end)
{
  const char* p = start;
  while(p < end)
  {
    if(*p == '\n')
      return p;
    if(*p == '\r' && (p + 1 < end) && *(p + 1) == '\n')
      return p;
    p++;
  }
  return end;
}

static bool parseHeader(const char* data, size_t size, size_t& headerSize, PropertyLayout& layout)
{
  if(size < 4 || std::strncmp(data, "ply", 3) != 0)
    return false;

  const char* ptr = data;
  const char* end = data + size;
  
  bool isBinaryLE = false;
  bool inVertex   = false;

  while(ptr < end)
  {
    const char* lineEnd = findLineEnd(ptr, end);
    std::string_view line(ptr, lineEnd - ptr);
    
    if(line == "end_header")
    {
      headerSize = (lineEnd < end && *lineEnd == '\r') ? (lineEnd - data + 2) : (lineEnd - data + 1);
      return isBinaryLE && layout.vertexCount > 0;
    }
    
    if(line == "format binary_little_endian 1.0")
      isBinaryLE = true;
    else if(line.starts_with("element vertex "))
    {
      std::string_view countStr = line.substr(15);
      std::from_chars(countStr.data(), countStr.data() + countStr.size(), layout.vertexCount);
      inVertex = true;
    }
    else if(line.starts_with("element "))
      inVertex = false;
    else if(inVertex && line.starts_with("property float "))
    {
      std::string_view name = line.substr(15);
      if(name == "x") layout.xOffset = layout.vertexStride;
      else if(name == "y") layout.yOffset = layout.vertexStride;
      else if(name == "z") layout.zOffset = layout.vertexStride;
      else if(name == "opacity") layout.opacityOffset = layout.vertexStride;
      else if(name.starts_with("f_dc_"))
      {
        int idx = 0;
        std::from_chars(name.data() + 5, name.data() + name.size(), idx);
        if(idx >= 0 && idx < 3) layout.dcOffset[idx] = layout.vertexStride;
      }
      else if(name.starts_with("f_rest_"))
      {
        int idx = 0;
        std::from_chars(name.data() + 7, name.data() + name.size(), idx);
        if(idx >= 0 && idx < 45)
        {
          layout.restOffset[idx] = layout.vertexStride;
          if(idx >= layout.restCount) layout.restCount = idx + 1;
        }
      }
      else if(name.starts_with("scale_"))
      {
        int idx = 0;
        std::from_chars(name.data() + 6, name.data() + name.size(), idx);
        if(idx >= 0 && idx < 3) layout.scaleOffset[idx] = layout.vertexStride;
      }
      else if(name.starts_with("rot_"))
      {
        int idx = 0;
        std::from_chars(name.data() + 4, name.data() + name.size(), idx);
        if(idx >= 0 && idx < 4) layout.rotOffset[idx] = layout.vertexStride;
      }
      else if(name.starts_with("motion_"))
      {
        int idx = 0;
        std::from_chars(name.data() + 7, name.data() + name.size(), idx);
        if(idx >= 0 && idx < 3) layout.motionOffset[idx] = layout.vertexStride;
      }
      else if(name == "t") layout.timeOffset = layout.vertexStride;
      else if(name == "t_scale") layout.timeScaleOffset = layout.vertexStride;
      layout.vertexStride += 4;
    }
    
    ptr = (lineEnd < end && *lineEnd == '\r') ? (lineEnd + 2) : (lineEnd + 1);
  }
  return false;
}

bool SplatLoaderFast::canLoad(const std::filesystem::path& filename)
{
  nvutils::FileReadMapping mapping;
  if(!mapping.open(filename))
    return false;
  
  size_t         headerSize;
  PropertyLayout layout;
  return parseHeader(static_cast<const char*>(mapping.data()), std::min(mapping.size(), size_t(16384)), headerSize, layout);
}

bool SplatLoaderFast::load(const std::filesystem::path& filename, SplatSet& output, std::function<void(float)> progressCallback)
{
  nvutils::FileReadMapping mapping;
  if(!mapping.open(filename))
    return false;

  size_t         headerSize;
  PropertyLayout layout;
  if(!parseHeader(static_cast<const char*>(mapping.data()), mapping.size(), headerSize, layout))
    return false;

  const char* dataStart = static_cast<const char*>(mapping.data()) + headerSize;
  const size_t count     = layout.vertexCount;
  const size_t stride    = layout.vertexStride;

  LOGI("PLY: %zu vertices, stride=%zu bytes, headerSize=%zu\n", count, stride, headerSize);
  LOGI("PLY offsets: x=%zu y=%zu z=%zu opacity=%zu\n", layout.xOffset, layout.yOffset, layout.zOffset, layout.opacityOffset);
  LOGI("PLY offsets: t=%zu t_scale=%zu motion=%zu/%zu/%zu\n", 
       layout.timeOffset, layout.timeScaleOffset, 
       layout.motionOffset[0], layout.motionOffset[1], layout.motionOffset[2]);
  
  size_t expectedDataSize = count * stride;
  size_t actualDataSize = mapping.size() - headerSize;
  LOGI("PLY data: expected=%zu actual=%zu\n", expectedDataSize, actualDataSize);
  
  if(actualDataSize < expectedDataSize)
  {
    LOGE("PLY file truncated: expected %zu bytes, got %zu\n", expectedDataSize, actualDataSize);
    return false;
  }

  output.clear();
  output.positions.resize(count * 3);
  output.f_dc.resize(count * 3);
  output.f_rest.resize(count * layout.restCount);
  output.opacity.resize(count);
  output.scale.resize(count * 3);
  output.rotation.resize(count * 4);

  bool useAVX2 = hasAVX2();
  
  auto extract_float = [&](size_t offset, float* dest, size_t destStride) {
    if(offset == static_cast<size_t>(-1)) return;
    nvutils::parallel_ranges_pooled<1024>(count, [&](uint64_t start, uint64_t end, uint32_t threadIdx) {
      uint64_t i = start;
#if defined(__AVX2__)
      if(useAVX2 && destStride == 1)
      {
        for(; i + 7 < end; i += 8)
        {
          __m256 v = _mm256_set_ps(
            *reinterpret_cast<const float*>(dataStart + (i + 7) * stride + offset),
            *reinterpret_cast<const float*>(dataStart + (i + 6) * stride + offset),
            *reinterpret_cast<const float*>(dataStart + (i + 5) * stride + offset),
            *reinterpret_cast<const float*>(dataStart + (i + 4) * stride + offset),
            *reinterpret_cast<const float*>(dataStart + (i + 3) * stride + offset),
            *reinterpret_cast<const float*>(dataStart + (i + 2) * stride + offset),
            *reinterpret_cast<const float*>(dataStart + (i + 1) * stride + offset),
            *reinterpret_cast<const float*>(dataStart + i * stride + offset));
          _mm256_storeu_ps(dest + i, v);
        }
      }
#endif
      for(; i < end; ++i)
      {
        dest[i * destStride] = *reinterpret_cast<const float*>(dataStart + i * stride + offset);
      }
    });
  };

  extract_float(layout.xOffset, output.positions.data() + 0, 3);
  extract_float(layout.yOffset, output.positions.data() + 1, 3);
  extract_float(layout.zOffset, output.positions.data() + 2, 3);
  if(progressCallback) progressCallback(0.2f);

  extract_float(layout.opacityOffset, output.opacity.data(), 1);
  if(progressCallback) progressCallback(0.3f);

  extract_float(layout.dcOffset[0], output.f_dc.data() + 0, 3);
  extract_float(layout.dcOffset[1], output.f_dc.data() + 1, 3);
  extract_float(layout.dcOffset[2], output.f_dc.data() + 2, 3);
  if(progressCallback) progressCallback(0.4f);

  extract_float(layout.scaleOffset[0], output.scale.data() + 0, 3);
  extract_float(layout.scaleOffset[1], output.scale.data() + 1, 3);
  extract_float(layout.scaleOffset[2], output.scale.data() + 2, 3);
  if(progressCallback) progressCallback(0.5f);

  extract_float(layout.rotOffset[0], output.rotation.data() + 0, 4);
  extract_float(layout.rotOffset[1], output.rotation.data() + 1, 4);
  extract_float(layout.rotOffset[2], output.rotation.data() + 2, 4);
  extract_float(layout.rotOffset[3], output.rotation.data() + 3, 4);
  if(progressCallback) progressCallback(0.6f);

  for(int j = 0; j < layout.restCount; ++j)
  {
    extract_float(layout.restOffset[j], output.f_rest.data() + j, layout.restCount);
    if(progressCallback && (j % 5 == 0)) progressCallback(0.6f + 0.3f * (float(j) / layout.restCount));
  }

  if(layout.timeOffset != static_cast<size_t>(-1))
  {
    output.has_time_data = true;
    output.motion.resize(count * 3);
    output.time.resize(count);
    output.time_scale.resize(count);
    extract_float(layout.motionOffset[0], output.motion.data() + 0, 3);
    extract_float(layout.motionOffset[1], output.motion.data() + 1, 3);
    extract_float(layout.motionOffset[2], output.motion.data() + 2, 3);
    extract_float(layout.timeOffset, output.time.data(), 1);
    extract_float(layout.timeScaleOffset, output.time_scale.data(), 1);
    
    for(size_t i = 0; i < count; ++i)
    {
      output.time_scale[i] = std::exp(output.time_scale[i]);
    }
    
    if(progressCallback) progressCallback(0.95f);
  }

  if(progressCallback) progressCallback(1.0f);
  return true;
}

}  // namespace vk_gaussian_splatting
