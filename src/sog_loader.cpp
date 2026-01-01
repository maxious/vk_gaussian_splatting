/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sog_loader.h"

#include <fstream>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <future>

#include <nvutils/logger.hpp>
#include <nvutils/parallel_work.hpp>
#include <tinygltf/json.hpp>

// WebP decoder
#include <webp/decode.h>

// ZIP archive reading using zlib
#include <zlib.h>
#include <cstring>

using nlohmann::json;
using namespace vk_gaussian_splatting;

namespace {

// Simple ZIP file reader using zlib
// ZIP format: https://pkware.cachefly.net/webdocs/casestudies/APPNOTE.TXT

struct ZipLocalFileHeader
{
  uint32_t signature;         // 0x04034b50
  uint16_t versionNeeded;
  uint16_t flags;
  uint16_t compression;
  uint16_t modTime;
  uint16_t modDate;
  uint32_t crc32;
  uint32_t compressedSize;
  uint32_t uncompressedSize;
  uint16_t fileNameLength;
  uint16_t extraFieldLength;
};

struct ZipCentralDirHeader
{
  uint32_t signature;         // 0x02014b50
  uint16_t versionMade;
  uint16_t versionNeeded;
  uint16_t flags;
  uint16_t compression;
  uint16_t modTime;
  uint16_t modDate;
  uint32_t crc32;
  uint32_t compressedSize;
  uint32_t uncompressedSize;
  uint16_t fileNameLength;
  uint16_t extraFieldLength;
  uint16_t commentLength;
  uint16_t diskStart;
  uint16_t internalAttrs;
  uint32_t externalAttrs;
  uint32_t localHeaderOffset;
};

struct ZipEndOfCentralDir
{
  uint32_t signature;         // 0x06054b50
  uint16_t diskNumber;
  uint16_t diskWithCentralDir;
  uint16_t numEntriesThisDisk;
  uint16_t numEntriesTotal;
  uint32_t centralDirSize;
  uint32_t centralDirOffset;
  uint16_t commentLength;
};

#pragma pack(push, 1)
struct ZipLocalFileHeaderPacked
{
  uint32_t signature;
  uint16_t versionNeeded;
  uint16_t flags;
  uint16_t compression;
  uint16_t modTime;
  uint16_t modDate;
  uint32_t crc32;
  uint32_t compressedSize;
  uint32_t uncompressedSize;
  uint16_t fileNameLength;
  uint16_t extraFieldLength;
};
#pragma pack(pop)

class SimpleZipReader
{
public:
  struct FileEntry
  {
    std::string filename;
    uint32_t    compressedSize;
    uint32_t    uncompressedSize;
    uint32_t    localHeaderOffset;
    uint16_t    compression;
  };

  bool open(const std::vector<uint8_t>& zipData)
  {
    m_data = &zipData;
    return parseZipDirectory();
  }

  const std::vector<FileEntry>& getEntries() const { return m_entries; }

  std::vector<uint8_t> extractFile(const std::string& filename) const
  {
    for(const auto& entry : m_entries)
    {
      if(entry.filename == filename)
      {
        return extractEntry(entry);
      }
    }
    return {};
  }

private:
  const std::vector<uint8_t>* m_data = nullptr;
  std::vector<FileEntry>      m_entries;

  bool parseZipDirectory()
  {
    if(m_data->size() < 22)
      return false;

    // Find End of Central Directory record (search backwards)
    const uint8_t* data = m_data->data();
    size_t         size = m_data->size();

    size_t eocdPos = size - 22;
    while(eocdPos > 0)
    {
      if(data[eocdPos] == 0x50 && data[eocdPos + 1] == 0x4b && data[eocdPos + 2] == 0x05 && data[eocdPos + 3] == 0x06)
      {
        break;
      }
      eocdPos--;
    }

    if(eocdPos == 0 && !(data[0] == 0x50 && data[1] == 0x4b && data[2] == 0x05 && data[3] == 0x06))
    {
      LOGE("Failed to find ZIP end of central directory\n");
      return false;
    }

    // Read EOCD
    size_t centralDirOffset = *reinterpret_cast<const uint32_t*>(data + eocdPos + 16);
    size_t numEntries       = *reinterpret_cast<const uint16_t*>(data + eocdPos + 10);

    // Parse central directory
    size_t pos = centralDirOffset;
    for(size_t i = 0; i < numEntries && pos < eocdPos; i++)
    {
      if(pos + 46 > size)
        break;

      uint32_t sig = *reinterpret_cast<const uint32_t*>(data + pos);
      if(sig != 0x02014b50)
        break;

      FileEntry entry;
      entry.compression      = *reinterpret_cast<const uint16_t*>(data + pos + 10);
      entry.compressedSize   = *reinterpret_cast<const uint32_t*>(data + pos + 20);
      entry.uncompressedSize = *reinterpret_cast<const uint32_t*>(data + pos + 24);
      uint16_t nameLen       = *reinterpret_cast<const uint16_t*>(data + pos + 28);
      uint16_t extraLen      = *reinterpret_cast<const uint16_t*>(data + pos + 30);
      uint16_t commentLen    = *reinterpret_cast<const uint16_t*>(data + pos + 32);
      entry.localHeaderOffset = *reinterpret_cast<const uint32_t*>(data + pos + 42);

      if(pos + 46 + nameLen > size)
        break;

      entry.filename = std::string(reinterpret_cast<const char*>(data + pos + 46), nameLen);
      m_entries.push_back(entry);

      pos += 46 + nameLen + extraLen + commentLen;
    }

    return !m_entries.empty();
  }

  std::vector<uint8_t> extractEntry(const FileEntry& entry) const
  {
    const uint8_t* data = m_data->data();
    size_t         size = m_data->size();

    if(entry.localHeaderOffset + 30 > size)
      return {};

    // Read local file header
    size_t   pos = entry.localHeaderOffset;
    uint32_t sig = *reinterpret_cast<const uint32_t*>(data + pos);
    if(sig != 0x04034b50)
      return {};

    uint16_t nameLen  = *reinterpret_cast<const uint16_t*>(data + pos + 26);
    uint16_t extraLen = *reinterpret_cast<const uint16_t*>(data + pos + 28);

    size_t dataOffset = pos + 30 + nameLen + extraLen;
    if(dataOffset + entry.compressedSize > size)
      return {};

    const uint8_t* compressedData = data + dataOffset;

    if(entry.compression == 0)
    {
      // Stored (no compression)
      return std::vector<uint8_t>(compressedData, compressedData + entry.uncompressedSize);
    }
    else if(entry.compression == 8)
    {
      // Deflate
      std::vector<uint8_t> output(entry.uncompressedSize);

      z_stream strm = {};
      strm.next_in  = const_cast<Bytef*>(compressedData);
      strm.avail_in = entry.compressedSize;
      strm.next_out = output.data();
      strm.avail_out = entry.uncompressedSize;

      // Use raw inflate (-MAX_WBITS for raw deflate data without zlib header)
      if(inflateInit2(&strm, -MAX_WBITS) != Z_OK)
        return {};

      int ret = inflate(&strm, Z_FINISH);
      inflateEnd(&strm);

      if(ret != Z_STREAM_END)
        return {};

      return output;
    }

    return {};
  }
};

// Inverse log transform: reverses logTransform(v) = sign(v) * ln(|v| + 1)
inline float invLogTransform(float v)
{
  float a = std::abs(v);
  float e = std::exp(a) - 1.0f;
  return v < 0.0f ? -e : e;
}

// Inverse sigmoid for opacity decoding
inline float sigmoidInv(float y)
{
  float e = std::clamp(y, 1e-6f, 1.0f - 1e-6f);
  return std::log(e / (1.0f - e));
}

// Read entire file into vector
std::vector<uint8_t> readFile(const std::filesystem::path& path)
{
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if(!file)
    return {};

  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<uint8_t> buffer(size);
  if(!file.read(reinterpret_cast<char*>(buffer.data()), size))
    return {};

  return buffer;
}

}  // namespace

bool SogLoader::parseMeta(const std::vector<uint8_t>& jsonData, SogMeta& meta)
{
  try
  {
    json j = json::parse(jsonData.begin(), jsonData.end());

    meta.version = j.value("version", 0u);
    meta.count   = j.value("count", 0u);

    if(meta.version < 2)
    {
      LOGE("SOG version %u not supported, requires version 2+\n", meta.version);
      return false;
    }

    // Parse means
    if(j.contains("means"))
    {
      auto& means = j["means"];
      if(means.contains("mins"))
        meta.means.mins = means["mins"].get<std::vector<float>>();
      if(means.contains("maxs"))
        meta.means.maxs = means["maxs"].get<std::vector<float>>();
      if(means.contains("files"))
        meta.means.files = means["files"].get<std::vector<std::string>>();
    }

    // Parse scales
    if(j.contains("scales"))
    {
      auto& scales = j["scales"];
      if(scales.contains("codebook"))
        meta.scales.codebook = scales["codebook"].get<std::vector<float>>();
      if(scales.contains("files"))
        meta.scales.files = scales["files"].get<std::vector<std::string>>();
    }

    // Parse quats
    if(j.contains("quats"))
    {
      auto& quats = j["quats"];
      if(quats.contains("files"))
        meta.quats.files = quats["files"].get<std::vector<std::string>>();
    }

    // Parse sh0
    if(j.contains("sh0"))
    {
      auto& sh0 = j["sh0"];
      if(sh0.contains("codebook"))
      {
        meta.sh0.codebook = sh0["codebook"].get<std::vector<float>>();

        // Convert SOG Gamma-space DC coefficients to Linear-space DC coefficients
        // SOG stores DC as: C_srgb = 0.5 + SH_C0 * coeff
        // We want: C_linear = 0.5 + SH_C0 * new_coeff
        static const float SH_C0 = 0.28209479177387814f;
        for(float& val : meta.sh0.codebook)
        {
          float srgb = std::clamp(0.5f + SH_C0 * val, 0.0f, 1.0f);
          float linear;
          if(srgb <= 0.04045f)
            linear = srgb / 12.92f;
          else
            linear = std::pow((srgb + 0.055f) / 1.055f, 2.4f);
          val = (linear - 0.5f) / SH_C0;
        }
      }
      if(sh0.contains("files"))
        meta.sh0.files = sh0["files"].get<std::vector<std::string>>();
    }

    // Parse optional shN
    if(j.contains("shN"))
    {
      auto& shN      = j["shN"];
      meta.shN.count = shN.value("count", 0u);
      meta.shN.bands = shN.value("bands", 0u);
      if(shN.contains("codebook"))
      {
        meta.shN.codebook = shN["codebook"].get<std::vector<float>>();
      }
      if(shN.contains("files"))
        meta.shN.files = shN["files"].get<std::vector<std::string>>();
    }

    // Parse optional FreeTimeGS fields
    if(j.contains("motion"))
    {
      auto& motion = j["motion"];
      if(motion.contains("codebook"))
        meta.motion.codebook = motion["codebook"].get<std::vector<float>>();
      if(motion.contains("files"))
        meta.motion.files = motion["files"].get<std::vector<std::string>>();
    }

    if(j.contains("t"))
    {
      auto& t = j["t"];
      if(t.contains("codebook"))
        meta.t.codebook = t["codebook"].get<std::vector<float>>();
      if(t.contains("files"))
        meta.t.files = t["files"].get<std::vector<std::string>>();
    }

    if(j.contains("t_scale"))
    {
      auto& t_scale = j["t_scale"];
      if(t_scale.contains("codebook"))
        meta.t_scale.codebook = t_scale["codebook"].get<std::vector<float>>();
      if(t_scale.contains("files"))
        meta.t_scale.files = t_scale["files"].get<std::vector<std::string>>();
    }

    return true;
  }
  catch(const std::exception& e)
  {
    LOGE("Failed to parse SOG meta.json: %s\n", e.what());
    return false;
  }
}

bool SogLoader::decodeWebP(const std::vector<uint8_t>& webpData, WebPImage& output)
{
  WebPDecoderConfig config;
  if(!WebPInitDecoderConfig(&config))
  {
    LOGE("Failed to initialize WebP decoder config\n");
    return false;
  }

  if(WebPGetFeatures(webpData.data(), webpData.size(), &config.input) != VP8_STATUS_OK)
  {
    LOGE("Failed to get WebP image features\n");
    return false;
  }

  output.width  = static_cast<uint32_t>(config.input.width);
  output.height = static_cast<uint32_t>(config.input.height);
  output.rgba.resize(output.width * output.height * 4);

  config.output.colorspace        = MODE_RGBA;
  config.output.u.RGBA.rgba       = output.rgba.data();
  config.output.u.RGBA.stride     = output.width * 4;
  config.output.u.RGBA.size       = output.rgba.size();
  config.output.is_external_memory = 1;

  config.options.use_threads = 1;

  VP8StatusCode status = WebPDecode(webpData.data(), webpData.size(), &config);
  WebPFreeDecBuffer(&config.output);

  if(status != VP8_STATUS_OK)
  {
    LOGE("Failed to decode WebP image (status %d)\n", status);
    return false;
  }

  return true;
}

void SogLoader::decodePositions(const WebPImage& meansL, const WebPImage& meansU, const SogMeta& meta, SplatSet& output)
{
  const uint32_t count = meta.count;
  output.positions.resize(count * 3);

  const float minX = meta.means.mins[0];
  const float minY = meta.means.mins[1];
  const float minZ = meta.means.mins[2];
  const float maxX = meta.means.maxs[0];
  const float maxY = meta.means.maxs[1];
  const float maxZ = meta.means.maxs[2];

  nvutils::parallel_ranges_pooled<1024>(count, [&](uint64_t start, uint64_t end, uint32_t threadIdx) {
    for(uint64_t i = start; i < end; i++)
    {
      const uint32_t offset = static_cast<uint32_t>(i * 4);
      uint16_t xVal = meansL.rgba[offset + 0] | (static_cast<uint16_t>(meansU.rgba[offset + 0]) << 8);
      uint16_t yVal = meansL.rgba[offset + 1] | (static_cast<uint16_t>(meansU.rgba[offset + 1]) << 8);
      uint16_t zVal = meansL.rgba[offset + 2] | (static_cast<uint16_t>(meansU.rgba[offset + 2]) << 8);
      float xNorm = static_cast<float>(xVal) / 65535.0f;
      float yNorm = static_cast<float>(yVal) / 65535.0f;
      float zNorm = static_cast<float>(zVal) / 65535.0f;
      float xScaled = minX + xNorm * (maxX - minX);
      float yScaled = minY + yNorm * (maxY - minY);
      float zScaled = minZ + zNorm * (maxZ - minZ);
      output.positions[i * 3 + 0] = invLogTransform(xScaled);
      output.positions[i * 3 + 1] = invLogTransform(yScaled);
      output.positions[i * 3 + 2] = invLogTransform(zScaled);
    }
  });
}

void SogLoader::decodeQuaternions(const WebPImage& quats, uint32_t count, SplatSet& output)
{
  output.rotation.resize(count * 4);
  const float sqrt2 = std::sqrt(2.0f);

  nvutils::parallel_ranges_pooled<1024>(count, [&](uint64_t start, uint64_t end, uint32_t threadIdx) {
    for(uint64_t i = start; i < end; i++)
    {
      const uint32_t offset = static_cast<uint32_t>(i * 4);
      uint8_t px  = quats.rgba[offset + 0];
      uint8_t py  = quats.rgba[offset + 1];
      uint8_t pz  = quats.rgba[offset + 2];
      uint8_t tag = quats.rgba[offset + 3];
      if(tag < 252) {
        output.rotation[i * 4 + 0] = 1.0f;
        output.rotation[i * 4 + 1] = 0.0f;
        output.rotation[i * 4 + 2] = 0.0f;
        output.rotation[i * 4 + 3] = 0.0f;
        continue;
      }
      int mode = tag - 252;
      float a = ((static_cast<float>(px) / 255.0f) - 0.5f) * sqrt2;
      float b = ((static_cast<float>(py) / 255.0f) - 0.5f) * sqrt2;
      float c = ((static_cast<float>(pz) / 255.0f) - 0.5f) * sqrt2;
      float d = std::sqrt(std::max(0.0f, 1.0f - (a * a + b * b + c * c)));
      float x, y, z, w;
      switch(mode) {
        case 0: x = a; y = b; z = c; w = d; break;
        case 1: x = d; y = b; z = c; w = a; break;
        case 2: x = b; y = d; z = c; w = a; break;
        case 3: x = b; y = c; z = d; w = a; break;
        default: x = 0.0f; y = 0.0f; z = 0.0f; w = 1.0f; break;
      }
      output.rotation[i * 4 + 0] = w;
      output.rotation[i * 4 + 1] = x;
      output.rotation[i * 4 + 2] = y;
      output.rotation[i * 4 + 3] = z;
    }
  });
}

void SogLoader::decodeScales(const WebPImage& scales, const std::vector<float>& codebook, uint32_t count, SplatSet& output)
{
  output.scale.resize(count * 3);
  nvutils::parallel_ranges_pooled<1024>(count, [&](uint64_t start, uint64_t end, uint32_t threadIdx) {
    for(uint64_t i = start; i < end; i++)
    {
      const uint32_t offset = static_cast<uint32_t>(i * 4);
      uint8_t xIdx = scales.rgba[offset + 0];
      uint8_t yIdx = scales.rgba[offset + 1];
      uint8_t zIdx = scales.rgba[offset + 2];
      output.scale[i * 3 + 0] = codebook[xIdx];
      output.scale[i * 3 + 1] = codebook[yIdx];
      output.scale[i * 3 + 2] = codebook[zIdx];
    }
  });
}

void SogLoader::decodeSh0(const WebPImage& sh0, const std::vector<float>& codebook, uint32_t count, SplatSet& output)
{
  output.f_dc.resize(count * 3);
  output.opacity.resize(count);
  nvutils::parallel_ranges_pooled<1024>(count, [&](uint64_t start, uint64_t end, uint32_t threadIdx) {
    for(uint64_t i = start; i < end; i++)
    {
      const uint32_t offset = static_cast<uint32_t>(i * 4);
      uint8_t rIdx    = sh0.rgba[offset + 0];
      uint8_t gIdx    = sh0.rgba[offset + 1];
      uint8_t bIdx    = sh0.rgba[offset + 2];
      uint8_t opacity = sh0.rgba[offset + 3];

      output.f_dc[i * 3 + 0] = codebook[rIdx];
      output.f_dc[i * 3 + 1] = codebook[gIdx];
      output.f_dc[i * 3 + 2] = codebook[bIdx];

      output.opacity[i] = sigmoidInv(static_cast<float>(opacity) / 255.0f);
    }
  });
}

void SogLoader::decodeShN(const WebPImage& centroids, const WebPImage& labels, const ShNInfo& shN, uint32_t count, SplatSet& output)
{
  if(shN.bands == 0 || shN.count == 0)
    return;

  static const uint32_t coeffsPerBand[] = {3, 8, 15};
  const uint32_t        shCoeffs        = coeffsPerBand[std::min(shN.bands - 1, 2u)];

  // f_rest stores SH coefficients for all 3 channels
  // INRIA layout: per-splat, grouped by channel (R coeffs, G coeffs, B coeffs)
  output.f_rest.resize(count * shCoeffs * 3);

  const uint32_t centroidsWidth = centroids.width;

  nvutils::parallel_ranges_pooled<512>(count, [&](uint64_t start, uint64_t end, uint32_t threadIdx) {
    for(uint64_t i = start; i < end; i++)
    {
      const uint32_t labelOffset = static_cast<uint32_t>(i * 4);

      // Get 16-bit palette index from labels texture
      uint16_t paletteIdx = labels.rgba[labelOffset + 0] | (static_cast<uint16_t>(labels.rgba[labelOffset + 1]) << 8);

      if(paletteIdx >= shN.count)
      {
        // Invalid index - use zero coefficients
        for(uint32_t j = 0; j < shCoeffs * 3; j++)
        {
          output.f_rest[i * shCoeffs * 3 + j] = 0.0f;
        }
        continue;
      }

      // For each SH coefficient
      for(uint32_t j = 0; j < shCoeffs; j++)
      {
        // Calculate centroid pixel location
        uint32_t cx              = (paletteIdx % 64) * shCoeffs + j;
        uint32_t cy              = paletteIdx / 64;
        uint32_t centroidOffset  = (cy * centroidsWidth + cx) * 4;

        // Extract RGB from centroid and map through codebook
        uint8_t rIdx = centroids.rgba[centroidOffset + 0];
        uint8_t gIdx = centroids.rgba[centroidOffset + 1];
        uint8_t bIdx = centroids.rgba[centroidOffset + 2];

        // Store in INRIA layout (grouped by channel)
        output.f_rest[i * shCoeffs * 3 + j]                   = shN.codebook[rIdx];
        output.f_rest[i * shCoeffs * 3 + shCoeffs + j]        = shN.codebook[gIdx];
        output.f_rest[i * shCoeffs * 3 + shCoeffs * 2 + j]    = shN.codebook[bIdx];
      }
    }
  });
}

void SogLoader::decodeMotion(const WebPImage& motion, const std::vector<float>& codebook, uint32_t count, SplatSet& output)
{
  output.motion.resize(count * 3);
  nvutils::parallel_ranges_pooled<1024>(count, [&](uint64_t start, uint64_t end, uint32_t threadIdx) {
    for(uint64_t i = start; i < end; i++)
    {
      const uint32_t offset = static_cast<uint32_t>(i * 4);
      uint8_t xIdx = motion.rgba[offset + 0];
      uint8_t yIdx = motion.rgba[offset + 1];
      uint8_t zIdx = motion.rgba[offset + 2];
      output.motion[i * 3 + 0] = codebook[xIdx];
      output.motion[i * 3 + 1] = codebook[yIdx];
      output.motion[i * 3 + 2] = codebook[zIdx];
    }
  });
}

void SogLoader::decodeTimeCenter(const WebPImage& t, const std::vector<float>& codebook, uint32_t count, SplatSet& output)
{
  output.time.resize(count);
  nvutils::parallel_ranges_pooled<1024>(count, [&](uint64_t start, uint64_t end, uint32_t threadIdx) {
    for(uint64_t i = start; i < end; i++)
    {
      const uint32_t offset = static_cast<uint32_t>(i * 4);
      uint8_t idx = t.rgba[offset + 0];
      output.time[i] = codebook[idx];
    }
  });
}

void SogLoader::decodeTimeScale(const WebPImage& t_scale, const std::vector<float>& codebook, uint32_t count, SplatSet& output)
{
  output.time_scale.resize(count);
  nvutils::parallel_ranges_pooled<1024>(count, [&](uint64_t start, uint64_t end, uint32_t threadIdx) {
    for(uint64_t i = start; i < end; i++)
    {
      const uint32_t offset = static_cast<uint32_t>(i * 4);
      uint8_t idx = t_scale.rgba[offset + 0];
      output.time_scale[i] = codebook[idx];
    }
  });
}

bool SogLoader::loadWithReader(const SogMeta& meta, FileReader reader, SplatSet& output, std::function<void(float)> progressCallback)
{
  const uint32_t count = meta.count;
  if(count == 0)
  {
    LOGE("SOG file has 0 splats\n");
    return false;
  }

  if(progressCallback)
    progressCallback(0.1f);

  std::future<WebPImage> meansL_fut, meansU_fut, quats_fut, scales_fut, sh0_fut, centroids_fut, labels_fut, motion_fut, t_fut, t_scale_fut;

  if(meta.means.files.size() >= 2)
  {
    meansL_fut = std::async(std::launch::async, [&]() {
      WebPImage img;
      decodeWebP(reader(meta.means.files[0]), img);
      return img;
    });
    meansU_fut = std::async(std::launch::async, [&]() {
      WebPImage img;
      decodeWebP(reader(meta.means.files[1]), img);
      return img;
    });
  }

  if(!meta.quats.files.empty())
  {
    quats_fut = std::async(std::launch::async, [&]() {
      WebPImage img;
      decodeWebP(reader(meta.quats.files[0]), img);
      return img;
    });
  }

  if(!meta.scales.files.empty() && !meta.scales.codebook.empty())
  {
    scales_fut = std::async(std::launch::async, [&]() {
      WebPImage img;
      decodeWebP(reader(meta.scales.files[0]), img);
      return img;
    });
  }

  if(!meta.sh0.files.empty() && !meta.sh0.codebook.empty())
  {
    sh0_fut = std::async(std::launch::async, [&]() {
      WebPImage img;
      decodeWebP(reader(meta.sh0.files[0]), img);
      return img;
    });
  }

  if(meta.shN.bands > 0 && meta.shN.files.size() >= 2 && !meta.shN.codebook.empty())
  {
    centroids_fut = std::async(std::launch::async, [&]() {
      WebPImage img;
      decodeWebP(reader(meta.shN.files[0]), img);
      return img;
    });
    labels_fut = std::async(std::launch::async, [&]() {
      WebPImage img;
      decodeWebP(reader(meta.shN.files[1]), img);
      return img;
    });
  }

  if(!meta.motion.files.empty() && !meta.motion.codebook.empty())
  {
    motion_fut = std::async(std::launch::async, [&]() {
      WebPImage img;
      decodeWebP(reader(meta.motion.files[0]), img);
      return img;
    });
  }

  if(!meta.t.files.empty() && !meta.t.codebook.empty())
  {
    t_fut = std::async(std::launch::async, [&]() {
      WebPImage img;
      decodeWebP(reader(meta.t.files[0]), img);
      return img;
    });
  }

  if(!meta.t_scale.files.empty() && !meta.t_scale.codebook.empty())
  {
    t_scale_fut = std::async(std::launch::async, [&]() {
      WebPImage img;
      decodeWebP(reader(meta.t_scale.files[0]), img);
      return img;
    });
  }

  if(meansL_fut.valid() && meansU_fut.valid())
  {
    WebPImage mL = meansL_fut.get();
    WebPImage mU = meansU_fut.get();
    if(!mL.rgba.empty() && !mU.rgba.empty())
      decodePositions(mL, mU, meta, output);
  }
  if(progressCallback)
    progressCallback(0.3f);

  if(quats_fut.valid())
  {
    WebPImage img = quats_fut.get();
    if(!img.rgba.empty())
      decodeQuaternions(img, count, output);
  }
  if(progressCallback)
    progressCallback(0.5f);

  if(scales_fut.valid())
  {
    WebPImage img = scales_fut.get();
    if(!img.rgba.empty())
      decodeScales(img, meta.scales.codebook, count, output);
  }
  if(progressCallback)
    progressCallback(0.7f);

  if(sh0_fut.valid())
  {
    WebPImage img = sh0_fut.get();
    if(!img.rgba.empty())
      decodeSh0(img, meta.sh0.codebook, count, output);
  }
  if(progressCallback)
    progressCallback(0.85f);

  if(centroids_fut.valid() && labels_fut.valid())
  {
    WebPImage centroids = centroids_fut.get();
    WebPImage labels    = labels_fut.get();
    if(!centroids.rgba.empty() && !labels.rgba.empty())
      decodeShN(centroids, labels, meta.shN, count, output);
  }

  if(motion_fut.valid())
  {
    WebPImage img = motion_fut.get();
    if(!img.rgba.empty())
    {
      decodeMotion(img, meta.motion.codebook, count, output);
      output.has_time_data = true;
    }
  }

  if(t_fut.valid())
  {
    WebPImage img = t_fut.get();
    if(!img.rgba.empty())
    {
      decodeTimeCenter(img, meta.t.codebook, count, output);
      output.has_time_data = true;
    }
  }

  if(t_scale_fut.valid())
  {
    WebPImage img = t_scale_fut.get();
    if(!img.rgba.empty())
    {
      decodeTimeScale(img, meta.t_scale.codebook, count, output);
      output.has_time_data = true;
    }
  }

  if(progressCallback)
    progressCallback(1.0f);

  output.convertCoordinates(spz::CoordinateSystem::RDF, spz::CoordinateSystem::RUB);
  if(output.has_time_data)
  {
    LOGI("Loaded SOG file: %u splats (parallelized, linearized DC, temporal)\n", count);
  }
  else
  {
    LOGI("Loaded SOG file: %u splats (parallelized, linearized DC)\n", count);
  }
  return true;
}

bool SogLoader::loadBundled(const std::filesystem::path& sogPath, SplatSet& output, std::function<void(float)> progressCallback)
{
  // Read the entire ZIP file
  std::vector<uint8_t> zipData = readFile(sogPath);
  if(zipData.empty())
  {
    LOGE("Failed to read SOG file: %s\n", sogPath.string().c_str());
    return false;
  }

  // Open ZIP archive
  SimpleZipReader zip;
  if(!zip.open(zipData))
  {
    LOGE("Failed to open SOG archive: %s\n", sogPath.string().c_str());
    return false;
  }

  // Find and read meta.json
  std::vector<uint8_t> metaData = zip.extractFile("meta.json");
  if(metaData.empty())
  {
    LOGE("meta.json not found in SOG archive\n");
    return false;
  }

  SogMeta meta;
  if(!parseMeta(metaData, meta))
  {
    return false;
  }

  // Create file reader lambda
  FileReader reader = [&zip](const std::string& filename) -> std::vector<uint8_t> {
    return zip.extractFile(filename);
  };

  return loadWithReader(meta, reader, output, progressCallback);
}

bool SogLoader::loadUnbundled(const std::filesystem::path& metaPath, SplatSet& output, std::function<void(float)> progressCallback)
{
  // Read meta.json
  std::vector<uint8_t> metaData = readFile(metaPath);
  if(metaData.empty())
  {
    LOGE("Failed to read meta.json: %s\n", metaPath.string().c_str());
    return false;
  }

  SogMeta meta;
  if(!parseMeta(metaData, meta))
  {
    return false;
  }

  // Get directory containing meta.json
  std::filesystem::path baseDir = metaPath.parent_path();

  // Create file reader lambda
  FileReader reader = [&baseDir](const std::string& filename) -> std::vector<uint8_t> {
    return readFile(baseDir / filename);
  };

  return loadWithReader(meta, reader, output, progressCallback);
}

bool SogLoader::load(const std::filesystem::path& filename, SplatSet& output, std::function<void(float)> progressCallback)
{
  std::string ext = filename.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

  if(ext == ".sog")
  {
    return loadBundled(filename, output, progressCallback);
  }
  else if(filename.filename() == "meta.json" || ext == ".json")
  {
    return loadUnbundled(filename, output, progressCallback);
  }
  else
  {
    LOGE("Unknown SOG file type: %s\n", filename.string().c_str());
    return false;
  }
}
