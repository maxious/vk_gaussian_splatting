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
 See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "npz_loader.h"
#include <fstream>
#include <cstring>
#include <algorithm>
#include <nvutils/logger.hpp>
#include <nvutils/file_mapping.hpp>
#include <nvutils/parallel_work.hpp>
#include <zlib.h>

namespace vk_viewer {

// NPZ file format constants
static constexpr uint32_t NPZ_MAGIC_LOCAL = 0x04034b50;   // "PK\x03\x04" - local file header
static constexpr uint32_t NPZ_MAGIC_EMPTY = 0x08074b50;   // "PK\x07\x08" - empty archive
static constexpr uint32_t NPZ_MAGIC_SPANNED = 0x06054b50; // "PK\x05\x06" - end of central directory
static constexpr size_t   NPZ_HEADER_MIN_SIZE = 64;

// Forward declaration of internal NPZ reader
class NpzReader
{
public:
  struct ArrayInfo
  {
    std::string name;
    std::vector<uint32_t> shape;
    std::string dtype;
    size_t dataOffset;
    size_t dataSize;
    bool   isCompressed;
    size_t elementSize;  // Size of each element in bytes
  };

  bool open(const std::filesystem::path& path);
  void close();

  bool isValid() const { return m_isValid; }
  const std::vector<ArrayInfo>& getArrays() const { return m_arrays; }
  const ArrayInfo* findArray(const std::string& name) const;
  ArrayInfo* findArray(const std::string& name);
  bool parseArrayShape(const std::string& name);

  template<typename T>
  bool readArray(const std::string& name, std::vector<T>& data);

  template<typename T>
  bool readArrayTimestep(const std::string& name, std::vector<T>& data, size_t timestep);

  size_t getTimestepCount(const std::string& arrayName) const;

private:
  bool parseZipCentralDirectory();
  bool readNpyHeader(std::ifstream& file, size_t& headerSize, std::vector<uint32_t>& shape, std::string& dtype);

  std::filesystem::path m_path;
  std::ifstream m_file;
  std::vector<char> m_fileData;
  std::vector<ArrayInfo> m_arrays;
  bool m_isValid = false;
};

bool NpzReader::open(const std::filesystem::path& path)
{
  m_path = path;
  m_file.open(path, std::ios::binary);
  if(!m_file.is_open())
  {
    LOGE("NPZ: Failed to open file: %s\n", path.string().c_str());
    return false;
  }

  // Read entire file into memory for easier processing
  m_file.seekg(0, std::ios::end);
  size_t fileSize = m_file.tellg();
  m_file.seekg(0, std::ios::beg);
  m_fileData.resize(fileSize);
  m_file.read(m_fileData.data(), fileSize);
  m_file.close();

  // Check ZIP magic number
  uint32_t magic = *reinterpret_cast<uint32_t*>(m_fileData.data());
  if(magic != NPZ_MAGIC_LOCAL && magic != NPZ_MAGIC_EMPTY && magic != NPZ_MAGIC_SPANNED)
  {
    LOGE("NPZ: Invalid magic number for file: %s (got 0x%08x)\n", path.string().c_str(), magic);
    return false;
  }

  m_isValid = true;
  return parseZipCentralDirectory();
}

void NpzReader::close()
{
  m_fileData.clear();
  m_arrays.clear();
  m_isValid = false;
}

bool NpzReader::parseZipCentralDirectory()
{
  m_arrays.clear();

  if(m_fileData.size() < 22)
  {
    LOGE("NPZ: File too small to be valid ZIP\n");
    return false;
  }

  // Find end of central directory record (EOCD)
  // EOCD signature is at the end of the file, or up to 22 bytes before
  size_t eocdPos = m_fileData.size() - 22;
  bool foundEOCD = false;
  uint16_t numEntries = 0;
  uint16_t cdSize = 0;
  uint32_t cdOffset = 0;

  for(size_t i = 0; i <= 22 && eocdPos >= i; ++i)
  {
    size_t pos = m_fileData.size() - 22 + i;
    uint32_t sig = *reinterpret_cast<uint32_t*>(m_fileData.data() + pos);
    if(sig == NPZ_MAGIC_SPANNED)  // PK\x05\x06
    {
      // Parse EOCD
      uint16_t diskNum = *reinterpret_cast<uint16_t*>(m_fileData.data() + pos + 4);
      uint16_t diskWithCD = *reinterpret_cast<uint16_t*>(m_fileData.data() + pos + 6);
      numEntries = *reinterpret_cast<uint16_t*>(m_fileData.data() + pos + 10);
      cdSize = *reinterpret_cast<uint16_t*>(m_fileData.data() + pos + 12);
      cdOffset = *reinterpret_cast<uint32_t*>(m_fileData.data() + pos + 16);

      foundEOCD = true;
      eocdPos = pos;
      break;
    }
  }

  if(!foundEOCD)
  {
    LOGE("NPZ: Could not find end of central directory\n");
    return false;
  }

  // Parse central directory entries
  size_t pos = cdOffset;
  for(uint16_t i = 0; i < numEntries && pos < cdOffset + cdSize; ++i)
  {
    uint32_t sig = *reinterpret_cast<uint32_t*>(m_fileData.data() + pos);
    if(sig != 0x02014b50)  // Not a central directory file header
    {
      LOGW("NPZ: Unexpected signature at CD position %zu: 0x%08x\n", pos, sig);
      break;
    }

    // Parse central directory file header
    uint16_t nameLen = *reinterpret_cast<uint16_t*>(m_fileData.data() + pos + 28);
    uint16_t extraLen = *reinterpret_cast<uint16_t*>(m_fileData.data() + pos + 30);
    uint16_t commentLen = *reinterpret_cast<uint16_t*>(m_fileData.data() + pos + 32);
    uint32_t localHeaderOffset = *reinterpret_cast<uint32_t*>(m_fileData.data() + pos + 42);

    // Extract filename
    std::string filename(m_fileData.data() + pos + 46, nameLen);
    // Remove .npy extension if present
    if(filename.size() > 4 && filename.substr(filename.size() - 4) == ".npy")
    {
      filename = filename.substr(0, filename.size() - 4);
    }

    // Skip to local file header to get compressed/uncompressed sizes
    size_t localPos = localHeaderOffset;
    if(localPos + 30 < m_fileData.size())
    {
      uint32_t localSig = *reinterpret_cast<uint32_t*>(m_fileData.data() + localPos);
      if(localSig == NPZ_MAGIC_LOCAL)
      {
        uint16_t localNameLen = *reinterpret_cast<uint16_t*>(m_fileData.data() + localPos + 26);
        uint16_t localExtraLen = *reinterpret_cast<uint16_t*>(m_fileData.data() + localPos + 28);
        uint32_t compressedSize = *reinterpret_cast<uint32_t*>(m_fileData.data() + localPos + 18);
        uint32_t uncompressedSize = *reinterpret_cast<uint32_t*>(m_fileData.data() + localPos + 22);

        // Calculate actual data offset (after local file header)
        size_t dataOffset = localPos + 30 + localNameLen + localExtraLen;

        ArrayInfo info;
        info.name = filename;
        info.dataOffset = dataOffset;
        info.dataSize = uncompressedSize;
        info.isCompressed = (compressedSize != uncompressedSize);
        info.elementSize = 4;  // Default, will be updated when shape is parsed

        m_arrays.push_back(info);
      }
    }

    // Move to next entry
    pos = pos + 46 + nameLen + extraLen + commentLen;
  }

  LOGI("NPZ: Found %zu arrays in file\n", m_arrays.size());
  return !m_arrays.empty();
}

const NpzReader::ArrayInfo* NpzReader::findArray(const std::string& name) const
{
  for(const auto& arr : m_arrays)
  {
    if(arr.name == name)
      return &arr;
  }
  return nullptr;
}

NpzReader::ArrayInfo* NpzReader::findArray(const std::string& name)
{
  for(auto& arr : m_arrays)
  {
    if(arr.name == name)
      return &arr;
  }
  return nullptr;
}

size_t NpzReader::getTimestepCount(const std::string& arrayName) const
{
  const ArrayInfo* info = findArray(arrayName);
  if(!info)
    return 0;

  // For SplatAD format, we need to infer from expected array sizes
  // Typical: means3D[T,N,3], rgb_colors[T,N,3], unnorm_rotations[T,N,4]
  // logit_opacities[N], log_scales[N,3] are static
  return 1;  // Will be updated after reading actual data
}

bool NpzReader::parseArrayShape(const std::string& name)
{
  ArrayInfo* info = findArray(name);
  if(!info)
  {
    LOGE("NPZ: Array '%s' not found\n", name.c_str());
    return false;
  }

  // If shape already parsed, return
  if(!info->shape.empty())
    return true;

  // Check bounds
  if(info->dataOffset + 6 > m_fileData.size())
  {
    LOGE("NPZ: Data offset out of bounds for array '%s': offset=%zu, size=%zu\n", name.c_str(), info->dataOffset, m_fileData.size());
    return false;
  }

  // Check .npy magic
  if(m_fileData[info->dataOffset] != '\x93' ||
     m_fileData[info->dataOffset + 1] != 'N' ||
     m_fileData[info->dataOffset + 2] != 'U' ||
     m_fileData[info->dataOffset + 3] != 'M' ||
     m_fileData[info->dataOffset + 4] != 'P' ||
     m_fileData[info->dataOffset + 5] != 'Y')
  {
    LOGE("NPZ: Invalid .npy magic for array '%s'\n", name.c_str());
    return false;
  }

  uint8_t majorVersion = static_cast<uint8_t>(m_fileData[info->dataOffset + 6]);
  uint8_t minorVersion = static_cast<uint8_t>(m_fileData[info->dataOffset + 7]);

  uint16_t headerLen = 0;
  if(majorVersion == 1)
  {
    headerLen = static_cast<uint8_t>(m_fileData[info->dataOffset + 8]);
  }
  else if(majorVersion >= 2)
  {
    headerLen = *reinterpret_cast<uint16_t*>(m_fileData.data() + info->dataOffset + 8);
  }

  // Validate header length
  if(headerLen < 10 || info->dataOffset + headerLen > m_fileData.size())
  {
    LOGE("NPZ: Invalid header length for array '%s': %u, offset=%zu, fileSize=%zu\n", name.c_str(), headerLen, info->dataOffset, m_fileData.size());
    return false;
  }

  // Extract shape from header dictionary
  std::vector<uint32_t> shape;
  size_t headerDictStart = info->dataOffset + 10;
  size_t headerDictEnd = headerDictStart + headerLen - 2;

  // Create string view for easier parsing
  std::string_view headerView(m_fileData.data() + headerDictStart,
                             headerDictEnd - headerDictStart);

  // Find 'shape' tuple in header
  size_t shapePos = headerView.find("'shape'");
  if(shapePos != std::string_view::npos)
  {
    size_t parenStart = headerView.find("(", shapePos);
    size_t parenEnd = headerView.find(")", parenStart);

    if(parenStart != std::string_view::npos && parenEnd != std::string_view::npos)
    {
       std::string_view sv = headerView.substr(parenStart + 1, parenEnd - parenStart - 1);
       std::string shapeStr(sv);
      std::stringstream ss(shapeStr);
      std::string token;
      while(std::getline(ss, token, ','))
      {
        if(!token.empty())
        {
          shape.push_back(static_cast<uint32_t>(std::stoul(token)));
        }
      }
    }
  }

  // Parse dtype
  std::string dtype;
  size_t dtypePos = headerView.find("'descr'");
  if(dtypePos != std::string_view::npos)
  {
    size_t colonStart = headerView.find(":", dtypePos);
    size_t commaEnd = headerView.find(",", colonStart);
    if(colonStart != std::string_view::npos && commaEnd != std::string_view::npos)
    {
      std::string_view dtypeView = headerView.substr(colonStart + 1, commaEnd - colonStart - 1);
      // Remove quotes and whitespace
      std::string dtypeStr(dtypeView);
      dtypeStr.erase(std::remove_if(dtypeStr.begin(), dtypeStr.end(), ::isspace), dtypeStr.end());
      if(dtypeStr.size() >= 2 && dtypeStr.front() == '\'' && dtypeStr.back() == '\'')
      {
        dtype = dtypeStr.substr(1, dtypeStr.size() - 2);
      }
    }
  }

  // Calculate element size from dtype
  size_t elementSize = 4;  // Default to float32
  if(dtype.find("u1") != std::string::npos || dtype.find("uint8") != std::string::npos)
    elementSize = 1;
  else if(dtype.find("f4") != std::string::npos || dtype.find("float32") != std::string::npos)
    elementSize = 4;
  else if(dtype.find("f8") != std::string::npos || dtype.find("float64") != std::string::npos)
    elementSize = 8;

  info->shape = shape;
  info->dtype = dtype;
  info->elementSize = elementSize;
  return true;
}

template<typename T>
bool NpzReader::readArray(const std::string& name, std::vector<T>& data)
{
  ArrayInfo* info = findArray(name);
  if(!info)
  {
    LOGE("NPZ: Array '%s' not found\n", name.c_str());
    return false;
  }

  // Use already parsed shape
  const std::vector<uint32_t>& shape = info->shape;

  // Check bounds
  if(info->dataOffset + 6 > m_fileData.size())
  {
    LOGE("NPZ: Data offset out of bounds for array '%s': offset=%zu, size=%zu\n", name.c_str(), info->dataOffset, m_fileData.size());
    return false;
  }

  // Check .npy magic
  if(m_fileData[info->dataOffset] != '\x93' ||
     m_fileData[info->dataOffset + 1] != 'N' ||
     m_fileData[info->dataOffset + 2] != 'U' ||
     m_fileData[info->dataOffset + 3] != 'M' ||
     m_fileData[info->dataOffset + 4] != 'P' ||
     m_fileData[info->dataOffset + 5] != 'Y')
  {
    LOGE("NPZ: Invalid .npy magic for array '%s'\n", name.c_str());
    return false;
  }

  uint8_t majorVersion = static_cast<uint8_t>(m_fileData[info->dataOffset + 6]);
  uint8_t minorVersion = static_cast<uint8_t>(m_fileData[info->dataOffset + 7]);

  uint16_t headerLen = 0;
  if(majorVersion == 1)
  {
    headerLen = static_cast<uint8_t>(m_fileData[info->dataOffset + 8]);
  }
  else if(majorVersion >= 2)
  {
    headerLen = *reinterpret_cast<uint16_t*>(m_fileData.data() + info->dataOffset + 8);
  }

  // Validate header length
  if(headerLen < 10 || info->dataOffset + headerLen > m_fileData.size())
  {
    LOGE("NPZ: Invalid header length for array '%s': %u, offset=%zu, fileSize=%zu\n", name.c_str(), headerLen, info->dataOffset, m_fileData.size());
    return false;
  }

  // Calculate total elements
  size_t totalElements = 1;
  for(auto s : shape)
    totalElements *= s;

  // Calculate data start position (after header)
  size_t dataStart = info->dataOffset + 10 + headerLen;

  // Validate data bounds
  size_t dataSizeBytes = totalElements * info->elementSize;
  if(dataStart > m_fileData.size() ||
     dataStart + dataSizeBytes > m_fileData.size())
  {
    LOGE("NPZ: Array data out of bounds for '%s': start=%zu, size=%zu, total=%zu\n",
         name.c_str(), dataStart, dataSizeBytes, m_fileData.size());
    return false;
  }

  data.resize(totalElements);

  // Handle type conversion
  if constexpr (std::is_same_v<T, float>)
  {
    if(info->elementSize == 1)  // uint8
    {
      const uint8_t* src = reinterpret_cast<const uint8_t*>(m_fileData.data() + dataStart);
      for(size_t i = 0; i < totalElements; ++i)
      {
        data[i] = static_cast<float>(src[i]) / 255.0f;
      }
    }
    else  // float
    {
      std::memcpy(data.data(), m_fileData.data() + dataStart, totalElements * sizeof(T));
    }
  }
  else
  {
    std::memcpy(data.data(), m_fileData.data() + dataStart, totalElements * sizeof(T));
  }

  return true;
}

template<typename T>
bool NpzReader::readArrayTimestep(const std::string& name, std::vector<T>& data, size_t timestep)
{
  std::vector<T> fullData;
  if(!readArray(name, fullData))
    return false;

  size_t timestepCount = getTimestepCount(name);
  if(timestep >= timestepCount)
  {
    LOGE("NPZ: Timestep %zu out of range (max: %zu)\n", timestep, timestepCount - 1);
    return false;
  }

  size_t perTimestep = fullData.size() / timestepCount;
  data.assign(fullData.data() + timestep * perTimestep,
              fullData.data() + (timestep + 1) * perTimestep);

  return true;
}

bool NpzLoader::canLoad(const std::filesystem::path& filename)
{
  if(filename.extension() != ".npz")
    return false;

  NpzReader reader;
  if(!reader.open(filename))
    return false;

  // Verify required arrays exist - support multiple formats:
  // Format 1: SplatAD training output (means3D, rgb_colors, logit_opacities, log_scales, unnorm_rotations)
  // Format 2: Simple splat data (means3d, colors) - no temporal data, pre-computed
  const auto& arrays = reader.getArrays();
  bool hasMeans = false, hasColors = false, hasOpacity = false, hasScale = false, hasRot = false;
  bool hasSimpleFormat = false;

  for(const NpzReader::ArrayInfo& arr : arrays)
  {
    if(arr.name == "means3D" || arr.name == "means3d" || arr.name == "means") hasMeans = true;
    if(arr.name == "rgb_colors" || arr.name == "colors") hasColors = true;
    if(arr.name == "logit_opacities" || arr.name == "opacities") hasOpacity = true;
    if(arr.name == "log_scales" || arr.name == "scales") hasScale = true;
    if(arr.name == "unnorm_rotations" || arr.name == "rotations") hasRot = true;
  }

  // Check for simple format (means3d + colors only)
  if(hasMeans && hasColors && !hasOpacity && !hasScale && !hasRot)
  {
    hasSimpleFormat = true;
    return hasMeans && hasColors;  // Only need these two
  }

  reader.close();
  return hasMeans && hasColors && hasOpacity && hasScale && hasRot;
}

bool NpzLoader::load(const std::filesystem::path& filename, SplatSet& output, std::function<void(float)> progressCallback)
{
  return loadTimestep(filename, output, 0, progressCallback);
}

bool NpzLoader::loadTimestep(const std::filesystem::path& filename, SplatSet& output, size_t timestep, std::function<void(float)> progressCallback)
{
  NpzReader reader;
  if(!reader.open(filename))
    return false;

  output.clear();

  const auto& arrays = reader.getArrays();

  // Try to find arrays with various possible names
  const NpzReader::ArrayInfo* meansInfo = reader.findArray("means3D");
  if(!meansInfo) meansInfo = reader.findArray("means3d");
  if(!meansInfo) meansInfo = reader.findArray("means");

  const NpzReader::ArrayInfo* colorsInfo = reader.findArray("rgb_colors");
  if(!colorsInfo) colorsInfo = reader.findArray("colors");

  // Parse shapes for required arrays
  if(meansInfo && !reader.parseArrayShape(meansInfo->name))
  {
    LOGE("NPZ: Failed to parse shape for means array\n");
    reader.close();
    return false;
  }
  if(colorsInfo && !reader.parseArrayShape(colorsInfo->name))
  {
    LOGE("NPZ: Failed to parse shape for colors array\n");
    reader.close();
    return false;
  }

  const NpzReader::ArrayInfo* opacityInfo = reader.findArray("logit_opacities");
  if(!opacityInfo) opacityInfo = reader.findArray("opacities");

  const NpzReader::ArrayInfo* scaleInfo = reader.findArray("log_scales");
  if(!scaleInfo) scaleInfo = reader.findArray("scales");

   const NpzReader::ArrayInfo* rotInfo = reader.findArray("unnorm_rotations");
   if(!rotInfo) rotInfo = reader.findArray("rotations");

   if(!meansInfo || !colorsInfo)
  {
    LOGE("NPZ: Missing required arrays (means3D/means3d and colors/rgb_colors)\n");
    reader.close();
    return false;
  }

  // Parse shapes for required arrays
  if(meansInfo && !reader.parseArrayShape(meansInfo->name))
  {
    LOGE("NPZ: Failed to parse shape for means array\n");
    reader.close();
    return false;
  }
  if(colorsInfo && !reader.parseArrayShape(colorsInfo->name))
  {
    LOGE("NPZ: Failed to parse shape for colors array\n");
    reader.close();
    return false;
  }

   // Detect simple format (only means + colors)
   bool simpleFormat = (!opacityInfo || !scaleInfo || !rotInfo);

  if(!meansInfo || !colorsInfo)
  {
    LOGE("NPZ: Missing required arrays (means3D/means3d and colors/rgb_colors)\n");
    reader.close();
    return false;
  }

  // Determine if temporal and calculate sizes
  bool temporal = (meansInfo->shape.size() == 3);
  size_t N = temporal ? meansInfo->shape[1] : meansInfo->shape[0];
  size_t T = temporal ? meansInfo->shape[0] : 1;

  if(timestep >= T)
  {
    LOGE("NPZ: Timestep %zu out of range (max: %zu)\n", timestep, T - 1);
    reader.close();
    return false;
  }

  LOGI("NPZ: Loading %s, N=%zu splats, T=%zu, format=%s\n",
       filename.string().c_str(), N, T, simpleFormat ? "simple" : "full");

  if(progressCallback) progressCallback(0.1f);

  // Read positions
  LOGI("NPZ: Reading positions from '%s'\n", meansInfo->name.c_str());
  std::vector<float> meansData;
  std::vector<float> allMeans;
  if(!reader.readArray(meansInfo->name, allMeans))
  {
    reader.close();
    return false;
  }
   size_t meansPerTimestep = allMeans.size() / T;
   meansData.assign(allMeans.begin() + timestep * meansPerTimestep,
                    allMeans.begin() + (timestep + 1) * meansPerTimestep);

  if(progressCallback) progressCallback(0.3f);

  // Read colors
  std::vector<float> colorsData;
  std::vector<float> allColors;
  std::vector<uint8_t> colorsUint8;  // For uint8 color data

  if(!reader.readArray(colorsInfo->name, allColors))
  {
    reader.close();
    return false;
  }

  // Check if colors are float or uint8
  bool colorsUint = (colorsInfo->dtype.find("uint8") != std::string::npos);
  if(colorsUint)
  {
    // Convert uint8 to float
    colorsUint8.resize(allColors.size());
    std::memcpy(colorsUint8.data(), allColors.data(), allColors.size() * sizeof(float));
    colorsData.resize(allColors.size());
    for(size_t i = 0; i < allColors.size(); ++i)
    {
      colorsData[i] = colorsUint8[i] / 255.0f;
    }
  }
  else
  {
    colorsData = allColors;
  }

  if(progressCallback) progressCallback(0.5f);

  // Initialize output vectors
  output.positions.resize(N * 3);
  output.f_dc.resize(N * 3);
  output.opacity.resize(N);
  output.scale.resize(N * 3);
  output.rotation.resize(N * 4);

  // Copy positions
  std::memcpy(output.positions.data(), meansData.data(), meansData.size() * sizeof(float));

  // Convert colors to SH DC coefficients
  constexpr float SH_C0 = 0.28209479177387814f;
  for(size_t i = 0; i < N * 3; ++i)
  {
    output.f_dc[i] = (colorsData[i] - 0.5f) * 2.0f;
  }

  if(simpleFormat)
  {
    // Simple format: generate default values for missing attributes
    LOGI("NPZ: Using simple format - generating default opacity, scale, rotation\n");

    // Default opacity (logit of 0.5 = 0)
    for(size_t i = 0; i < N; ++i)
    {
      output.opacity[i] = 0.0f;  // logit(0.5)
    }

    // Default scale (small log-scale = exp(-4) ~ 0.018)
    for(size_t i = 0; i < N * 3; ++i)
    {
      output.scale[i] = -4.0f;  // log-scale
    }

    // Default rotation (identity quaternion: w=1, x=y=z=0)
    for(size_t i = 0; i < N; ++i)
    {
      output.rotation[i * 4 + 0] = 1.0f;  // w
      output.rotation[i * 4 + 1] = 0.0f;  // x
      output.rotation[i * 4 + 2] = 0.0f;  // y
      output.rotation[i * 4 + 3] = 0.0f;  // z
    }
  }
  else
  {
    // Full SplatAD format
    if(progressCallback) progressCallback(0.6f);

    // Load opacity
    std::vector<float> opacityData;
    if(!reader.readArray(opacityInfo->name, opacityData))
    {
      reader.close();
      return false;
    }

    if(progressCallback) progressCallback(0.7f);

    // Load scales
    std::vector<float> allScales;
    if(!reader.readArray(scaleInfo->name, allScales))
    {
      reader.close();
      return false;
    }
    std::vector<float> scaleData;
    if(allScales.size() == N * 3)
    {
      scaleData = allScales;
    }
    else
    {
      size_t scalePerTimestep = allScales.size() / T;
      scaleData.assign(allScales.begin() + timestep * scalePerTimestep * 3,
                       allScales.begin() + (timestep + 1) * scalePerTimestep * 3);
    }

    if(progressCallback) progressCallback(0.8f);

    // Load rotations
    std::vector<float> allRots;
    if(!reader.readArray(rotInfo->name, allRots))
    {
      reader.close();
      return false;
    }
    std::vector<float> rotData;
    bool rotTemporal = (allRots.size() == N * 4 * T);
    if(rotTemporal)
    {
      size_t rotPerTimestep = allRots.size() / T;
      rotData.assign(allRots.begin() + timestep * rotPerTimestep * 4,
                     allRots.begin() + (timestep + 1) * rotPerTimestep * 4);
    }
    else
    {
      rotData = allRots;
    }

    if(progressCallback) progressCallback(0.9f);

    // Copy opacity (logit -> sigmoid -> logit for viewer format)
    for(size_t i = 0; i < N; ++i)
    {
      float sigmoid = 1.0f / (1.0f + std::exp(-opacityData[i]));
      output.opacity[i] = std::log(sigmoid / (1.0f - sigmoid + 1e-6f));
    }

    // Copy scales
    std::memcpy(output.scale.data(), scaleData.data(), scaleData.size() * sizeof(float));

    // Normalize and convert quaternions
    for(size_t i = 0; i < N; ++i)
    {
      float nx = rotData[i * 4 + 0];
      float ny = rotData[i * 4 + 1];
      float nz = rotData[i * 4 + 2];
      float nw = rotData[i * 4 + 3];
      float norm = std::sqrt(nx * nx + ny * ny + nz * nz + nw * nw);
      if(norm > 0)
      {
        nx /= norm;
        ny /= norm;
        nz /= norm;
        nw /= norm;
      }
      output.rotation[i * 4 + 0] = nw;
      output.rotation[i * 4 + 1] = nx;
      output.rotation[i * 4 + 2] = ny;
      output.rotation[i * 4 + 3] = nz;
    }
  }

  reader.close();

  LOGI("NPZ: Loaded %zu splats successfully\n", N);
  if(progressCallback) progressCallback(1.0f);

  return true;
}

bool NpzLoader::hasMultipleTimesteps(const std::filesystem::path& filename)
{
  NpzReader reader;
  if(!reader.open(filename))
    return false;

  const NpzReader::ArrayInfo* meansInfo = reader.findArray("means3D");
  bool result = meansInfo && meansInfo->shape.size() == 3;

  reader.close();
  return result;
}

size_t NpzLoader::getTimestepCount(const std::filesystem::path& filename)
{
  NpzReader reader;
  if(!reader.open(filename))
    return 1;

  const NpzReader::ArrayInfo* meansInfo = reader.findArray("means3D");
  size_t result = meansInfo && meansInfo->shape.size() == 3 ? meansInfo->shape[0] : 1;

  reader.close();
  return result;
}

uint32_t NpzLoader::getTimestepDurationMs(const std::filesystem::path& filename)
{
  // NPZ format doesn't natively store frame duration
  // Could be estimated from filename (e.g., "scene_30fps.npz") or default to 33ms (30fps)
  return 33;  // Default to ~30fps
}

}  // namespace vk_viewer
