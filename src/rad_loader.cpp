#include "rad_loader.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include <glm/gtc/packing.hpp>
#include <nvutils/logger.hpp>
#include <tinygltf/json.hpp>
#include <zlib.h>


using nlohmann::json;

namespace vk_viewer {
namespace {

constexpr uint32_t RAD_MAGIC       = 0x30444152;  // 'RAD0'
constexpr uint32_t RAD_CHUNK_MAGIC = 0x43444152;  // 'RADC'
constexpr float    PI_F            = 3.14159265358979323846f;

inline size_t roundUp8(size_t size)
{
  return (size + 7) & ~size_t(7);
}

inline float sigmoidInv(float y)
{
  float e = std::clamp(y, 1e-6f, 1.0f - 1e-6f);
  return std::log(e / (1.0f - e));
}

bool hasExtensionLower(const std::filesystem::path& filePath, std::string ext)
{
  auto fileExt = filePath.extension().string();
  std::transform(fileExt.begin(), fileExt.end(), fileExt.begin(), ::tolower);
  return fileExt == ext;
}

uint16_t readU16LE(const uint8_t* p)
{
  return uint16_t(p[0]) | (uint16_t(p[1]) << 8);
}

uint32_t readU32LE(const uint8_t* p)
{
  return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}

uint64_t readU64LE(const uint8_t* p)
{
  return uint64_t(readU32LE(p)) | (uint64_t(readU32LE(p + 4)) << 32);
}

std::vector<uint8_t> readFile(const std::filesystem::path& path)
{
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if(!file)
    return {};

  std::streamsize size = file.tellg();
  if(size <= 0)
    return {};

  file.seekg(0, std::ios::beg);
  std::vector<uint8_t> buffer(static_cast<size_t>(size));
  if(!file.read(reinterpret_cast<char*>(buffer.data()), size))
    return {};

  return buffer;
}

bool inflateWithWindowBits(const uint8_t* data, size_t size, int windowBits, std::vector<uint8_t>& out)
{
  z_stream strm{};
  strm.next_in  = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(data));
  strm.avail_in = static_cast<uInt>(std::min<size_t>(size, std::numeric_limits<uInt>::max()));

  if(inflateInit2(&strm, windowBits) != Z_OK)
    return false;

  std::vector<uint8_t> result;
  result.reserve(size * 2);
  std::array<uint8_t, 64 * 1024> chunk{};

  int ret = Z_OK;
  while(ret == Z_OK)
  {
    strm.next_out  = chunk.data();
    strm.avail_out = static_cast<uInt>(chunk.size());
    ret            = inflate(&strm, Z_NO_FLUSH);

    size_t produced = chunk.size() - strm.avail_out;
    if(produced > 0)
      result.insert(result.end(), chunk.data(), chunk.data() + produced);

    if(ret == Z_BUF_ERROR && strm.avail_in == 0)
      ret = Z_STREAM_END;
  }

  inflateEnd(&strm);

  if(ret != Z_STREAM_END)
    return false;

  out = std::move(result);
  return true;
}

bool decompressZlibLike(const uint8_t* data, size_t size, std::vector<uint8_t>& out)
{
  out.clear();
  if(inflateWithWindowBits(data, size, 32 + MAX_WBITS, out))
    return true;
  if(inflateWithWindowBits(data, size, MAX_WBITS, out))
    return true;
  if(inflateWithWindowBits(data, size, -MAX_WBITS, out))
    return true;
  return false;
}

float decodeHalf(uint16_t bits)
{
  return glm::unpackHalf1x16(bits);
}

size_t bandsForMaxSh(int maxSh)
{
  if(maxSh <= 0)
    return 0;
  if(maxSh == 1)
    return 3;
  if(maxSh == 2)
    return 8;
  return 15;
}

size_t shCoeffOffsetForProperty(const std::string& property)
{
  if(property == "sh1")
    return 0;
  if(property == "sh2")
    return 3;
  if(property == "sh3")
    return 8;
  return 0;
}

size_t shCoeffCountForProperty(const std::string& property)
{
  if(property == "sh1")
    return 3;
  if(property == "sh2")
    return 5;
  if(property == "sh3")
    return 7;
  return 0;
}

std::array<float, 4> decodeQuatOct888(const uint8_t* data)
{
  const float x0 = float(data[0]) / 255.0f * 2.0f - 1.0f;
  const float y0 = float(data[1]) / 255.0f * 2.0f - 1.0f;
  const float z0 = 1.0f - std::abs(x0) - std::abs(y0);
  const float t  = std::max(0.0f, -z0);

  const float x = x0 >= 0.0f ? (x0 - t) : (x0 + t);
  const float y = y0 >= 0.0f ? (y0 - t) : (y0 + t);
  const float z = z0;

  const float len2 = x * x + y * y + z * z;
  const float invL = len2 > 1e-12f ? 1.0f / std::sqrt(len2) : 1.0f;

  const float ax = x * invL;
  const float ay = y * invL;
  const float az = z * invL;

  const float halfTheta = (float(data[2]) / 255.0f) * 0.5f * PI_F;
  float       s         = std::sin(halfTheta);
  float       w         = std::cos(halfTheta);

  return {ax * s, ay * s, az * s, w};
}

std::vector<float> decodeF32(const uint8_t* data, size_t bytes, size_t dims, size_t count)
{
  if(bytes != count * dims * 4)
    return {};

  std::vector<float> out;
  out.reserve(count * dims);

  for(size_t i = 0; i < count; i++)
  {
    size_t index = i * 4;
    for(size_t d = 0; d < dims; d++)
    {
      out.push_back(std::bit_cast<float>(readU32LE(data + index)));
      index += count * 4;
    }
  }

  return out;
}

std::vector<float> decodeF16(const uint8_t* data, size_t bytes, size_t dims, size_t count)
{
  if(bytes != count * dims * 2)
    return {};

  std::vector<float> out;
  out.reserve(count * dims);

  for(size_t i = 0; i < count; i++)
  {
    size_t index = i * 2;
    for(size_t d = 0; d < dims; d++)
    {
      out.push_back(decodeHalf(readU16LE(data + index)));
      index += count * 2;
    }
  }

  return out;
}

std::vector<float> decodeF32LeBytes(const uint8_t* data, size_t bytes, size_t dims, size_t count)
{
  if(bytes != count * dims * 4)
    return {};

  std::vector<float> out;
  out.reserve(count * dims);

  size_t stride = count * dims;
  for(size_t i = 0; i < count; i++)
  {
    for(size_t d = 0; d < dims; d++)
    {
      size_t                 index = count * d + i;
      std::array<uint8_t, 4> b{};
      for(size_t k = 0; k < 4; k++)
        b[k] = data[index + stride * k];
      out.push_back(std::bit_cast<float>(readU32LE(b.data())));
    }
  }

  return out;
}

std::vector<float> decodeF16LeBytes(const uint8_t* data, size_t bytes, size_t dims, size_t count)
{
  if(bytes != count * dims * 2)
    return {};

  std::vector<float> out;
  out.reserve(count * dims);

  size_t stride = count * dims;
  for(size_t i = 0; i < count; i++)
  {
    for(size_t d = 0; d < dims; d++)
    {
      size_t   index = count * d + i;
      uint16_t h     = uint16_t(data[index]) | (uint16_t(data[index + stride]) << 8);
      out.push_back(decodeHalf(h));
    }
  }

  return out;
}

std::vector<float> decodeR8(const uint8_t* data, size_t bytes, size_t dims, size_t count, float minv, float maxv)
{
  if(bytes != count * dims)
    return {};

  std::vector<float> out;
  out.reserve(count * dims);

  for(size_t i = 0; i < count; i++)
  {
    size_t index = i;
    for(size_t d = 0; d < dims; d++)
    {
      out.push_back((float(data[index]) / 255.0f) * (maxv - minv) + minv);
      index += count;
    }
  }

  return out;
}

std::vector<float> decodeR8Delta(const uint8_t* data, size_t bytes, size_t dims, size_t count, float minv, float maxv)
{
  if(bytes != count * dims)
    return {};

  std::vector<float> out;
  out.reserve(count * dims);
  std::vector<uint8_t> last(dims, 0);

  for(size_t i = 0; i < count; i++)
  {
    size_t index = i;
    for(size_t d = 0; d < dims; d++)
    {
      uint8_t value = uint8_t(last[d] + data[index]);
      last[d]       = value;
      out.push_back((float(value) / 255.0f) * (maxv - minv) + minv);
      index += count;
    }
  }

  return out;
}

std::vector<float> decodeS8(const uint8_t* data, size_t bytes, size_t dims, size_t count, float maxv)
{
  if(bytes != count * dims)
    return {};

  std::vector<float> out;
  out.reserve(count * dims);

  for(size_t i = 0; i < count; i++)
  {
    size_t index = i;
    for(size_t d = 0; d < dims; d++)
    {
      int8_t v = static_cast<int8_t>(data[index]);
      out.push_back((float(v) / 127.0f) * maxv);
      index += count;
    }
  }

  return out;
}

std::vector<float> decodeS8Delta(const uint8_t* data, size_t bytes, size_t dims, size_t count, float maxv)
{
  if(bytes != count * dims)
    return {};

  std::vector<float> out;
  out.reserve(count * dims);
  std::vector<uint8_t> last(dims, 0);

  for(size_t i = 0; i < count; i++)
  {
    size_t index = i;
    for(size_t d = 0; d < dims; d++)
    {
      uint8_t value = uint8_t(last[d] + data[index]);
      last[d]       = value;
      out.push_back((float(static_cast<int8_t>(value)) / 127.0f) * maxv);
      index += count;
    }
  }

  return out;
}

float decodeScale8(uint8_t scale, float lnScaleMin, float lnScaleMax)
{
  if(scale == 0)
    return 0.0f;

  float lnScaleScale = (lnScaleMax - lnScaleMin) / 254.0f;
  return std::exp(lnScaleMin + float(scale - 1) * lnScaleScale);
}

std::vector<float> decodeLn0R8(const uint8_t* data, size_t bytes, size_t dims, size_t count, float minv, float maxv)
{
  if(bytes != count * dims)
    return {};

  std::vector<float> out;
  out.reserve(count * dims);

  for(size_t i = 0; i < count; i++)
  {
    size_t index = i;
    for(size_t d = 0; d < dims; d++)
    {
      out.push_back(decodeScale8(data[index], minv, maxv));
      index += count;
    }
  }

  return out;
}

std::vector<float> decodeLnF16(const uint8_t* data, size_t bytes, size_t dims, size_t count)
{
  if(bytes != count * dims * 2)
    return {};

  std::vector<float> out;
  out.reserve(count * dims);

  for(size_t i = 0; i < count; i++)
  {
    size_t index = i * 2;
    for(size_t d = 0; d < dims; d++)
    {
      out.push_back(std::exp(decodeHalf(readU16LE(data + index))));
      index += count * 2;
    }
  }

  return out;
}

std::vector<uint16_t> decodeU16(const uint8_t* data, size_t bytes, size_t dims, size_t count)
{
  if(dims != 1 || bytes != count * 2)
    return {};

  std::vector<uint16_t> out;
  out.reserve(count);
  for(size_t i = 0; i < count; i++)
    out.push_back(readU16LE(data + i * 2));
  return out;
}

std::vector<uint32_t> decodeU32(const uint8_t* data, size_t bytes, size_t dims, size_t count)
{
  if(dims != 1 || bytes != count * 4)
    return {};

  std::vector<uint32_t> out;
  out.reserve(count);
  for(size_t i = 0; i < count; i++)
    out.push_back(readU32LE(data + i * 4));
  return out;
}

struct RadProperty
{
  uint64_t    offset = 0;
  uint64_t    bytes  = 0;
  std::string name;
  std::string encoding;
  std::string compression;
  bool        hasCompression = false;
  bool        hasMin         = false;
  bool        hasMax         = false;
  float       minv           = 0.0f;
  float       maxv           = 0.0f;
};

struct RadChunkMetaParsed
{
  uint32_t                 version      = 0;
  uint64_t                 base         = 0;
  uint64_t                 count        = 0;
  uint64_t                 payloadBytes = 0;
  int                      maxSh        = 0;
  bool                     lodTree      = false;
  std::vector<RadProperty> properties;
};

struct RadDecoded
{
  size_t                count         = 0;
  int                   maxSh         = 0;
  bool                  lodTree       = false;
  bool                  allocated     = false;
  bool                  hasChildCount = false;
  bool                  hasChildStart = false;
  std::vector<float>    center;
  std::vector<float>    opacity;
  std::vector<float>    rgb;
  std::vector<float>    scales;
  std::vector<float>    quat;
  std::vector<float>    shCoeffRgb;
  std::vector<uint16_t> childCount;
  std::vector<uint32_t> childStart;

  void init(size_t numSplats, int maxShDegree, bool hasLodTree)
  {
    count     = numSplats;
    maxSh     = std::clamp(maxShDegree, 0, 3);
    lodTree   = hasLodTree;
    allocated = true;
    center.assign(count * 3, 0.0f);
    opacity.assign(count, 0.0f);
    rgb.assign(count * 3, 0.0f);
    scales.assign(count * 3, 0.0f);
    quat.assign(count * 4, 0.0f);
    shCoeffRgb.assign(count * bandsForMaxSh(maxSh) * 3, 0.0f);
    childCount.clear();
    childStart.clear();
    hasChildCount = false;
    hasChildStart = false;
    if(lodTree)
    {
      childCount.assign(count, 0);
      childStart.assign(count, 0);
    }
  }
};

bool parseTopMeta(const std::string& text, size_t& count, int& maxSh, bool& lodTree)
{
  try
  {
    json j = json::parse(text);
    if(j.value("version", 0u) != 1u)
    {
      LOGE("RAD: unsupported version %u\n", j.value("version", 0u));
      return false;
    }
    if(j.value("type", std::string()) != "gsplat")
    {
      LOGE("RAD: unsupported type %s\n", j.value("type", std::string()).c_str());
      return false;
    }

    count   = static_cast<size_t>(j.value("count", 0ull));
    maxSh   = std::clamp(j.value("maxSh", 0), 0, 3);
    lodTree = j.value("lodTree", false);
    if(count == 0)
    {
      LOGE("RAD: invalid splat count 0\n");
      return false;
    }
    return true;
  }
  catch(const std::exception& e)
  {
    LOGE("RAD: failed to parse top metadata: %s\n", e.what());
    return false;
  }
}

bool parseChunkMeta(const std::string& text, RadChunkMetaParsed& out)
{
  try
  {
    json j      = json::parse(text);
    out.version = j.value("version", 0u);
    if(out.version != 1u)
    {
      LOGE("RAD: unsupported chunk version %u\n", out.version);
      return false;
    }

    out.base         = j.value("base", 0ull);
    out.count        = j.value("count", 0ull);
    out.payloadBytes = j.value("payloadBytes", 0ull);
    out.maxSh        = std::clamp(j.value("maxSh", 0), 0, 3);
    out.lodTree      = j.value("lodTree", false);
    out.properties.clear();

    if(!j.contains("properties") || !j["properties"].is_array())
    {
      LOGE("RAD: chunk missing properties array\n");
      return false;
    }

    for(const auto& p : j["properties"])
    {
      RadProperty prop;
      prop.offset         = p.value("offset", 0ull);
      prop.bytes          = p.value("bytes", 0ull);
      prop.name           = p.value("property", std::string());
      prop.encoding       = p.value("encoding", std::string());
      prop.hasCompression = p.contains("compression") && !p["compression"].is_null();
      if(prop.hasCompression)
        prop.compression = p["compression"].get<std::string>();
      prop.hasMin = p.contains("min") && !p["min"].is_null();
      prop.hasMax = p.contains("max") && !p["max"].is_null();
      if(prop.hasMin)
        prop.minv = p["min"].get<float>();
      if(prop.hasMax)
        prop.maxv = p["max"].get<float>();
      out.properties.push_back(std::move(prop));
    }

    return true;
  }
  catch(const std::exception& e)
  {
    LOGE("RAD: failed to parse chunk metadata: %s\n", e.what());
    return false;
  }
}

bool decodePropertyBytes(const RadProperty&    prop,
                         const uint8_t*        payloadData,
                         size_t                payloadSize,
                         std::vector<uint8_t>& scratch,
                         const uint8_t*&       outData,
                         size_t&               outSize)
{
  if(prop.offset + prop.bytes > payloadSize)
  {
    LOGE("RAD: property %s out of bounds (offset=%llu bytes=%llu payload=%zu)\n", prop.name.c_str(),
         static_cast<unsigned long long>(prop.offset), static_cast<unsigned long long>(prop.bytes), payloadSize);
    return false;
  }

  const uint8_t* src = payloadData + static_cast<size_t>(prop.offset);
  size_t         len = static_cast<size_t>(prop.bytes);

  if(prop.hasCompression)
  {
    if(prop.compression != "gz")
    {
      LOGE("RAD: unsupported property compression %s\n", prop.compression.c_str());
      return false;
    }

    std::vector<uint8_t> decompressed;
    if(!decompressZlibLike(src, len, decompressed))
    {
      LOGE("RAD: failed to decompress property %s\n", prop.name.c_str());
      return false;
    }
    scratch = std::move(decompressed);
    outData = scratch.data();
    outSize = scratch.size();
    return true;
  }

  scratch.clear();
  outData = src;
  outSize = len;
  return true;
}

bool writeFloatSlice(std::vector<float>& dst, size_t dstDims, size_t base, size_t count, const std::vector<float>& src, size_t srcDims)
{
  if(dstDims != srcDims)
    return false;
  if(src.size() != count * srcDims)
    return false;
  if(base + count > dst.size() / dstDims)
    return false;

  std::copy(src.begin(), src.end(), dst.begin() + static_cast<std::ptrdiff_t>(base * dstDims));
  return true;
}

bool writeU16Slice(std::vector<uint16_t>& dst, size_t base, size_t count, const std::vector<uint16_t>& src)
{
  if(src.size() != count || base + count > dst.size())
    return false;
  std::copy(src.begin(), src.end(), dst.begin() + static_cast<std::ptrdiff_t>(base));
  return true;
}

bool writeU32Slice(std::vector<uint32_t>& dst, size_t base, size_t count, const std::vector<uint32_t>& src)
{
  if(src.size() != count || base + count > dst.size())
    return false;
  std::copy(src.begin(), src.end(), dst.begin() + static_cast<std::ptrdiff_t>(base));
  return true;
}

bool writeShSlice(RadDecoded& decoded, size_t base, size_t count, const std::string& propName, const std::vector<float>& src)
{
  size_t coeffOffset = shCoeffOffsetForProperty(propName);
  size_t coeffCount  = shCoeffCountForProperty(propName);
  size_t bands       = bandsForMaxSh(decoded.maxSh);

  if(coeffCount == 0 || bands == 0)
    return false;
  if(coeffOffset + coeffCount > bands)
    return false;
  if(src.size() != count * coeffCount * 3)
    return false;
  if(base + count > decoded.count)
    return false;

  for(size_t i = 0; i < count; ++i)
  {
    const float* srcSplat = src.data() + i * coeffCount * 3;
    float*       dstSplat = decoded.shCoeffRgb.data() + ((base + i) * bands + coeffOffset) * 3;
    std::copy(srcSplat, srcSplat + coeffCount * 3, dstSplat);
  }

  return true;
}

bool decodeChunkProperties(const RadChunkMetaParsed& chunk, const uint8_t* payload, size_t payloadSize, RadDecoded& decoded)
{
  const size_t base  = static_cast<size_t>(chunk.base);
  const size_t count = static_cast<size_t>(chunk.count);

  std::vector<uint8_t> scratch;

  for(const auto& prop : chunk.properties)
  {
    const uint8_t* propBytes = nullptr;
    size_t         propSize  = 0;
    if(!decodePropertyBytes(prop, payload, payloadSize, scratch, propBytes, propSize))
      return false;

    if(prop.name == "center")
    {
      std::vector<float> values;
      if(prop.encoding == "f32")
        values = decodeF32(propBytes, propSize, 3, count);
      else if(prop.encoding == "f16")
        values = decodeF16(propBytes, propSize, 3, count);
      else if(prop.encoding == "f32_lebytes")
        values = decodeF32LeBytes(propBytes, propSize, 3, count);
      else if(prop.encoding == "f16_lebytes")
        values = decodeF16LeBytes(propBytes, propSize, 3, count);
      else
      {
        LOGE("RAD: unsupported center encoding %s\n", prop.encoding.c_str());
        return false;
      }
      if(values.empty() || !writeFloatSlice(decoded.center, 3, base, count, values, 3))
      {
        LOGE("RAD: failed writing center property\n");
        return false;
      }
    }
    else if(prop.name == "alpha")
    {
      std::vector<float> values;
      if(prop.encoding == "f32")
        values = decodeF32(propBytes, propSize, 1, count);
      else if(prop.encoding == "f16")
        values = decodeF16(propBytes, propSize, 1, count);
      else if(prop.encoding == "r8")
      {
        if(!prop.hasMin || !prop.hasMax)
        {
          LOGE("RAD: alpha r8 missing min/max\n");
          return false;
        }
        values = decodeR8(propBytes, propSize, 1, count, prop.minv, prop.maxv);
      }
      else
      {
        LOGE("RAD: unsupported alpha encoding %s\n", prop.encoding.c_str());
        return false;
      }
      if(values.empty() || !writeFloatSlice(decoded.opacity, 1, base, count, values, 1))
      {
        LOGE("RAD: failed writing alpha property\n");
        return false;
      }
    }
    else if(prop.name == "rgb")
    {
      std::vector<float> values;
      if(prop.encoding == "f32")
        values = decodeF32(propBytes, propSize, 3, count);
      else if(prop.encoding == "f16")
        values = decodeF16(propBytes, propSize, 3, count);
      else if(prop.encoding == "r8")
      {
        if(!prop.hasMin || !prop.hasMax)
        {
          LOGE("RAD: rgb r8 missing min/max\n");
          return false;
        }
        values = decodeR8(propBytes, propSize, 3, count, prop.minv, prop.maxv);
      }
      else if(prop.encoding == "r8_delta")
      {
        if(!prop.hasMin || !prop.hasMax)
        {
          LOGE("RAD: rgb r8_delta missing min/max\n");
          return false;
        }
        values = decodeR8Delta(propBytes, propSize, 3, count, prop.minv, prop.maxv);
      }
      else
      {
        LOGE("RAD: unsupported rgb encoding %s\n", prop.encoding.c_str());
        return false;
      }
      if(values.empty() || !writeFloatSlice(decoded.rgb, 3, base, count, values, 3))
      {
        LOGE("RAD: failed writing rgb property\n");
        return false;
      }
    }
    else if(prop.name == "scales")
    {
      std::vector<float> values;
      if(prop.encoding == "f32")
        values = decodeF32(propBytes, propSize, 3, count);
      else if(prop.encoding == "ln_f16")
        values = decodeLnF16(propBytes, propSize, 3, count);
      else if(prop.encoding == "ln_0r8")
      {
        if(!prop.hasMin || !prop.hasMax)
        {
          LOGE("RAD: scales ln_0r8 missing min/max\n");
          return false;
        }
        values = decodeLn0R8(propBytes, propSize, 3, count, prop.minv, prop.maxv);
      }
      else
      {
        LOGE("RAD: unsupported scales encoding %s\n", prop.encoding.c_str());
        return false;
      }
      if(values.empty() || !writeFloatSlice(decoded.scales, 3, base, count, values, 3))
      {
        LOGE("RAD: failed writing scales property\n");
        return false;
      }
    }
    else if(prop.name == "orientation")
    {
      std::vector<float> values;
      if(prop.encoding == "oct88r8")
      {
        if(propSize != count * 3)
        {
          LOGE("RAD: orientation oct88r8 size mismatch\n");
          return false;
        }
        values.reserve(count * 4);
        for(size_t i = 0; i < count; ++i)
        {
          auto q = decodeQuatOct888(propBytes + i * 3);
          values.insert(values.end(), q.begin(), q.end());
        }
      }
      else if(prop.encoding == "f32")
      {
        auto xyz = decodeF32(propBytes, propSize, 3, count);
        if(xyz.empty())
        {
          LOGE("RAD: failed decoding orientation f32\n");
          return false;
        }
        values.reserve(count * 4);
        for(size_t i = 0; i < count; ++i)
        {
          float x = xyz[i * 3 + 0];
          float y = xyz[i * 3 + 1];
          float z = xyz[i * 3 + 2];
          float w = std::sqrt(std::max(0.0f, 1.0f - x * x - y * y - z * z));
          values.insert(values.end(), {x, y, z, w});
        }
      }
      else if(prop.encoding == "f16")
      {
        auto xyz = decodeF16(propBytes, propSize, 3, count);
        if(xyz.empty())
        {
          LOGE("RAD: failed decoding orientation f16\n");
          return false;
        }
        values.reserve(count * 4);
        for(size_t i = 0; i < count; ++i)
        {
          float x = xyz[i * 3 + 0];
          float y = xyz[i * 3 + 1];
          float z = xyz[i * 3 + 2];
          float w = std::sqrt(std::max(0.0f, 1.0f - x * x - y * y - z * z));
          values.insert(values.end(), {x, y, z, w});
        }
      }
      else
      {
        LOGE("RAD: unsupported orientation encoding %s\n", prop.encoding.c_str());
        return false;
      }
      if(values.empty() || !writeFloatSlice(decoded.quat, 4, base, count, values, 4))
      {
        LOGE("RAD: failed writing orientation property\n");
        return false;
      }
    }
    else if(prop.name == "sh1" || prop.name == "sh2" || prop.name == "sh3")
    {
      size_t             coeffCount = shCoeffCountForProperty(prop.name);
      size_t             dims       = coeffCount * 3;
      std::vector<float> values;
      if(prop.encoding == "f32")
        values = decodeF32(propBytes, propSize, dims, count);
      else if(prop.encoding == "f16")
        values = decodeF16(propBytes, propSize, dims, count);
      else if(prop.encoding == "r8")
      {
        if(!prop.hasMin || !prop.hasMax)
        {
          LOGE("RAD: %s r8 missing min/max\n", prop.name.c_str());
          return false;
        }
        values = decodeR8(propBytes, propSize, dims, count, prop.minv, prop.maxv);
      }
      else if(prop.encoding == "r8_delta")
      {
        if(!prop.hasMin || !prop.hasMax)
        {
          LOGE("RAD: %s r8_delta missing min/max\n", prop.name.c_str());
          return false;
        }
        values = decodeR8Delta(propBytes, propSize, dims, count, prop.minv, prop.maxv);
      }
      else if(prop.encoding == "s8")
      {
        if(!prop.hasMax)
        {
          LOGE("RAD: %s s8 missing max\n", prop.name.c_str());
          return false;
        }
        values = decodeS8(propBytes, propSize, dims, count, prop.maxv);
      }
      else if(prop.encoding == "s8_delta")
      {
        if(!prop.hasMax)
        {
          LOGE("RAD: %s s8_delta missing max\n", prop.name.c_str());
          return false;
        }
        values = decodeS8Delta(propBytes, propSize, dims, count, prop.maxv);
      }
      else
      {
        LOGE("RAD: unsupported %s encoding %s\n", prop.name.c_str(), prop.encoding.c_str());
        return false;
      }

      if(values.empty() || !writeShSlice(decoded, base, count, prop.name, values))
      {
        LOGE("RAD: failed writing %s property\n", prop.name.c_str());
        return false;
      }
    }
    else if(prop.name == "child_count")
    {
      if(prop.encoding != "u16")
      {
        LOGE("RAD: unsupported child_count encoding %s\n", prop.encoding.c_str());
        return false;
      }
      auto values = decodeU16(propBytes, propSize, 1, count);
      if(values.empty() || decoded.childCount.empty() || !writeU16Slice(decoded.childCount, base, count, values))
      {
        LOGE("RAD: failed writing child_count property\n");
        return false;
      }
      decoded.hasChildCount = true;
    }
    else if(prop.name == "child_start")
    {
      if(prop.encoding != "u32")
      {
        LOGE("RAD: unsupported child_start encoding %s\n", prop.encoding.c_str());
        return false;
      }
      auto values = decodeU32(propBytes, propSize, 1, count);
      if(values.empty() || decoded.childStart.empty() || !writeU32Slice(decoded.childStart, base, count, values))
      {
        LOGE("RAD: failed writing child_start property\n");
        return false;
      }
      decoded.hasChildStart = true;
    }
  }

  return true;
}

bool parseChunkAt(const std::vector<uint8_t>& bytes, size_t& offset, RadDecoded& decoded, bool allowInitFromChunk)
{
  if(offset + 8 > bytes.size())
    return false;
  if(readU32LE(bytes.data() + offset) != RAD_CHUNK_MAGIC)
  {
    LOGE("RAD: invalid chunk magic at offset %zu\n", offset);
    return false;
  }

  uint32_t metaLen = readU32LE(bytes.data() + offset + 4);
  size_t   metaEnd = offset + 8 + roundUp8(metaLen);
  if(metaEnd + 8 > bytes.size())
  {
    LOGE("RAD: truncated chunk header\n");
    return false;
  }

  std::string        metaText(reinterpret_cast<const char*>(bytes.data() + offset + 8), metaLen);
  RadChunkMetaParsed chunk;
  if(!parseChunkMeta(metaText, chunk))
    return false;

  uint64_t payloadBytes = readU64LE(bytes.data() + metaEnd);
  if(payloadBytes != chunk.payloadBytes)
  {
    LOGW("RAD: chunk payloadBytes mismatch header=%llu meta=%llu\n", static_cast<unsigned long long>(payloadBytes),
         static_cast<unsigned long long>(chunk.payloadBytes));
  }

  size_t payloadStart = metaEnd + 8;
  size_t payloadSize  = static_cast<size_t>(payloadBytes);
  if(payloadStart + payloadSize > bytes.size())
  {
    LOGE("RAD: truncated chunk payload\n");
    return false;
  }

  if(!decoded.allocated)
  {
    if(!allowInitFromChunk)
    {
      LOGE("RAD: internal error, chunk before top metadata init\n");
      return false;
    }
    decoded.init(static_cast<size_t>(chunk.count), chunk.maxSh, chunk.lodTree);
  }

  if(static_cast<size_t>(chunk.base + chunk.count) > decoded.count)
  {
    LOGE("RAD: chunk range out of bounds base=%llu count=%llu total=%zu\n", static_cast<unsigned long long>(chunk.base),
         static_cast<unsigned long long>(chunk.count), decoded.count);
    return false;
  }

  if(!decodeChunkProperties(chunk, bytes.data() + payloadStart, payloadSize, decoded))
    return false;

  offset = payloadStart + payloadSize;
  return true;
}

bool parseRadFile(const std::vector<uint8_t>& bytes, RadDecoded& decoded, std::function<void(float)> progressCallback)
{
  if(bytes.size() < 8)
  {
    LOGE("RAD: file too small\n");
    return false;
  }

  size_t   offset = 0;
  uint32_t magic  = readU32LE(bytes.data());

  if(progressCallback)
    progressCallback(0.02f);

  if(magic == RAD_MAGIC)
  {
    uint32_t metaLen = readU32LE(bytes.data() + 4);
    size_t   metaEnd = 8 + roundUp8(metaLen);
    if(metaEnd > bytes.size())
    {
      LOGE("RAD: truncated top metadata\n");
      return false;
    }

    std::string metaText(reinterpret_cast<const char*>(bytes.data() + 8), metaLen);
    size_t      count   = 0;
    int         maxSh   = 0;
    bool        lodTree = false;
    if(!parseTopMeta(metaText, count, maxSh, lodTree))
      return false;

    decoded.init(count, maxSh, lodTree);
    offset = metaEnd;
  }
  else if(magic == RAD_CHUNK_MAGIC)
  {
    offset = 0;
  }
  else
  {
    LOGE("RAD: invalid magic 0x%08x\n", magic);
    return false;
  }

  size_t chunkIndex = 0;
  while(offset < bytes.size())
  {
    if(offset + 4 > bytes.size())
      break;

    uint32_t chunkMagic = readU32LE(bytes.data() + offset);
    if(chunkMagic != RAD_CHUNK_MAGIC)
    {
      if(offset == bytes.size())
        break;
      LOGE("RAD: unexpected data after chunks at offset %zu (magic=0x%08x)\n", offset, chunkMagic);
      return false;
    }

    if(!parseChunkAt(bytes, offset, decoded, magic == RAD_CHUNK_MAGIC))
      return false;

    chunkIndex++;
    if(progressCallback)
    {
      float p = 0.05f + 0.90f * (float(offset) / float(bytes.size()));
      progressCallback(std::clamp(p, 0.0f, 0.99f));
    }
  }

  if(!decoded.allocated)
  {
    LOGE("RAD: no chunks decoded\n");
    return false;
  }

  LOGI("RAD: decoded %zu splats from %zu chunk(s), maxSh=%d, lodTree=%d\n", decoded.count, chunkIndex, decoded.maxSh,
       decoded.lodTree ? 1 : 0);
  return true;
}

bool buildOutputSplatSet(const RadDecoded& decoded, SplatSet& output)
{
  output.clear();

  std::vector<size_t> keep;
  keep.reserve(decoded.count);

  size_t interiorCount = 0;
  if(decoded.lodTree && decoded.hasChildCount && decoded.hasChildStart)
  {
    for(size_t i = 0; i < decoded.count; ++i)
    {
      if(decoded.childCount[i] == 0)
      {
        keep.push_back(i);
      }
      else
      {
        interiorCount++;
      }
    }
    LOGI("RAD: keeping %zu leaf splats (dropping %zu interior LOD splats)\n", keep.size(), interiorCount);
  }
  else
  {
    if(decoded.lodTree)
    {
      LOGW("RAD: lodTree present but child arrays missing; loading all splats\n");
    }
    keep.resize(decoded.count);
    std::iota(keep.begin(), keep.end(), size_t(0));
  }

  if(keep.empty())
  {
    LOGE("RAD: no splats selected for output\n");
    return false;
  }

  const size_t outCount = keep.size();
  const size_t bands    = bandsForMaxSh(decoded.maxSh);

  output.positions.resize(outCount * 3);
  output.f_dc.resize(outCount * 3);
  output.f_rest.resize(outCount * bands * 3);
  output.opacity.resize(outCount);
  output.scale.resize(outCount * 3);
  output.rotation.resize(outCount * 4);

  size_t lodOpacityLeaves = 0;
  for(size_t outIdx = 0; outIdx < outCount; ++outIdx)
  {
    size_t inIdx = keep[outIdx];

    std::copy_n(decoded.center.data() + inIdx * 3, 3, output.positions.data() + outIdx * 3);
    std::copy_n(decoded.rgb.data() + inIdx * 3, 3, output.f_dc.data() + outIdx * 3);
    std::copy_n(decoded.scales.data() + inIdx * 3, 3, output.scale.data() + outIdx * 3);

    const float qx                  = decoded.quat[inIdx * 4 + 0];
    const float qy                  = decoded.quat[inIdx * 4 + 1];
    const float qz                  = decoded.quat[inIdx * 4 + 2];
    const float qw                  = decoded.quat[inIdx * 4 + 3];
    output.rotation[outIdx * 4 + 0] = qw;
    output.rotation[outIdx * 4 + 1] = qx;
    output.rotation[outIdx * 4 + 2] = qy;
    output.rotation[outIdx * 4 + 3] = qz;

    float alpha = decoded.opacity[inIdx];
    if(alpha > 1.0f)
    {
      lodOpacityLeaves++;
      alpha = 1.0f;
    }
    output.opacity[outIdx] = sigmoidInv(alpha);

    if(bands > 0)
    {
      const float* inSh  = decoded.shCoeffRgb.data() + inIdx * bands * 3;
      float*       outSh = output.f_rest.data() + outIdx * bands * 3;
      for(size_t j = 0; j < bands; ++j)
      {
        outSh[j]             = inSh[j * 3 + 0];
        outSh[bands + j]     = inSh[j * 3 + 1];
        outSh[2 * bands + j] = inSh[j * 3 + 2];
      }
    }
  }

  if(lodOpacityLeaves > 0)
  {
    LOGW("RAD: %zu selected leaf splats had alpha > 1.0; clamped before logit conversion\n", lodOpacityLeaves);
  }

  output.convertCoordinates(spz::CoordinateSystem::RDF, spz::CoordinateSystem::RUB);

  return output.size() > 0;
}

}  // namespace

bool RadLoader::canLoad(const std::filesystem::path& filename)
{
  if(!hasExtensionLower(filename, ".rad"))
    return false;

  auto data = readFile(filename);
  if(data.size() < 4)
    return false;

  uint32_t magic = readU32LE(data.data());
  return magic == RAD_MAGIC || magic == RAD_CHUNK_MAGIC;
}

bool RadLoader::load(const std::filesystem::path& filename, SplatSet& output, std::function<void(float)> progressCallback)
{
  auto startTime = std::chrono::high_resolution_clock::now();

  auto data = readFile(filename);
  if(data.empty())
  {
    LOGE("RAD: failed to read file %s\n", filename.string().c_str());
    return false;
  }

  RadDecoded decoded;
  if(!parseRadFile(data, decoded, progressCallback))
    return false;

  if(!buildOutputSplatSet(decoded, output))
  {
    LOGE("RAD: failed to build output splat set\n");
    return false;
  }

  if(progressCallback)
    progressCallback(1.0f);

  auto      endTime  = std::chrono::high_resolution_clock::now();
  long long loadTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
  LOGI("RAD file loaded in %lldms (%zu splats, SH=%d)%s\n", loadTime, output.size(), output.maxShDegree(),
       decoded.lodTree ? " [leafs from lodTree]" : "");
  return true;
}

}  // namespace vk_viewer
