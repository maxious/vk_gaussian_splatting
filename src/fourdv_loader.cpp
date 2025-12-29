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

#include "fourdv_loader.h"
#include <nvutils/logger.hpp>
#include "miniply.h"
#include <cmath>

namespace vk_gaussian_splatting {

// Helper for normalizing unpacking
inline float normalize(uint32_t value, uint32_t bits)
{
  return float(value) / float((1 << bits) - 1);
}

// Helper for lerp unpacking
inline float unpackLerp(float minVal, float maxVal, uint32_t value, uint32_t bits)
{
  return minVal + (maxVal - minVal) * normalize(value, bits);
}

bool FourDvLoader::load(const std::filesystem::path& filename, SplatSet& output, std::function<void(float)> progressCallback)
{
  miniply::PLYReader reader(filename.string().c_str());
  if (!reader.valid())
  {
    LOGE("Error: failed to open 4DV file: %s\n", filename.string().c_str());
    return false;
  }

  std::vector<ChunkInfo> chunks;
  std::vector<PackedVertex> packedVertices;
  std::vector<uint8_t> shData;
  uint32_t chunkIndices[28];
  uint32_t vertexIndices[6];
  uint32_t shIndices[45];

  bool chunkFound = false;
  bool vertexFound = false;
  bool shFound = false;

  while (reader.has_element())
  {
    if (reader.element_is("chunk") && reader.load_element())
    {
      const uint32_t numChunks = reader.num_rows();
      chunks.resize(numChunks);
      
      if (reader.find_properties(chunkIndices, 28, 
          "min_x", "max_x", "min_y", "max_y", "min_z", "max_z",
          "min_scale_x", "max_scale_x", "min_scale_y", "max_scale_y", "min_scale_z", "max_scale_z",
          "min_r", "max_r", "min_g", "max_g", "min_b", "max_b",
          "min_motion_x", "max_motion_x", "min_motion_y", "max_motion_y", "min_motion_z", "max_motion_z",
          "min_time_scale", "max_time_scale", "min_time", "max_time"))
      {
        // Extract directly into struct memory layout assuming it matches standard float packing
        // This is risky if struct packing is different, so safer to extract to temp buffers or property by property
        // For simplicity and safety let's assume miniply extracts floats correctly
        // We'll read component by component to be safe
        std::vector<float> data(numChunks);
        
        // Define a macro to reduce boilerplate
        #define EXTRACT_CHUNK_PROP(propName, memberName) \
          reader.extract_properties(chunkIndices + propName, 1, miniply::PLYPropertyType::Float, data.data()); \
          for(uint32_t i=0; i<numChunks; ++i) chunks[i].memberName = data[i];

        // Indices mapped manually based on find_properties order
        // 0: min_x, 1: max_x, etc.
        EXTRACT_CHUNK_PROP(0, min_x); EXTRACT_CHUNK_PROP(1, max_x);
        EXTRACT_CHUNK_PROP(2, min_y); EXTRACT_CHUNK_PROP(3, max_y);
        EXTRACT_CHUNK_PROP(4, min_z); EXTRACT_CHUNK_PROP(5, max_z);
        EXTRACT_CHUNK_PROP(6, min_scale_x); EXTRACT_CHUNK_PROP(7, max_scale_x);
        EXTRACT_CHUNK_PROP(8, min_scale_y); EXTRACT_CHUNK_PROP(9, max_scale_y);
        EXTRACT_CHUNK_PROP(10, min_scale_z); EXTRACT_CHUNK_PROP(11, max_scale_z);
        EXTRACT_CHUNK_PROP(12, min_r); EXTRACT_CHUNK_PROP(13, max_r);
        EXTRACT_CHUNK_PROP(14, min_g); EXTRACT_CHUNK_PROP(15, max_g);
        EXTRACT_CHUNK_PROP(16, min_b); EXTRACT_CHUNK_PROP(17, max_b);
        EXTRACT_CHUNK_PROP(18, min_motion_x); EXTRACT_CHUNK_PROP(19, max_motion_x);
        EXTRACT_CHUNK_PROP(20, min_motion_y); EXTRACT_CHUNK_PROP(21, max_motion_y);
        EXTRACT_CHUNK_PROP(22, min_motion_z); EXTRACT_CHUNK_PROP(23, max_motion_z);
        EXTRACT_CHUNK_PROP(24, min_time_scale); EXTRACT_CHUNK_PROP(25, max_time_scale);
        EXTRACT_CHUNK_PROP(26, min_time); EXTRACT_CHUNK_PROP(27, max_time);

        #undef EXTRACT_CHUNK_PROP
        chunkFound = true;
      }
    }
    else if (reader.element_is("vertex") && reader.load_element())
    {
      const uint32_t numVerts = reader.num_rows();
      packedVertices.resize(numVerts);
      
      if (reader.find_properties(vertexIndices, 6, "packed_position", "packed_rotation", "packed_scale", "packed_color", "packed_motion", "packed_time"))
      {
        // Extract directly into struct since these are all uint32
        // We handle member offset manually to be safe
        std::vector<uint32_t> data(numVerts);

        #define EXTRACT_VERT_PROP(propIdx, memberName) \
          reader.extract_properties(vertexIndices + propIdx, 1, miniply::PLYPropertyType::Int, data.data()); \
          for(uint32_t i=0; i<numVerts; ++i) packedVertices[i].memberName = data[i];

        EXTRACT_VERT_PROP(0, packed_position);
        EXTRACT_VERT_PROP(1, packed_rotation);
        EXTRACT_VERT_PROP(2, packed_scale);
        EXTRACT_VERT_PROP(3, packed_color);
        EXTRACT_VERT_PROP(4, packed_motion);
        EXTRACT_VERT_PROP(5, packed_time);

        #undef EXTRACT_VERT_PROP
        vertexFound = true;
      }
    }
    else if (reader.element_is("sh") && reader.load_element())
    {
      const uint32_t numVerts = reader.num_rows();
      // SH has 45 coeffs * 1 byte
      shData.resize(numVerts * 45);
      
      // We need to generate the 45 property names "f_rest_0"..."f_rest_44"
      // Since miniply requires varargs, we'll just extract them in a loop or batches if possible
      // Actually, miniply extract_properties supports array of indices
      // But we need to find them first. 
      // Let's assume they are contiguous and standard names as seen in the header dump
      
      // Manual extraction for simplicity, one by one is slow but robust
      std::vector<const char*> propNames;
      std::vector<std::string> propNameStrings; // Keep strings alive
      propNameStrings.reserve(45);
      for(int i=0; i<45; ++i) {
        propNameStrings.push_back("f_rest_" + std::to_string(i));
        propNames.push_back(propNameStrings.back().c_str());
      }

      if(reader.find_properties(shIndices, 45, 
        propNames[0], propNames[1], propNames[2], propNames[3], propNames[4], 
        propNames[5], propNames[6], propNames[7], propNames[8], propNames[9],
        propNames[10], propNames[11], propNames[12], propNames[13], propNames[14],
        propNames[15], propNames[16], propNames[17], propNames[18], propNames[19],
        propNames[20], propNames[21], propNames[22], propNames[23], propNames[24],
        propNames[25], propNames[26], propNames[27], propNames[28], propNames[29],
        propNames[30], propNames[31], propNames[32], propNames[33], propNames[34],
        propNames[35], propNames[36], propNames[37], propNames[38], propNames[39],
        propNames[40], propNames[41], propNames[42], propNames[43], propNames[44]))
      {
          reader.extract_properties(shIndices, 45, miniply::PLYPropertyType::UChar, shData.data());
          shFound = true;
      }
    }
    
    if (progressCallback) progressCallback(0.1f); // Just a little pulse
    reader.next_element();
  }

  if (!chunkFound || !vertexFound)
  {
    LOGE("Error: 4DV file missing chunks or vertices\n");
    return false;
  }

  // Decompress
  const size_t numSplats = packedVertices.size();
  output.clear();
  output.has_time_data = true;
  output.positions.resize(numSplats * 3);
  output.rotation.resize(numSplats * 4);
  output.scale.resize(numSplats * 3);
  output.f_dc.resize(numSplats * 3);
  output.f_rest.resize(numSplats * 45); // Assuming SH45
  output.opacity.resize(numSplats);
  output.motion.resize(numSplats * 3);
  output.time.resize(numSplats);
  output.time_scale.resize(numSplats);
  
  output.minTime = std::numeric_limits<float>::max();
  output.maxTime = std::numeric_limits<float>::lowest();

  const uint32_t chunkSize = 256;

  for(size_t i = 0; i < numSplats; ++i)
  {
    const size_t chunkIdx = i / chunkSize;
    if (chunkIdx >= chunks.size()) break;

    const ChunkInfo& chunk = chunks[chunkIdx];
    
    if (i % chunkSize == 0) {
        output.minTime = std::min(output.minTime, chunk.min_time);
        output.maxTime = std::max(output.maxTime, chunk.max_time);
    }

    const PackedVertex& vert = packedVertices[i];

    // --- Position (11-10-11 bits) ---
    uint32_t px = (vert.packed_position >> 21) & 0x7FF; // 11 bits
    uint32_t py = (vert.packed_position >> 11) & 0x3FF; // 10 bits
    uint32_t pz = vert.packed_position & 0x7FF;         // 11 bits
    output.positions[i*3 + 0] = unpackLerp(chunk.min_x, chunk.max_x, px, 11);
    output.positions[i*3 + 1] = unpackLerp(chunk.min_y, chunk.max_y, py, 10);
    output.positions[i*3 + 2] = unpackLerp(chunk.min_z, chunk.max_z, pz, 11);

    // --- Rotation (30 bits payload + 2 bits index at MSB) ---
    uint32_t r0    = (vert.packed_rotation >> 20) & 0x3FF; // 10 bits
    uint32_t r1    = (vert.packed_rotation >> 10) & 0x3FF; // 10 bits
    uint32_t r2    = vert.packed_rotation & 0x3FF;         // 10 bits
    uint32_t r_idx = (vert.packed_rotation >> 30) & 0x3;   // 2 bits (MSB)
    
    // Decoding quaternion from 3 smallest components
    // Map 0..1023 to -0.707..0.707 (since largest is at least 0.5)
    // Actually standard SOG maps to range [-1/sqrt(2), 1/sqrt(2)]
    float a = (normalize(r0, 10) * 1.41421356f) - 0.70710678f;
    float b = (normalize(r1, 10) * 1.41421356f) - 0.70710678f;
    float c = (normalize(r2, 10) * 1.41421356f) - 0.70710678f;
    float d = std::sqrt(std::max(0.0f, 1.0f - (a*a + b*b + c*c)));

    float q_x, q_y, q_z, q_w;
    
    switch(r_idx) {
        case 0:
            q_x = a; q_y = b; q_z = c; q_w = d; 
            break;
        case 1:
            q_x = d; q_y = b; q_z = c; q_w = a;
            break;
        case 2:
            q_x = b; q_y = d; q_z = c; q_w = a;
            break;
        case 3:
            q_x = b; q_y = c; q_z = d; q_w = a;
            break;
    }
    
    output.rotation[i*4 + 0] = q_w;
    output.rotation[i*4 + 1] = q_x;
    output.rotation[i*4 + 2] = q_y;
    output.rotation[i*4 + 3] = q_z;

    // --- Scale (Log space? Linear?) ---
    // Chunk has min/max scale, so likely linear interpolation of log-scale or direct scale
    // 32 bits -> 11-10-11 packing to match JS
    uint32_t sx = (vert.packed_scale >> 21) & 0x7FF; // 11 bits
    uint32_t sy = (vert.packed_scale >> 11) & 0x3FF; // 10 bits
    uint32_t sz = vert.packed_scale & 0x7FF;         // 11 bits
    
    float lsx = unpackLerp(chunk.min_scale_x, chunk.max_scale_x, sx, 11);
    float lsy = unpackLerp(chunk.min_scale_y, chunk.max_scale_y, sy, 10);
    float lsz = unpackLerp(chunk.min_scale_z, chunk.max_scale_z, sz, 11);
    
    output.scale[i*3 + 0] = (chunk.min_scale_x >= 0.0f) ? std::log(std::max(lsx, 1e-7f)) : lsx;
    output.scale[i*3 + 1] = (chunk.min_scale_y >= 0.0f) ? std::log(std::max(lsy, 1e-7f)) : lsy;
    output.scale[i*3 + 2] = (chunk.min_scale_z >= 0.0f) ? std::log(std::max(lsz, 1e-7f)) : lsz;

    // --- Color / Opacity ---
    // packed_color: 8R 8G 8B 8A(opacity)
    uint32_t cr = (vert.packed_color >> 24) & 0xFF;
    uint32_t cg = (vert.packed_color >> 16) & 0xFF;
    uint32_t cb = (vert.packed_color >> 8) & 0xFF;
    uint32_t ca = vert.packed_color & 0xFF;
    
    // Colors are relative to chunk min/max for high precision?
    // Or just absolute 8-bit?
    // Header has min_r/max_r... so it IS relative!
    // But 8 bits? That's just standard quantization.
    // Let's use the chunk bounds.
    float fcr = unpackLerp(chunk.min_r, chunk.max_r, cr, 8);
    float fcg = unpackLerp(chunk.min_g, chunk.max_g, cg, 8);
    float fcb = unpackLerp(chunk.min_b, chunk.max_b, cb, 8);
    
    const float SH_C0 = 0.28209479177387814f;
    output.f_dc[i*3 + 0] = (fcr - 0.5f) / SH_C0;
    output.f_dc[i*3 + 1] = (fcg - 0.5f) / SH_C0;
    output.f_dc[i*3 + 2] = (fcb - 0.5f) / SH_C0;
    
    float op = unpackLerp(0.0f, 1.0f, ca, 8);
    op = std::clamp(op, 1.0f/255.0f, 254.0f/255.0f);
    output.opacity[i] = std::log(op / (1.0f - op));

    // --- Motion ---
    // 11-10-11 packing
    uint32_t mx = (vert.packed_motion >> 21) & 0x7FF;
    uint32_t my = (vert.packed_motion >> 11) & 0x3FF;
    uint32_t mz = vert.packed_motion & 0x7FF;
    output.motion[i*3 + 0] = unpackLerp(chunk.min_motion_x, chunk.max_motion_x, mx, 11);
    output.motion[i*3 + 1] = unpackLerp(chunk.min_motion_y, chunk.max_motion_y, my, 10);
    output.motion[i*3 + 2] = unpackLerp(chunk.min_motion_z, chunk.max_motion_z, mz, 11);

    // --- Time ---
    // packed_time: 11-10-11 packing (matching JS unpack111011 usage on packedT.y)
    // MSB 11 bits: Scale
    // Middle 10 bits: Time
    // LSB 11 bits: Unused? (JS uses it as 3rd component of mix, but mix vector is 0..1 constant? No, mix(..., unpack111011))
    // Actually JS: vec3 timeData = mix(vec3(z, x, 0), vec3(w, y, 1), unpack(...))
    // Component 1 (MSB 11): Mixes z -> w (min_time_scale -> max_time_scale) => Time Scale
    // Component 2 (Mid 10): Mixes x -> y (min_time -> max_time) => Time Center
    // Component 3 (LSB 11): Mixes 0 -> 1 => Unused / 1.0?
    
    uint32_t t_scale_bits  = (vert.packed_time >> 21) & 0x7FF; // 11 bits
    uint32_t t_center_bits = (vert.packed_time >> 11) & 0x3FF; // 10 bits
    
    output.time[i]       = unpackLerp(chunk.min_time, chunk.max_time, t_center_bits, 10);
    output.time_scale[i] = unpackLerp(chunk.min_time_scale, chunk.max_time_scale, t_scale_bits, 11);

    // --- SH Rest ---
    if(shFound)
    {
      // Copy SH data, normalizing 0..255 to -1..1 or similar?
      // SH coeffs in PLY are typically floats.
      // Quantized SH usually maps byte 0..255 to -1..1 or dynamic range?
      // Standard SOG uses separate codebooks.
      // Here we have "sh" element with uchar properties.
      // Without min/max for SH in chunk, this implies a fixed range.
      // Common range is [-0.5, 0.5] or [-1, 1].
      // Let's guess [-2.0, 2.0] covers most SH.
      // Or maybe it's just raw byte and shader handles it?
      // Let's use [-1, 1] for now.
      for(int k=0; k<45; ++k)
      {
        uint8_t val = shData[i*45 + k];
        // (val / 255) * 2 - 1 ??
        // Actually, if we look at SOG, it uses quantization.
        output.f_rest[i*45 + k] = (float(val) / 255.0f) * 2.0f - 1.0f; // Guesswork
      }
    }
    
    if (progressCallback && (i % 10000 == 0)) 
    {
        progressCallback(float(i) / float(numSplats));
    }
  }
  
  // Coordinate system conversion: RDF to RUB (standard internal format)
  // output.convertCoordinates(spz::CoordinateSystem::RDF, spz::CoordinateSystem::RUB);
  
  return true;
}

} // namespace vk_gaussian_splatting
