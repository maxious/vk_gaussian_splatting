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
 *
 * Unit tests for PLY material property parsing (basecolor, roughness, metallic).
 * Uses in-memory PLY data (no file I/O).
 */

#include "doctest.h"

#include <cstring>
#include <charconv>
#include <string_view>
#include <vector>

#include "splat_set.h"

namespace {

struct TestPropertyLayout
{
    size_t vertexCount  = 0;
    size_t vertexStride = 0;
    size_t basecolorOffset[3] = {static_cast<size_t>(-1), static_cast<size_t>(-1), static_cast<size_t>(-1)};
    size_t roughnessOffset = static_cast<size_t>(-1);
    size_t metallicOffset  = static_cast<size_t>(-1);

    size_t xOffset = static_cast<size_t>(-1);
    size_t yOffset = static_cast<size_t>(-1);
    size_t zOffset = static_cast<size_t>(-1);
    size_t f_dcOffset[3] = {static_cast<size_t>(-1), static_cast<size_t>(-1), static_cast<size_t>(-1)};
    size_t opacityOffset = static_cast<size_t>(-1);
    size_t scaleOffset[3] = {static_cast<size_t>(-1), static_cast<size_t>(-1), static_cast<size_t>(-1)};
    size_t rotOffset[4]   = {static_cast<size_t>(-1), static_cast<size_t>(-1), static_cast<size_t>(-1), static_cast<size_t>(-1)};
    size_t restOffset[45];
    int    restCount = 0;

    TestPropertyLayout()
    {
        for(int i = 0; i < 45; ++i)
            restOffset[i] = static_cast<size_t>(-1);
    }
};

// Mirrors splat_loader_fast.cpp::parseHeader
bool parseHeaderInMemory(std::string_view header, TestPropertyLayout& layout)
{
    const char* ptr = header.data();
    const char* end = header.data() + header.size();

    if(header.size() < 4 || header.substr(0, 3) != "ply")
        return false;

    bool isBinaryLE = false;
    bool inVertex   = false;

    while(ptr < end)
    {
        const char* lineStart = ptr;
        while(ptr < end && *ptr != '\n' && *ptr != '\r') ptr++;
        std::string_view line(lineStart, ptr - lineStart);

        if(ptr < end && *ptr == '\r') ptr++;
        if(ptr < end && *ptr == '\n') ptr++;

        if(line == "end_header")
            return isBinaryLE && layout.vertexCount > 0;

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
                if(idx >= 0 && idx < 3) layout.f_dcOffset[idx] = layout.vertexStride;
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
            else if(name.starts_with("basecolor_"))
            {
                int idx = 0;
                std::from_chars(name.data() + 10, name.data() + name.size(), idx);
                if(idx >= 0 && idx < 3) layout.basecolorOffset[idx] = layout.vertexStride;
            }
            else if(name == "roughness") layout.roughnessOffset = layout.vertexStride;
            else if(name == "metallic") layout.metallicOffset = layout.vertexStride;
            layout.vertexStride += 4;
        }
    }
    return false;
}

std::vector<uint8_t> buildPlyInMemory(const std::string& headerLines,
                                       const std::vector<std::vector<float>>& vertexData)
{
    std::vector<uint8_t> buffer;
    buffer.insert(buffer.end(), headerLines.begin(), headerLines.end());

    for(const auto& vertex : vertexData)
    {
        for(float f : vertex)
        {
            const auto* bytes = reinterpret_cast<const uint8_t*>(&f);
            buffer.insert(buffer.end(), bytes, bytes + sizeof(float));
        }
    }
    return buffer;
}

// Matches SplatLoaderFast::load material extraction logic
void populateSplatSetFromLayout(const TestPropertyLayout& layout,
                                 const uint8_t* vertexData,
                                 size_t vertexDataSize,
                                 vk_viewer::SplatSet& output)
{
    const size_t count  = layout.vertexCount;
    const size_t stride = layout.vertexStride;

    if(count == 0 || vertexDataSize < count * stride)
        return;

    output.clear();

    output.positions.resize(count * 3);
    output.f_dc.resize(count * 3);
    output.f_rest.resize(count * layout.restCount);
    output.opacity.resize(count);
    output.scale.resize(count * 3);
    output.rotation.resize(count * 4);

    for(size_t i = 0; i < count; ++i)
    {
        const uint8_t* base = vertexData + i * stride;

        if(layout.xOffset != static_cast<size_t>(-1))
            output.positions[i * 3 + 0] = *reinterpret_cast<const float*>(base + layout.xOffset);
        if(layout.yOffset != static_cast<size_t>(-1))
            output.positions[i * 3 + 1] = *reinterpret_cast<const float*>(base + layout.yOffset);
        if(layout.zOffset != static_cast<size_t>(-1))
            output.positions[i * 3 + 2] = *reinterpret_cast<const float*>(base + layout.zOffset);

        if(layout.opacityOffset != static_cast<size_t>(-1))
            output.opacity[i] = *reinterpret_cast<const float*>(base + layout.opacityOffset);

        if(layout.f_dcOffset[0] != static_cast<size_t>(-1))
            output.f_dc[i * 3 + 0] = *reinterpret_cast<const float*>(base + layout.f_dcOffset[0]);
        if(layout.f_dcOffset[1] != static_cast<size_t>(-1))
            output.f_dc[i * 3 + 1] = *reinterpret_cast<const float*>(base + layout.f_dcOffset[1]);
        if(layout.f_dcOffset[2] != static_cast<size_t>(-1))
            output.f_dc[i * 3 + 2] = *reinterpret_cast<const float*>(base + layout.f_dcOffset[2]);

        for(int r = 0; r < layout.restCount; ++r)
        {
            if(layout.restOffset[r] != static_cast<size_t>(-1))
                output.f_rest[i * layout.restCount + r] = *reinterpret_cast<const float*>(base + layout.restOffset[r]);
        }

        for(int s = 0; s < 3; ++s)
        {
            if(layout.scaleOffset[s] != static_cast<size_t>(-1))
                output.scale[i * 3 + s] = *reinterpret_cast<const float*>(base + layout.scaleOffset[s]);
        }

        for(int r = 0; r < 4; ++r)
        {
            if(layout.rotOffset[r] != static_cast<size_t>(-1))
                output.rotation[i * 4 + r] = *reinterpret_cast<const float*>(base + layout.rotOffset[r]);
        }
    }

    bool hasMaterialData = (layout.basecolorOffset[0] != static_cast<size_t>(-1)) ||
                           (layout.roughnessOffset != static_cast<size_t>(-1)) ||
                           (layout.metallicOffset != static_cast<size_t>(-1));

    if(hasMaterialData)
    {
        output.has_material_data = true;
        output.basecolor.resize(count * 3);
        output.roughness.resize(count);
        output.metallic.resize(count);

        for(size_t i = 0; i < count; ++i)
        {
            const uint8_t* base = vertexData + i * stride;

            if(layout.basecolorOffset[0] != static_cast<size_t>(-1))
            {
                output.basecolor[i * 3 + 0] = *reinterpret_cast<const float*>(base + layout.basecolorOffset[0]);
                output.basecolor[i * 3 + 1] = *reinterpret_cast<const float*>(base + layout.basecolorOffset[1]);
                output.basecolor[i * 3 + 2] = *reinterpret_cast<const float*>(base + layout.basecolorOffset[2]);
            }
            else
            {
                output.basecolor[i * 3 + 0] = output.f_dc[i * 3 + 0];
                output.basecolor[i * 3 + 1] = output.f_dc[i * 3 + 1];
                output.basecolor[i * 3 + 2] = output.f_dc[i * 3 + 2];
            }

            if(layout.roughnessOffset != static_cast<size_t>(-1))
                output.roughness[i] = *reinterpret_cast<const float*>(base + layout.roughnessOffset);
            else
                output.roughness[i] = 0.5f;

            if(layout.metallicOffset != static_cast<size_t>(-1))
                output.metallic[i] = *reinterpret_cast<const float*>(base + layout.metallicOffset);
            else
                output.metallic[i] = 0.0f;
        }
    }
}

std::string buildStandardGaussianHeader(size_t vertexCount)
{
    std::string hdr;
    hdr += "ply\n";
    hdr += "format binary_little_endian 1.0\n";
    hdr += "element vertex " + std::to_string(vertexCount) + "\n";
    hdr += "property float x\n";
    hdr += "property float y\n";
    hdr += "property float z\n";
    hdr += "property float f_dc_0\n";
    hdr += "property float f_dc_1\n";
    hdr += "property float f_dc_2\n";
    hdr += "property float opacity\n";
    hdr += "property float scale_0\n";
    hdr += "property float scale_1\n";
    hdr += "property float scale_2\n";
    hdr += "property float rot_0\n";
    hdr += "property float rot_1\n";
    hdr += "property float rot_2\n";
    hdr += "property float rot_3\n";
    for(int i = 0; i < 45; ++i)
        hdr += "property float f_rest_" + std::to_string(i) + "\n";
    hdr += "end_header\n";
    return hdr;
}

std::vector<float> buildStandardVertex(float x, float y, float z,
                                        float dc_r, float dc_g, float dc_b,
                                        float opacity_val,
                                        float sx, float sy, float sz,
                                        float rx, float ry, float rz, float rw)
{
    std::vector<float> v = {x, y, z, dc_r, dc_g, dc_b, opacity_val,
                             sx, sy, sz, rx, ry, rz, rw};
    for(int i = 0; i < 45; ++i)
        v.push_back(0.0f);
    return v;
}

} // namespace

TEST_SUITE("PLY Material")
{
    TEST_CASE("No material properties")
    {
        const size_t N = 2;
        std::string header = buildStandardGaussianHeader(N);

        std::vector<std::vector<float>> vertices;
        vertices.push_back(buildStandardVertex(1.0f, 2.0f, 3.0f,
                                                0.1f, 0.2f, 0.3f,
                                                0.9f,
                                                0.01f, 0.02f, 0.03f,
                                                0.0f, 0.0f, 0.0f, 1.0f));
        vertices.push_back(buildStandardVertex(4.0f, 5.0f, 6.0f,
                                                0.4f, 0.5f, 0.6f,
                                                0.8f,
                                                0.04f, 0.05f, 0.06f,
                                                0.0f, 0.0f, 0.0f, 1.0f));

        auto buffer = buildPlyInMemory(header, vertices);

        TestPropertyLayout layout;
        std::string_view headerView(reinterpret_cast<const char*>(buffer.data()),
                                     header.size());
        CHECK(parseHeaderInMemory(headerView, layout));
        CHECK(layout.vertexCount == N);
        CHECK(layout.vertexStride == 236U);

        CHECK(layout.basecolorOffset[0] == static_cast<size_t>(-1));
        CHECK(layout.basecolorOffset[1] == static_cast<size_t>(-1));
        CHECK(layout.basecolorOffset[2] == static_cast<size_t>(-1));
        CHECK(layout.roughnessOffset == static_cast<size_t>(-1));
        CHECK(layout.metallicOffset == static_cast<size_t>(-1));

        vk_viewer::SplatSet ss;
        populateSplatSetFromLayout(layout,
                                    buffer.data() + header.size(),
                                    buffer.size() - header.size(),
                                    ss);

        CHECK(ss.size() == N);
        CHECK_FALSE(ss.has_material_data);
        CHECK(ss.basecolor.empty());
        CHECK(ss.roughness.empty());
        CHECK(ss.metallic.empty());

        CHECK(ss.f_dc.size() == N * 3);
        CHECK(ss.f_dc[0] == doctest::Approx(0.1f));
        CHECK(ss.f_dc[1] == doctest::Approx(0.2f));
        CHECK(ss.f_dc[2] == doctest::Approx(0.3f));
    }

    TEST_CASE("All 6 material properties")
    {
        const size_t N = 1;
        std::string header;
        header += "ply\n";
        header += "format binary_little_endian 1.0\n";
        header += "element vertex " + std::to_string(N) + "\n";
        header += "property float x\n";
        header += "property float y\n";
        header += "property float z\n";
        header += "property float f_dc_0\n";
        header += "property float f_dc_1\n";
        header += "property float f_dc_2\n";
        header += "property float opacity\n";
        header += "property float scale_0\n";
        header += "property float scale_1\n";
        header += "property float scale_2\n";
        header += "property float rot_0\n";
        header += "property float rot_1\n";
        header += "property float rot_2\n";
        header += "property float rot_3\n";
        for(int i = 0; i < 45; ++i)
            header += "property float f_rest_" + std::to_string(i) + "\n";
        header += "property float basecolor_0\n";
        header += "property float basecolor_1\n";
        header += "property float basecolor_2\n";
        header += "property float roughness\n";
        header += "property float metallic\n";
        header += "end_header\n";

        auto v = buildStandardVertex(10.0f, 20.0f, 30.0f,
                                      0.25f, 0.35f, 0.45f,
                                      0.7f, 0.05f, 0.06f, 0.07f,
                                      0.0f, 0.0f, 0.0f, 1.0f);
        v.push_back(0.8f);
        v.push_back(0.3f);
        v.push_back(0.1f);
        v.push_back(0.4f);
        v.push_back(0.9f);

        auto buffer = buildPlyInMemory(header, {v});

        TestPropertyLayout layout;
        std::string_view headerView(reinterpret_cast<const char*>(buffer.data()),
                                     header.size());
        CHECK(parseHeaderInMemory(headerView, layout));

        CHECK(layout.basecolorOffset[0] != static_cast<size_t>(-1));
        CHECK(layout.basecolorOffset[1] != static_cast<size_t>(-1));
        CHECK(layout.basecolorOffset[2] != static_cast<size_t>(-1));
        CHECK(layout.roughnessOffset != static_cast<size_t>(-1));
        CHECK(layout.metallicOffset != static_cast<size_t>(-1));
        CHECK(layout.vertexStride == 256U);

        vk_viewer::SplatSet ss;
        populateSplatSetFromLayout(layout,
                                    buffer.data() + header.size(),
                                    buffer.size() - header.size(),
                                    ss);

        CHECK(ss.size() == N);
        CHECK(ss.has_material_data);

        CHECK(ss.basecolor.size() == N * 3);
        CHECK(ss.basecolor[0] == doctest::Approx(0.8f));
        CHECK(ss.basecolor[1] == doctest::Approx(0.3f));
        CHECK(ss.basecolor[2] == doctest::Approx(0.1f));

        CHECK(ss.roughness.size() == N);
        CHECK(ss.roughness[0] == doctest::Approx(0.4f));
        CHECK(ss.metallic.size() == N);
        CHECK(ss.metallic[0] == doctest::Approx(0.9f));

        CHECK(ss.f_dc[0] == doctest::Approx(0.25f));
        CHECK(ss.f_dc[1] == doctest::Approx(0.35f));
        CHECK(ss.f_dc[2] == doctest::Approx(0.45f));
    }

    TEST_CASE("Partial material - only roughness")
    {
        const size_t N = 2;
        std::string header;
        header += "ply\n";
        header += "format binary_little_endian 1.0\n";
        header += "element vertex " + std::to_string(N) + "\n";
        header += "property float x\n";
        header += "property float y\n";
        header += "property float z\n";
        header += "property float f_dc_0\n";
        header += "property float f_dc_1\n";
        header += "property float f_dc_2\n";
        header += "property float opacity\n";
        header += "property float scale_0\n";
        header += "property float scale_1\n";
        header += "property float scale_2\n";
        header += "property float rot_0\n";
        header += "property float rot_1\n";
        header += "property float rot_2\n";
        header += "property float rot_3\n";
        for(int i = 0; i < 45; ++i)
            header += "property float f_rest_" + std::to_string(i) + "\n";
        header += "property float roughness\n";
        header += "end_header\n";

        auto v0 = buildStandardVertex(1.0f, 1.0f, 1.0f,
                                       0.5f, 0.1f, 0.1f,
                                       0.8f, 0.1f, 0.1f, 0.1f,
                                       0.0f, 0.0f, 0.0f, 1.0f);
        v0.push_back(0.25f);

        auto v1 = buildStandardVertex(2.0f, 2.0f, 2.0f,
                                       0.1f, 0.5f, 0.1f,
                                       0.7f, 0.2f, 0.2f, 0.2f,
                                       0.0f, 0.0f, 0.0f, 1.0f);
        v1.push_back(0.75f);

        auto buffer = buildPlyInMemory(header, {v0, v1});

        TestPropertyLayout layout;
        std::string_view headerView(reinterpret_cast<const char*>(buffer.data()),
                                     header.size());
        CHECK(parseHeaderInMemory(headerView, layout));

        CHECK(layout.basecolorOffset[0] == static_cast<size_t>(-1));
        CHECK(layout.basecolorOffset[1] == static_cast<size_t>(-1));
        CHECK(layout.basecolorOffset[2] == static_cast<size_t>(-1));
        CHECK(layout.roughnessOffset != static_cast<size_t>(-1));
        CHECK(layout.metallicOffset == static_cast<size_t>(-1));

        vk_viewer::SplatSet ss;
        populateSplatSetFromLayout(layout,
                                    buffer.data() + header.size(),
                                    buffer.size() - header.size(),
                                    ss);

        CHECK(ss.size() == N);
        CHECK(ss.has_material_data);

        CHECK(ss.basecolor.size() == N * 3);
        CHECK(ss.basecolor[0] == doctest::Approx(0.5f));
        CHECK(ss.basecolor[1] == doctest::Approx(0.1f));
        CHECK(ss.basecolor[2] == doctest::Approx(0.1f));
        CHECK(ss.basecolor[3] == doctest::Approx(0.1f));
        CHECK(ss.basecolor[4] == doctest::Approx(0.5f));
        CHECK(ss.basecolor[5] == doctest::Approx(0.1f));

        CHECK(ss.roughness.size() == N);
        CHECK(ss.roughness[0] == doctest::Approx(0.25f));
        CHECK(ss.roughness[1] == doctest::Approx(0.75f));

        CHECK(ss.metallic.size() == N);
        CHECK(ss.metallic[0] == doctest::Approx(0.0f));
        CHECK(ss.metallic[1] == doctest::Approx(0.0f));
    }

    TEST_CASE("Default values - basecolor only, roughness and metallic fall back")
    {
        const size_t N = 3;
        std::string header;
        header += "ply\n";
        header += "format binary_little_endian 1.0\n";
        header += "element vertex " + std::to_string(N) + "\n";
        header += "property float x\n";
        header += "property float y\n";
        header += "property float z\n";
        header += "property float f_dc_0\n";
        header += "property float f_dc_1\n";
        header += "property float f_dc_2\n";
        header += "property float opacity\n";
        header += "property float scale_0\n";
        header += "property float scale_1\n";
        header += "property float scale_2\n";
        header += "property float rot_0\n";
        header += "property float rot_1\n";
        header += "property float rot_2\n";
        header += "property float rot_3\n";
        for(int i = 0; i < 45; ++i)
            header += "property float f_rest_" + std::to_string(i) + "\n";
        header += "property float basecolor_0\n";
        header += "property float basecolor_1\n";
        header += "property float basecolor_2\n";
        header += "end_header\n";

        std::vector<std::vector<float>> vertices;
        for(size_t i = 0; i < N; ++i)
        {
            float fi = static_cast<float>(i);
            auto v = buildStandardVertex(fi, fi, fi,
                                         0.1f + fi * 0.1f, 0.2f + fi * 0.1f, 0.3f + fi * 0.1f,
                                         0.5f + fi * 0.1f,
                                         0.01f, 0.01f, 0.01f,
                                         0.0f, 0.0f, 0.0f, 1.0f);
            v.push_back(1.0f - fi * 0.2f);
            v.push_back(0.5f + fi * 0.1f);
            v.push_back(fi * 0.3f);
            vertices.push_back(v);
        }

        auto buffer = buildPlyInMemory(header, vertices);

        TestPropertyLayout layout;
        std::string_view headerView(reinterpret_cast<const char*>(buffer.data()),
                                     header.size());
        CHECK(parseHeaderInMemory(headerView, layout));
        CHECK(layout.roughnessOffset == static_cast<size_t>(-1));
        CHECK(layout.metallicOffset == static_cast<size_t>(-1));

        vk_viewer::SplatSet ss;
        populateSplatSetFromLayout(layout,
                                    buffer.data() + header.size(),
                                    buffer.size() - header.size(),
                                    ss);

        CHECK(ss.size() == N);
        CHECK(ss.has_material_data);

        CHECK(ss.basecolor[0] == doctest::Approx(1.0f));
        CHECK(ss.basecolor[1] == doctest::Approx(0.5f));
        CHECK(ss.basecolor[2] == doctest::Approx(0.0f));

        CHECK(ss.roughness.size() == N);
        for(size_t i = 0; i < N; ++i)
            CHECK(ss.roughness[i] == doctest::Approx(0.5f));

        CHECK(ss.metallic.size() == N);
        for(size_t i = 0; i < N; ++i)
            CHECK(ss.metallic[i] == doctest::Approx(0.0f));

        CHECK(ss.f_dc[0] == doctest::Approx(0.1f));
        CHECK(ss.f_dc[1] == doctest::Approx(0.2f));
        CHECK(ss.f_dc[2] == doctest::Approx(0.3f));
    }

    TEST_CASE("Material data preserved through reorder")
    {
        vk_viewer::SplatSet ss;
        ss.positions = {0.0f, 0.0f, 0.0f,
                        10.0f, 10.0f, 10.0f,
                         5.0f,  5.0f,  5.0f};
        ss.f_dc = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f};
        ss.f_rest.resize(3 * 45, 0.0f);
        ss.opacity = {0.7f, 0.8f, 0.9f};
        ss.scale = {0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f, 0.07f, 0.08f, 0.09f};
        ss.rotation = {0.0f, 0.0f, 0.0f, 1.0f,
                        0.0f, 0.0f, 0.0f, 1.0f,
                        0.0f, 0.0f, 0.0f, 1.0f};

        ss.has_material_data = true;
        ss.basecolor = {1.0f, 0.0f, 0.0f,
                         0.0f, 1.0f, 0.0f,
                         0.0f, 0.0f, 1.0f};
        ss.roughness = {0.1f, 0.5f, 0.9f};
        ss.metallic  = {0.0f, 0.0f, 1.0f};

        CHECK(ss.size() == 3);

        ss.reorderByMortonCode();

        CHECK(ss.size() == 3);
        CHECK(ss.has_material_data);
        CHECK(ss.basecolor.size() == 9);
        CHECK(ss.roughness.size() == 3);
        CHECK(ss.metallic.size() == 3);

        float basecolorSum = 0.0f;
        for(float f : ss.basecolor) basecolorSum += f;
        CHECK(basecolorSum == doctest::Approx(3.0f));

        float roughnessSum = 0.0f;
        for(float f : ss.roughness) roughnessSum += f;
        CHECK(roughnessSum == doctest::Approx(1.5f));

        float metallicSum = 0.0f;
        for(float f : ss.metallic) metallicSum += f;
        CHECK(metallicSum == doctest::Approx(1.0f));

        for(size_t i = 0; i < 3; ++i)
        {
            CHECK(ss.roughness[i] >= 0.0f);
            CHECK(ss.metallic[i] >= 0.0f);
        }
    }

    TEST_CASE("Material data in empty SplatSet")
    {
        vk_viewer::SplatSet ss;
        CHECK(ss.size() == 0);
        CHECK_FALSE(ss.has_material_data);
        CHECK(ss.basecolor.empty());
        CHECK(ss.roughness.empty());
        CHECK(ss.metallic.empty());

        ss.has_material_data = true;
        ss.basecolor = {1.0f, 2.0f, 3.0f};
        ss.roughness = {0.5f};
        ss.metallic = {1.0f};
        ss.clear();

        CHECK_FALSE(ss.has_material_data);
        CHECK(ss.basecolor.empty());
        CHECK(ss.roughness.empty());
        CHECK(ss.metallic.empty());
    }

    TEST_CASE("Empty PLY file zero vertices")
    {
        std::string header;
        header += "ply\n";
        header += "format binary_little_endian 1.0\n";
        header += "element vertex 0\n";
        header += "property float x\n";
        header += "property float y\n";
        header += "property float z\n";
        header += "property float f_dc_0\n";
        header += "property float f_dc_1\n";
        header += "property float f_dc_2\n";
        header += "property float opacity\n";
        header += "property float basecolor_0\n";
        header += "property float basecolor_1\n";
        header += "property float basecolor_2\n";
        header += "property float roughness\n";
        header += "property float metallic\n";
        header += "end_header\n";

        TestPropertyLayout layout;
        std::string_view headerView(header.data(), header.size());
        CHECK_FALSE(parseHeaderInMemory(headerView, layout));
        CHECK(layout.vertexCount == 0);

        vk_viewer::SplatSet ss;
        populateSplatSetFromLayout(layout,
                                    reinterpret_cast<const uint8_t*>(header.data() + header.size()),
                                    0,
                                    ss);

        CHECK(ss.size() == 0);
        CHECK_FALSE(ss.has_material_data);
        CHECK(ss.basecolor.empty());
        CHECK(ss.roughness.empty());
        CHECK(ss.metallic.empty());
    }

    TEST_CASE("Stride calculation includes material properties")
    {
        auto countStride = [](const std::string& header) {
            TestPropertyLayout layout;
            std::string_view hv(header.data(), header.size());
            CHECK(parseHeaderInMemory(hv, layout));
            return layout.vertexStride;
        };

        std::string baseHeader;
        baseHeader += "ply\n";
        baseHeader += "format binary_little_endian 1.0\n";
        baseHeader += "element vertex 1\n";
        baseHeader += "property float x\n";
        baseHeader += "property float y\n";
        baseHeader += "property float z\n";
        baseHeader += "end_header\n";

        size_t baseStride = countStride(baseHeader);
        CHECK(baseStride == 12U);

        std::string withBC = baseHeader;
        withBC.insert(withBC.find("end_header"), "property float basecolor_0\n");
        withBC.insert(withBC.find("end_header"), "property float basecolor_1\n");
        withBC.insert(withBC.find("end_header"), "property float basecolor_2\n");
        size_t bcStride = countStride(withBC);
        CHECK(bcStride == baseStride + 12U);

        std::string withRough = withBC;
        withRough.insert(withRough.find("end_header"), "property float roughness\n");
        size_t roughStride = countStride(withRough);
        CHECK(roughStride == bcStride + 4U);

        std::string withMetal = withRough;
        withMetal.insert(withMetal.find("end_header"), "property float metallic\n");
        size_t metalStride = countStride(withMetal);
        CHECK(metalStride == roughStride + 4U);

        TestPropertyLayout fullLayout;
        std::string_view hv(withMetal.data(), withMetal.size());
        CHECK(parseHeaderInMemory(hv, fullLayout));
        CHECK(fullLayout.basecolorOffset[0] == 12);
        CHECK(fullLayout.basecolorOffset[1] == 16);
        CHECK(fullLayout.basecolorOffset[2] == 20);
        CHECK(fullLayout.roughnessOffset == 24);
        CHECK(fullLayout.metallicOffset == 28);
    }
}
