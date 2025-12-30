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

#include "vdz_loader.h"
#include <nvutils/logger.hpp>
#include <fstream>
#include <vector>

bool VDZLoader::loadVDZFile(const std::filesystem::path& filename, DepthFrame& outFrame)
{
    // Check file exists
    if (!std::filesystem::exists(filename)) {
        LOGE("VDZ file not found: %s\n", filename.string().c_str());
        return false;
    }

    // Read entire file into buffer
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        LOGE("Failed to open VDZ file: %s\n", filename.string().c_str());
        return false;
    }

    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(fileSize);
    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
    file.close();

    if (buffer.empty()) {
        LOGE("VDZ file is empty: %s\n", filename.string().c_str());
        return false;
    }

    // Parse using existing depth parser
    bool success = parseDepthFrame(buffer, outFrame);
    
    if (success) {
        LOGI("Successfully loaded VDZ file: %s (%ux%u pixels, timestamp=%ums)\n",
             filename.string().c_str(), outFrame.width, outFrame.height, outFrame.timestampMs);
    } else {
        LOGE("Failed to parse VDZ file: %s\n", filename.string().c_str());
    }

    return success;
}
