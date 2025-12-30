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

#pragma once

#include <filesystem>
#include <vector>
#include <cstdint>
#include "depth_parser.h"

/**
 * @brief Loads a single VDZ (Vulkan Depth Z) file from disk
 * 
 * VDZ files contain depth data in one of two formats:
 * - VDZ1: Uncompressed uint16 depth values
 * - VDZ2: zlib-compressed uint16 depth values
 */
class VDZLoader {
public:
    /**
     * @brief Load a VDZ file from disk
     * 
     * @param filename Path to the .vdz file
     * @param outFrame Output DepthFrame structure containing parsed depth data
     * @return true if successfully loaded and parsed, false otherwise
     */
    static bool loadVDZFile(const std::filesystem::path& filename, DepthFrame& outFrame);
};
