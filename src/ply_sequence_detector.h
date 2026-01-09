/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under Apache License, Version 2.0 (the "License");
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
#include <string>

namespace vk_gaussian_splatting {

/**
 * @brief Detects file format and determines if it's a PLY sequence
 * 
 * This utility provides automatic detection of:
 * - Single PLY files (.ply)
 * - PLY sequence directories (frame_*.ply pattern)
 * - Supported audio files in sequence directories
 * 
 * Usage:
 * - Check if path is PLY sequence → use PlySequenceLoader
 * - Check if path is single PLY → use existing SplatLoaderFast
 * - Check if audio available → use AudioPlayer for sync
 */
class PlySequenceDetector {
public:
    /**
     * @brief Result of file format detection
     */
    enum class FormatType {
        UNKNOWN,        // Not a recognized format
        SINGLE_PLY,     // Single PLY file (frame_000000.ply or random name)
        PLY_SEQUENCE,   // Directory containing PLY sequence
        AUDIO_ONLY      // Audio file without PLY sequence
    };

    /**
     * @brief Detection result with metadata
     */
    struct DetectionResult {
        FormatType formatType = FormatType::UNKNOWN;
        std::filesystem::path sequencePath;
        std::filesystem::path audioPath;
        size_t frameCount = 0;
        float frameRate = 30.0f;
        uint64_t durationMs = 0;
        bool hasAudio = false;
    };

    /**
     * @brief Detect file format for a given path
     * @param path File or directory path to check
     * @return DetectionResult with format info
     */
    static DetectionResult detect(const std::filesystem::path& path);

    /**
     * @brief Check if path is a PLY sequence directory
     * @param path Directory to check
     * @return true if it's a valid PLY sequence directory
     */
    static bool isPlySequence(const std::filesystem::path& path);

    /**
     * @brief Check if path is a single PLY file
     * @param path File to check
     * @return true if it's a PLY file
     */
    static bool isSinglePly(const std::filesystem::path& path);

    /**
     * @brief Check if path is an audio file
     * @param path File to check
     * @return true if it's a supported audio format
     */
    static bool isAudioFile(const std::filesystem::path& path);

    /**
     * @brief Get human-readable format name
     * @param format Format type
     * @return String description
     */
    static const char* getFormatName(FormatType format);

private:
    static bool matchesFramePattern(const std::string& filename);
    static size_t countFramesInDirectory(const std::filesystem::path& dirPath);
    static std::filesystem::path findAudioInDirectory(const std::filesystem::path& dirPath);
};

} // namespace vk_gaussian_splatting