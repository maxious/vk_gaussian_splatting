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

#include "ply_sequence_detector.h"
#include <algorithm>
#include <regex>

namespace vk_gaussian_splatting {

PlySequenceDetector::DetectionResult PlySequenceDetector::detect(const std::filesystem::path& path) {
    DetectionResult result;
    
    if (!std::filesystem::exists(path)) {
        return result;
    }
    
    if (std::filesystem::is_directory(path)) {
        // Check if it's a PLY sequence directory
        size_t frameCount = countFramesInDirectory(path);
        if (frameCount > 0) {
            result.formatType = FormatType::PLY_SEQUENCE;
            result.sequencePath = path;
            result.frameCount = frameCount;
            result.durationMs = static_cast<uint64_t>(frameCount * 1000.0f / result.frameRate);
            
            // Check for audio file
            std::filesystem::path audioPath = findAudioInDirectory(path);
            if (!audioPath.empty()) {
                result.hasAudio = true;
                result.audioPath = audioPath;
            }
        }
    } else if (std::filesystem::is_regular_file(path)) {
        // Check if it's a single PLY file
        if (isSinglePly(path)) {
            result.formatType = FormatType::SINGLE_PLY;
            result.sequencePath = path;
        }
        // Check if it's an audio file
        else if (isAudioFile(path)) {
            result.formatType = FormatType::AUDIO_ONLY;
            result.audioPath = path;
        }
    }
    
    return result;
}

bool PlySequenceDetector::isPlySequence(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path) || !std::filesystem::is_directory(path)) {
        return false;
    }
    
    return countFramesInDirectory(path) > 0;
}

bool PlySequenceDetector::isSinglePly(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path) || !std::filesystem::is_regular_file(path)) {
        return false;
    }
    
    return path.extension() == ".ply";
}

bool PlySequenceDetector::isAudioFile(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path) || !std::filesystem::is_regular_file(path)) {
        return false;
    }
    
    static const std::vector<std::string> audioExtensions = {
        ".wav", ".mp3", ".aac", ".ogg", ".flac", ".m4a", ".wma"
    };
    
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    return std::find(audioExtensions.begin(), audioExtensions.end(), ext) != audioExtensions.end();
}

const char* PlySequenceDetector::getFormatName(FormatType format) {
    switch (format) {
        case FormatType::UNKNOWN:
            return "Unknown";
        case FormatType::SINGLE_PLY:
            return "Single PLY File";
        case FormatType::PLY_SEQUENCE:
            return "PLY Sequence";
        case FormatType::AUDIO_ONLY:
            return "Audio File";
        default:
            return "Unknown";
    }
}

bool PlySequenceDetector::matchesFramePattern(const std::string& filename) {
    // Pattern: frame_XXXXXXXX.ply (e.g., frame_00000001.ply)
    if (filename.length() < 6 || filename.substr(0, 6) != "frame_") {
        return false;
    }
    
    if (filename.length() < 10 || filename.substr(filename.length() - 4) != ".ply") {
        return false;
    }
    
    // Check that the middle part is all digits
    std::string numberPart = filename.substr(6, filename.length() - 10);
    return !numberPart.empty() && 
           std::all_of(numberPart.begin(), numberPart.end(), ::isdigit);
}

size_t PlySequenceDetector::countFramesInDirectory(const std::filesystem::path& dirPath) {
    if (!std::filesystem::exists(dirPath) || !std::filesystem::is_directory(dirPath)) {
        return 0;
    }
    
    size_t count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(dirPath)) {
        if (entry.is_regular_file() && matchesFramePattern(entry.path().filename().string())) {
            count++;
        }
    }
    
    return count;
}

std::filesystem::path PlySequenceDetector::findAudioInDirectory(const std::filesystem::path& dirPath) {
    if (!std::filesystem::exists(dirPath) || !std::filesystem::is_directory(dirPath)) {
        return std::filesystem::path();
    }
    
    static const std::vector<std::string> audioExtensions = {
        ".wav", ".mp3", ".aac", ".ogg", ".flac", ".m4a", ".wma"
    };
    
    for (const auto& ext : audioExtensions) {
        std::filesystem::path audioPath = dirPath / ("audio" + ext);
        if (std::filesystem::exists(audioPath)) {
            return audioPath;
        }
    }
    
    return std::filesystem::path();
}

} // namespace vk_gaussian_splatting