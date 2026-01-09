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

#include <chrono>
#include <memory>
#include <string>

#include "ply_sequence_loader.h"
#include "audio_player.h"

namespace vk_gaussian_splatting {

/**
 * @brief Playback state for animation
 */
enum class PlaybackState {
    STOPPED,
    PLAYING,
    PAUSED
};

/**
 * @brief Animation controller for synchronized PLY sequence and audio playback
 * 
 * This class coordinates frame loading and audio playback to provide
 * synchronized video playback of PLY sequences with audio tracks.
 * 
 * Features:
 * - Frame-by-frame playback with configurable frame rate
 * - Audio synchronization with volume control
 * - Play/pause/stop/seek controls
 * - Loop playback
 * - Real-time position tracking
 */
class AnimationController {
public:
    AnimationController() = default;
    ~AnimationController() = default;

    /**
     * @brief Load a PLY sequence directory
     * @param dirPath Path to directory containing frame_*.ply and audio files
     * @return true if successfully loaded
     */
    bool loadSequence(const std::filesystem::path& dirPath, float frameRate = 30.0f);

    /**
     * @brief Close and unload current sequence
     */
    void closeSequence();

    /**
     * @brief Start or resume playback
     */
    void play();

    /**
     * @brief Pause playback
     */
    void pause();

    /**
     * @brief Stop playback and reset to beginning
     */
    void stop();

    /**
     * @brief Seek to specific timestamp
     * @param timestampMs Target position in milliseconds
     */
    void seek(uint64_t timestampMs);

    /**
     * @brief Set playback speed (0.5x, 1.0x, 2.0x, etc.)
     */
    void setPlaybackSpeed(float speed);

    /**
     * @brief Get current playback speed
     */
    float getPlaybackSpeed() const { return m_playbackSpeed; }

    /**
     * @brief Enable/disable looping
     */
    void setLooping(bool enabled);

    /**
     * @brief Check if looping is enabled
     */
    bool isLooping() const { return m_isLooping; }

    /**
     * @brief Get current playback state
     */
    PlaybackState getPlaybackState() const { return m_playbackState; }

    /**
     * @brief Get current frame index
     */
    size_t getCurrentFrame() const { return m_currentFrame; }

    /**
     * @brief Get total frame count
     */
    size_t getTotalFrames() const { return m_totalFrames; }

    /**
     * @brief Get current position in milliseconds
     */
    uint64_t getCurrentPositionMs() const { return m_currentPositionMs; }

    /**
     * @brief Get total duration in milliseconds
     */
    uint64_t getTotalDurationMs() const { return m_totalDurationMs; }

    /**
     * @brief Get audio player (for volume control, etc.)
     */
    AudioPlayer* getAudioPlayer() { return m_audioPlayer.get(); }

    /**
     * @brief Update playback state (call during main loop)
     * @param deltaTime Time since last update in milliseconds
     */
    void update(float deltaTime);

private:
    void seekToFrame(size_t frameIndex);
    void synchronizeAudio();
    void updatePosition();
    
    std::unique_ptr<PlySequenceLoader>       m_plyLoader;
    std::unique_ptr<AudioPlayer>           m_audioPlayer;
    
    PlaybackState                          m_playbackState = PlaybackState::STOPPED;
    
    size_t                                 m_currentFrame = 0;
    uint64_t                                m_currentPositionMs = 0;
    uint64_t                                m_totalDurationMs = 0;
    size_t                                 m_totalFrames = 0;
    
    float                                   m_playbackSpeed = 1.0f;
    bool                                    m_isLooping = false;
    bool                                    m_hasAudio = false;
    
    // Timing for smooth playback
    uint64_t                                m_accumulatedTime = 0;
    float                                   m_frameTimeAccumulator = 0.0f;
};

} // namespace vk_gaussian_splatting