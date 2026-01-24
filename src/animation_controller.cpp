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

#include "animation_controller.h"
#include <nvutils/logger.hpp>

namespace vk_viewer {

AnimationController::~AnimationController() {
    closeSequence();
}

bool AnimationController::loadSequence(const std::filesystem::path& dirPath, float frameRate) {
    closeSequence();
    
    // Load PLY sequence
    m_plyLoader = std::make_unique<PlySequenceLoader>();
    if (!m_plyLoader->open(dirPath, frameRate)) {
        LOGE("AnimationController: Failed to load PLY sequence from: %s", dirPath.string().c_str());
        return false;
    }
    
    // Load audio if available
    if (m_plyLoader->hasAudio()) {
        m_audioPlayer = std::make_unique<AudioPlayer>();
        if (!m_audioPlayer->load(m_plyLoader->getAudioPath().string())) {
            LOGE("AnimationController: Failed to load audio from: %s", m_plyLoader->getAudioPath().string().c_str());
            return false;
        }
        m_hasAudio = true;
    } else {
        m_hasAudio = false;
    }
    
    m_totalFrames = m_plyLoader->getFrameCount();
    m_totalDurationMs = m_plyLoader->getDurationMs();
    m_currentFrame = 0;
    m_currentPositionMs = 0;
    m_accumulatedTime = 0;
    m_frameTimeAccumulator = 0.0f;
    
    LOGI("AnimationController: Loaded sequence: %zu frames (%.2f fps), duration: %.2fs", 
           m_totalFrames, m_plyLoader->getFrameRate(), m_totalDurationMs / 1000.0f);
    
    return true;
}

void AnimationController::closeSequence() {
    if (m_plyLoader) {
        m_plyLoader->close();
        m_plyLoader.reset();
    }
    
    if (m_audioPlayer) {
        m_audioPlayer->close();
        m_audioPlayer.reset();
    }
    
    m_playbackState = PlaybackState::STOPPED;
    m_hasAudio = false;
    m_totalFrames = 0;
    m_totalDurationMs = 0;
    m_currentFrame = 0;
    m_currentPositionMs = 0;
    m_accumulatedTime = 0;
    m_frameTimeAccumulator = 0.0f;
}

void AnimationController::play() {
    if (m_playbackState == PlaybackState::PAUSED) {
        m_playbackState = PlaybackState::PLAYING;
        if (m_hasAudio) {
            m_audioPlayer->play();
        }
    } else if (m_playbackState == PlaybackState::STOPPED) {
        m_playbackState = PlaybackState::PLAYING;
        m_accumulatedTime = 0;
        m_frameTimeAccumulator = 0.0f;
        if (m_hasAudio) {
            m_audioPlayer->play();
        }
    }
}

void AnimationController::pause() {
    if (m_playbackState == PlaybackState::PLAYING) {
        m_playbackState = PlaybackState::PAUSED;
        if (m_hasAudio) {
            m_audioPlayer->pause();
        }
    }
}

void AnimationController::stop() {
    m_playbackState = PlaybackState::STOPPED;
    m_currentFrame = 0;
    m_currentPositionMs = 0;
    m_accumulatedTime = 0;
    m_frameTimeAccumulator = 0.0f;
    
    if (m_hasAudio) {
        m_audioPlayer->stop();
    }
    
    synchronizeAudio();
}

void AnimationController::seek(uint64_t timestampMs) {
    size_t targetFrame = static_cast<size_t>(timestampMs / m_plyLoader->getFrameDurationMs());
    targetFrame = std::min(targetFrame, m_totalFrames - 1);
    
    seekToFrame(targetFrame);
}

void AnimationController::setPlaybackSpeed(float speed) {
    m_playbackSpeed = speed;
}

void AnimationController::setLooping(bool enabled) {
    m_isLooping = enabled;
}

void AnimationController::setVolume(float volume) {
    m_volume = std::max(0.0f, std::min(1.0f, volume));
    if (m_hasAudio && m_audioPlayer) {
        m_audioPlayer->setVolume(m_volume);
    }
}

bool AnimationController::getCurrentFrameData(SplatSet& outFrame) {
    if (!m_plyLoader || m_totalFrames == 0) {
        return false;
    }
    
    // Use interpolated frame if enabled
    if (m_plyLoader->isInterpolationEnabled()) {
        return m_plyLoader->getInterpolatedFrame(static_cast<uint32_t>(m_currentPositionMs), outFrame);
    }
    
    return m_plyLoader->getFrame(m_currentFrame, outFrame);
}

void AnimationController::setInterpolationEnabled(bool enabled) {
    if (m_plyLoader) {
        m_plyLoader->setInterpolationEnabled(enabled);
    }
}

bool AnimationController::isInterpolationEnabled() const {
    return m_plyLoader ? m_plyLoader->isInterpolationEnabled() : false;
}

void AnimationController::update(float deltaTime) {
    if (m_playbackState != PlaybackState::PLAYING || m_totalFrames == 0) {
        return;
    }
    
    // Update timing with playback speed
    m_accumulatedTime += static_cast<uint64_t>(deltaTime * m_playbackSpeed * 1000.0);
    m_frameTimeAccumulator += deltaTime * m_playbackSpeed;
    
    // Check if we should advance to next frame
    float frameDurationMs = static_cast<float>(m_plyLoader->getFrameDurationMs());
    while (m_frameTimeAccumulator >= frameDurationMs) {
        m_frameTimeAccumulator -= frameDurationMs;
        m_currentFrame++;
        
        // Handle end of sequence
        if (m_currentFrame >= m_totalFrames) {
            if (m_isLooping) {
                m_currentFrame = 0;
                m_currentPositionMs = 0;
                m_frameTimeAccumulator = 0.0f;
            } else {
                // Stop at end
                m_playbackState = PlaybackState::STOPPED;
                m_currentFrame = m_totalFrames - 1;
                m_frameTimeAccumulator = 0.0f;
            }
            break;
        }
        
        // Load current frame data (for rendering)
        if (m_currentFrame < m_totalFrames) {
            // Frame would be loaded by the main application
        }
    }
    
    updatePosition();
    
    // Update audio position to stay in sync
    if (m_hasAudio && m_audioPlayer->isLoaded()) {
        synchronizeAudio();
    }
}

void AnimationController::seekToFrame(size_t frameIndex) {
    m_currentFrame = frameIndex;
    m_currentPositionMs = frameIndex * m_plyLoader->getFrameDurationMs();
    m_accumulatedTime = m_currentPositionMs;
    m_frameTimeAccumulator = 0.0f;
    
    synchronizeAudio();
}

void AnimationController::synchronizeAudio() {
    if (!m_hasAudio) {
        return;
    }
    
    uint64_t audioPosition = m_currentPositionMs;
    m_audioPlayer->seek(audioPosition);
}

void AnimationController::updatePosition() {
    m_currentPositionMs = m_currentFrame * m_plyLoader->getFrameDurationMs();
}

} // namespace vk_viewer