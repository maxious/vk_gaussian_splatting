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

#include "animation_ui.h"
#include <imgui/imgui.h>
#include "nvgui/nvgui.h"

namespace vk_gaussian_splatting {

AnimationUI::AnimationUI() = default;
~AnimationUI() = default;

void AnimationUI::initialize(std::shared_ptr<AnimationController> animationController) {
    m_animationController = animationController;
}

void AnimationUI::setAnimationController(std::shared_ptr<AnimationController> animationController) {
    m_animationController = animationController;
}

void AnimationUI::renderAnimationControls(bool showControls) {
    if (!m_animationController || !showControls) {
        return false;
    }
    
    bool isPlaying = m_animationController->getPlaybackState() == PlaybackState::PLAYING;
    bool hasSequence = m_animationController->getTotalFrames() > 0;
    
    if (ImGui::Begin("Animation Controls", nullptr, showControls ? 0 : ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoFocusOnAppearing)) {
        
        // Playback controls
        ImGui::Text("Sequence: %zu frames (%.2fs)", 
                  m_animationController->getTotalFrames(), 
                  m_animationController->getTotalDurationMs() / 1000.0f);
        
        ImGui::Separator();
        
        if (hasSequence) {
            renderPlaybackControls();
            ImGui::SameLine();
            renderTimelineControls();
            ImGui::Separator();
            renderProgressDisplay();
        } else {
            ImGui::Text("No PLY sequence loaded");
        }
        
        ImGui::End();
    }
    
    return isPlaying;
}

void AnimationUI::renderPlaybackControls() {
    PlaybackState state = m_animationController->getPlaybackState();
    
    // Play/Pause/Stop buttons
    if (ImGui::Button("Play")) {
        m_animationController->play();
    }
    ImGui::SameLine();
    
    if (state == PlaybackState::PLAYING) {
        if (ImGui::Button("Pause")) {
            m_animationController->pause();
        }
    } else {
        if (ImGui::Button("Stop")) {
            m_animationController->stop();
        }
    }
    
    // Loop control
    ImGui::SameLine();
    bool isLooping = m_animationController->isLooping();
    if (ImGui::Checkbox("Loop", &isLooping)) {
        m_animationController->setLooping(isLooping);
    }
    
    // Speed control
    ImGui::SameLine();
    float speed = m_animationController->getPlaybackSpeed();
    if (ImGui::SliderFloat("Speed", &speed, 0.1f, 5.0f, "%.1fx")) {
        m_animationController->setPlaybackSpeed(speed);
    }
    
    // Seek control
    ImGui::SameLine();
    uint64_t currentPos = m_animationController->getCurrentPositionMs();
    uint64_t totalDuration = m_animationController->getTotalDurationMs();
    
    if (ImGui::SliderScalar("Position", ImGuiDataType_S64, &currentPos, nullptr, 0, 
                         static_cast<double>(totalDuration), "%.1fms")) {
        m_animationController->seek(currentPos);
    }
    
    ImGui::SameLine();
    if (ImGui::Button("Reset")) {
        m_animationController->stop();
    }
}

void AnimationUI::renderTimelineControls() {
    if (!m_animationController) {
        return;
    }
    
    size_t currentFrame = m_animationController->getCurrentFrame();
    size_t totalFrames = m_animationController->getTotalFrames();
    uint64_t currentPos = m_animationController->getCurrentPositionMs();
    uint64_t totalDuration = m_animationController->getTotalDurationMs();
    
    // Timeline slider
    if (ImGui::SliderScalar("Timeline", ImGuiDataType_S64, &currentPos, nullptr, 0,
                         static_cast<double>(totalDuration), "%.1fms")) {
        m_animationController->seek(currentPos);
    }
    
    // Frame info
    ImGui::SameLine();
    ImGui::Text("Frame: %zu / %zu", currentFrame, totalFrames);
    
    // Time display
    ImGui::SameLine();
    uint32_t seconds = static_cast<uint32_t>(currentPos / 1000);
    uint32_t minutes = seconds / 60;
    uint32_t hours = minutes / 60;
    
    ImGui::Text("Time: %02d:%02d:%02d.%03d", hours, minutes % 60, seconds % 60);
}

void AnimationUI::renderAudioControls() {
    if (!m_animationController) {
        return;
    }
    
    bool hasAudio = m_animationController->getAudioPlayer() && 
                   m_animationController->getAudioPlayer()->isLoaded();
    
    if (!hasAudio) {
        ImGui::Text("No audio loaded");
        return;
    }
    
    AudioPlayer* audioPlayer = m_animationController->getAudioPlayer();
    
    // Volume control
    ImGui::Text("Audio:");
    ImGui::SameLine();
    
    float volume = audioPlayer->getVolume();
    if (ImGui::SliderFloat("Volume", &volume, 0.0f, 1.0f, "%.2f")) {
        audioPlayer->setVolume(volume);
    }
    
    // Audio position info
    ImGui::SameLine();
    uint64_t audioPos = audioPlayer->getPositionMs();
    uint64_t audioDuration = audioPlayer->getDurationMs();
    
    ImGui::Text("Audio: %.1f / %.1fs", 
              static_cast<double>(audioPos) / audioDuration,
              static_cast<double>(audioDuration) / 1000.0);
}

void AnimationUI::renderProgressDisplay() {
    if (!m_animationController) {
        return;
    }
    
    size_t currentFrame = m_animationController->getCurrentFrame();
    size_t totalFrames = m_animationController->getTotalFrames();
    
    float progress = static_cast<float>(currentFrame) / static_cast<float>(totalFrames);
    
    ImGui::ProgressBar(progress, ImVec2(200, 0), nullptr);
    ImGui::SameLine();
    ImGui::Text("%.1f%%", progress * 100.0f);
}

bool AnimationUI::getCurrentFrameData(SplatSet& outFrame) {
    if (!m_animationController) {
        return false;
    }
    
    // Get current frame from PLY loader
    return m_animationController->getPlyLoader()->getFrame(m_animationController->getCurrentFrame(), outFrame);
}

} // namespace vk_gaussian_splatting