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
#include "nvgui/fonts.hpp"
#include "nvgui/tooltip.hpp"
#include <cinttypes>

namespace vk_gaussian_splatting {

void AnimationUI::initialize(std::shared_ptr<AnimationController> animationController) {
    m_animationController = animationController;
}

void AnimationUI::setAnimationController(std::shared_ptr<AnimationController> animationController) {
    m_animationController = animationController;
}

void AnimationUI::renderAnimationControls(bool showControls) {
    if (!m_animationController || !showControls) {
        return;
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
            ImGui::Separator();
            renderTimelineControls();
            ImGui::Separator();
            renderProgressDisplay();
        } else {
            ImGui::Text("No PLY sequence loaded");
        }

        ImGui::End();
    }
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

    // Reset button
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

    // Frame slider
    int frameSlider = static_cast<int>(currentFrame);
    int maxFrame = static_cast<int>(totalFrames > 0 ? totalFrames - 1 : 0);
    if (ImGui::SliderInt("Frame", &frameSlider, 0, maxFrame)) {
        if (frameSlider >= 0 && frameSlider < static_cast<int>(totalFrames)) {
            m_animationController->seekToFrame(static_cast<size_t>(frameSlider));
        }
    }

    ImGui::SameLine();
    ImGui::Text("Frame: %zu / %zu", currentFrame, totalFrames);

    // Time display
    uint64_t currentPos = m_animationController->getCurrentPositionMs();
    uint64_t seconds = currentPos / 1000;
    uint64_t minutes = seconds / 60;
    uint64_t hours = minutes / 60;

    ImGui::SameLine();
    ImGui::Text("Time: %02" PRIu64 ":%02" PRIu64 ":%02" PRIu64 ".%03" PRIu64,
                hours % 24, minutes % 60, seconds % 60, currentPos % 1000);
}

void AnimationUI::renderAudioControls() {
    if (!m_animationController) {
        return;
    }

    bool hasAudio = m_animationController->hasAudio();

    if (!hasAudio) {
        ImGui::Text("No audio loaded");
        return;
    }

    // Volume control
    float volume = m_animationController->getVolume();
    if (ImGui::SliderFloat("Volume", &volume, 0.0f, 1.0f, "%.2f")) {
        m_animationController->setVolume(volume);
    }
}

void AnimationUI::renderProgressDisplay() {
    if (!m_animationController) {
        return;
    }

    size_t currentFrame = m_animationController->getCurrentFrame();
    size_t totalFrames = m_animationController->getTotalFrames();

    float progress = totalFrames > 0 ? static_cast<float>(currentFrame) / static_cast<float>(totalFrames) : 0.0f;

    ImGui::ProgressBar(progress, ImVec2(200, 0), nullptr);
    ImGui::SameLine();
    ImGui::Text("%.1f%%", progress * 100.0f);
}

bool AnimationUI::getCurrentFrameData(SplatSet& outFrame) {
    if (!m_animationController) {
        return false;
    }

    return m_animationController->getCurrentFrameData(outFrame);
}

}  // namespace vk_gaussian_splatting
