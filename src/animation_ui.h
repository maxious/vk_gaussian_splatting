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

#include <memory>
#include "animation_controller.h"

namespace vk_gaussian_splatting {

/**
 * @brief Animation playback UI controls
 * 
 * Provides ImGui controls for synchronized PLY sequence and audio playback.
 * Integrates with the existing UI to add timeline, playback controls,
 * and audio management to the viewer.
 */
class AnimationUI {
public:
    AnimationUI() = default;
    ~AnimationUI() = default;

    /**
     * @brief Initialize animation UI with animation controller
     * @param animationController The animation controller to control
     */
    void initialize(std::shared_ptr<AnimationController> animationController);

    /**
     * @brief Set the animation controller for rendering
     * @param animationController The animation controller to use for getting current frame
     */
    void setAnimationController(std::shared_ptr<AnimationController> animationController);

    /**
     * @brief Render animation controls UI
     * @param showControls Whether to show the controls
     * @return True if animation is currently playing
     */
    bool renderAnimationControls(bool showControls = true);

    /**
     * @brief Get the current frame splat data for rendering
     * @param outFrame Output frame data
     * @return True if frame data is available
     */
    bool getCurrentFrameData(SplatSet& outFrame);

private:
    std::shared_ptr<AnimationController> m_animationController;
    
    void renderPlaybackControls();
    void renderTimelineControls();
    void renderAudioControls();
    void renderProgressDisplay();
};

} // namespace vk_gaussian_splatting