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

#ifndef _CAMERA_TRAJECTORY_H_
#define _CAMERA_TRAJECTORY_H_

#include <vector>
#include <string>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "camera_set.h"

namespace vk_gaussian_splatting {

enum class TrajectoryType
{
  ORBIT,           // Circular orbit around the look-at point
  SWIPE,           // Horizontal pan left-to-right
  ROTATE_FORWARD,  // Orbit with forward/backward zoom oscillation (ml-sharp default)
  SHAKE,           // Oscillating shake effect
};

struct TrajectoryParams
{
  TrajectoryType type           = TrajectoryType::ORBIT;
  int            numFrames      = 300;   // Total frames to generate
  float          orbitRadius    = 0.5f;  // Lateral movement range (meters)
  float          zoomRange      = 0.3f;  // Forward/backward range for rotate_forward (meters)
  bool           lookAtCenter   = true;  // Keep looking at the scene center
  int            numOrbits      = 1;     // Number of complete orbits/cycles
  bool           pingPong       = false; // Return to start (for swipe/shake)
};

class CameraTrajectory
{
public:
  static std::vector<Camera> generate(const Camera&           startCamera,
                                       const TrajectoryParams& params);

  static std::vector<Camera> generateOrbit(const Camera& startCamera,
                                            int           numFrames,
                                            float         radius,
                                            int           numOrbits,
                                            bool          lookAtCenter);

  static std::vector<Camera> generateSwipe(const Camera& startCamera,
                                            int           numFrames,
                                            float         range,
                                            bool          pingPong,
                                            bool          lookAtCenter);

  static std::vector<Camera> generateRotateForward(const Camera& startCamera,
                                                    int           numFrames,
                                                    float         lateralRange,
                                                    float         zoomRange,
                                                    int           numOrbits,
                                                    bool          lookAtCenter);

  static std::vector<Camera> generateShake(const Camera& startCamera,
                                            int           numFrames,
                                            float         range,
                                            bool          lookAtCenter);

  static std::vector<Camera> generateFromKeyframes(const std::vector<Camera>& keyframes,
                                                    int                        totalFrames,
                                                    bool                       loop);

  static Camera interpolate(const Camera& a, const Camera& b, float t);

private:
  static constexpr float PI = 3.14159265358979323846f;
};

}  // namespace vk_gaussian_splatting

#endif
