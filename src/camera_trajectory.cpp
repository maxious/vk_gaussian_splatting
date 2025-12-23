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

#include "camera_trajectory.h"
#include <cmath>
#include <algorithm>

namespace vk_gaussian_splatting {

std::vector<Camera> CameraTrajectory::generate(const Camera& startCamera, const TrajectoryParams& params)
{
  switch(params.type)
  {
    case TrajectoryType::ORBIT:
      return generateOrbit(startCamera, params.numFrames, params.orbitRadius, params.numOrbits, params.lookAtCenter);

    case TrajectoryType::SWIPE:
      return generateSwipe(startCamera, params.numFrames, params.orbitRadius, params.pingPong, params.lookAtCenter);

    case TrajectoryType::ROTATE_FORWARD:
      return generateRotateForward(startCamera, params.numFrames, params.orbitRadius, params.zoomRange,
                                   params.numOrbits, params.lookAtCenter);

    case TrajectoryType::SHAKE:
      return generateShake(startCamera, params.numFrames, params.orbitRadius, params.lookAtCenter);

    default:
      return {startCamera};
  }
}

std::vector<Camera> CameraTrajectory::generateOrbit(const Camera& startCamera,
                                                     int           numFrames,
                                                     float         radius,
                                                     int           numOrbits,
                                                     bool          lookAtCenter)
{
  std::vector<Camera> result;
  result.reserve(numFrames);

  glm::vec3 center   = startCamera.ctr;
  glm::vec3 startEye = startCamera.eye;
  glm::vec3 up       = startCamera.up;

  glm::vec3 toCamera   = startEye - center;
  float     distance   = glm::length(toCamera);
  glm::vec3 forward    = glm::normalize(toCamera);
  glm::vec3 right      = glm::normalize(glm::cross(up, forward));
  glm::vec3 cameraUp   = glm::normalize(glm::cross(forward, right));

  for(int i = 0; i < numFrames; ++i)
  {
    float t     = static_cast<float>(i) / static_cast<float>(numFrames);
    float angle = t * 2.0f * PI * static_cast<float>(numOrbits);

    float offsetX = radius * std::sin(angle);
    float offsetY = radius * std::cos(angle) - radius;

    glm::vec3 offset = right * offsetX + cameraUp * offsetY;
    glm::vec3 newEye = startEye + offset;

    Camera cam   = startCamera;
    cam.eye      = newEye;
    if(lookAtCenter)
    {
      cam.ctr = center;
    }
    else
    {
      cam.ctr = newEye + glm::normalize(center - startEye) * distance;
    }
    result.push_back(cam);
  }

  return result;
}

std::vector<Camera> CameraTrajectory::generateSwipe(const Camera& startCamera,
                                                     int           numFrames,
                                                     float         range,
                                                     bool          pingPong,
                                                     bool          lookAtCenter)
{
  std::vector<Camera> result;
  result.reserve(numFrames);

  glm::vec3 center   = startCamera.ctr;
  glm::vec3 startEye = startCamera.eye;
  glm::vec3 up       = startCamera.up;

  glm::vec3 forward = glm::normalize(center - startEye);
  glm::vec3 right   = glm::normalize(glm::cross(forward, up));

  for(int i = 0; i < numFrames; ++i)
  {
    float t;
    if(pingPong)
    {
      float phase = static_cast<float>(i) / static_cast<float>(numFrames);
      t           = std::sin(phase * PI);
    }
    else
    {
      t = static_cast<float>(i) / static_cast<float>(numFrames - 1);
    }

    float     offsetX = range * (t - 0.5f) * 2.0f;
    glm::vec3 offset  = right * offsetX;
    glm::vec3 newEye  = startEye + offset;

    Camera cam = startCamera;
    cam.eye    = newEye;
    if(lookAtCenter)
    {
      cam.ctr = center;
    }
    else
    {
      cam.ctr = center + offset;
    }
    result.push_back(cam);
  }

  return result;
}

std::vector<Camera> CameraTrajectory::generateRotateForward(const Camera& startCamera,
                                                             int           numFrames,
                                                             float         lateralRange,
                                                             float         zoomRange,
                                                             int           numOrbits,
                                                             bool          lookAtCenter)
{
  std::vector<Camera> result;
  result.reserve(numFrames);

  glm::vec3 center   = startCamera.ctr;
  glm::vec3 startEye = startCamera.eye;
  glm::vec3 up       = startCamera.up;

  glm::vec3 forward = glm::normalize(center - startEye);
  glm::vec3 right   = glm::normalize(glm::cross(forward, up));

  for(int i = 0; i < numFrames; ++i)
  {
    float t     = static_cast<float>(i) / static_cast<float>(numFrames);
    float angle = t * 2.0f * PI * static_cast<float>(numOrbits);

    float offsetX = lateralRange * std::sin(angle);
    float offsetZ = zoomRange * (1.0f - std::cos(angle)) * 0.5f;

    glm::vec3 offset = right * offsetX + forward * offsetZ;
    glm::vec3 newEye = startEye + offset;

    Camera cam = startCamera;
    cam.eye    = newEye;
    if(lookAtCenter)
    {
      cam.ctr = center;
    }
    result.push_back(cam);
  }

  return result;
}

std::vector<Camera> CameraTrajectory::generateShake(const Camera& startCamera,
                                                     int           numFrames,
                                                     float         range,
                                                     bool          lookAtCenter)
{
  std::vector<Camera> result;
  result.reserve(numFrames);

  glm::vec3 center   = startCamera.ctr;
  glm::vec3 startEye = startCamera.eye;
  glm::vec3 up       = startCamera.up;

  glm::vec3 forward  = glm::normalize(center - startEye);
  glm::vec3 right    = glm::normalize(glm::cross(forward, up));
  glm::vec3 cameraUp = glm::normalize(glm::cross(right, forward));

  int halfFrames = numFrames / 2;

  for(int i = 0; i < numFrames; ++i)
  {
    float     t = static_cast<float>(i) / static_cast<float>(numFrames);
    float     angle = t * 2.0f * PI * 2.0f;
    glm::vec3 offset;

    if(i < halfFrames)
    {
      offset = right * range * std::sin(angle);
    }
    else
    {
      offset = cameraUp * range * std::sin(angle);
    }

    glm::vec3 newEye = startEye + offset;

    Camera cam = startCamera;
    cam.eye    = newEye;
    if(lookAtCenter)
    {
      cam.ctr = center;
    }
    result.push_back(cam);
  }

  return result;
}

std::vector<Camera> CameraTrajectory::generateFromKeyframes(const std::vector<Camera>& keyframes,
                                                             int                        totalFrames,
                                                             bool                       loop)
{
  if(keyframes.empty())
    return {};

  if(keyframes.size() == 1)
    return std::vector<Camera>(totalFrames, keyframes[0]);

  std::vector<Camera> result;
  result.reserve(totalFrames);

  size_t numSegments = loop ? keyframes.size() : keyframes.size() - 1;
  int    framesPerSegment = totalFrames / static_cast<int>(numSegments);

  for(int i = 0; i < totalFrames; ++i)
  {
    float globalT      = static_cast<float>(i) / static_cast<float>(totalFrames);
    float segmentFloat = globalT * static_cast<float>(numSegments);
    int   segmentIndex = static_cast<int>(segmentFloat);
    float localT       = segmentFloat - static_cast<float>(segmentIndex);

    segmentIndex = std::min(segmentIndex, static_cast<int>(numSegments) - 1);

    size_t fromIdx = segmentIndex % keyframes.size();
    size_t toIdx   = (segmentIndex + 1) % keyframes.size();

    result.push_back(interpolate(keyframes[fromIdx], keyframes[toIdx], localT));
  }

  return result;
}

Camera CameraTrajectory::interpolate(const Camera& a, const Camera& b, float t)
{
  t = std::clamp(t, 0.0f, 1.0f);

  float smoothT = t * t * (3.0f - 2.0f * t);

  Camera result;
  result.model = a.model;
  result.eye   = glm::mix(a.eye, b.eye, smoothT);
  result.ctr   = glm::mix(a.ctr, b.ctr, smoothT);
  result.up    = glm::normalize(glm::mix(a.up, b.up, smoothT));
  result.fov   = glm::mix(a.fov, b.fov, smoothT);
  result.clip  = glm::mix(a.clip, b.clip, smoothT);

  result.dofEnabled = a.dofEnabled;
  result.focusDist  = glm::mix(a.focusDist, b.focusDist, smoothT);
  result.aperture   = glm::mix(a.aperture, b.aperture, smoothT);

  return result;
}

}  // namespace vk_gaussian_splatting
