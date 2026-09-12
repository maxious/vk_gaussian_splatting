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

#include "splat_set.h"
#include <cmath>

namespace vk_viewer {

/**
 * @brief SLERP interpolation for quaternions with hemisphere consistency
 * 
 * Performs spherical linear interpolation between two quaternions.
 * Handles the double-cover issue by checking dot product sign.
 * Falls back to normalized lerp for nearly-parallel quaternions.
 * 
 * @param q0 Start quaternion (w, x, y, z)
 * @param q1 End quaternion (w, x, y, z)
 * @param t Interpolation factor [0, 1]
 * @param out Output quaternion (w, x, y, z)
 */
inline void slerpQuaternion(const float* q0, const float* q1, float t, float* out)
{
    // Quaternion layout: [w, x, y, z] (indices 0, 1, 2, 3)
    float dot = q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3];
    
    // Handle hemisphere consistency - flip q1 if on opposite hemisphere
    float q1f[4] = {q1[0], q1[1], q1[2], q1[3]};
    if (dot < 0.0f) {
        dot = -dot;
        q1f[0] = -q1f[0];
        q1f[1] = -q1f[1];
        q1f[2] = -q1f[2];
        q1f[3] = -q1f[3];
    }
    
    // For nearly-parallel quaternions, use normalized linear interpolation
    if (dot > 0.9995f) {
        out[0] = q0[0] + t * (q1f[0] - q0[0]);
        out[1] = q0[1] + t * (q1f[1] - q0[1]);
        out[2] = q0[2] + t * (q1f[2] - q0[2]);
        out[3] = q0[3] + t * (q1f[3] - q0[3]);
        
        // Normalize
        float len = std::sqrt(out[0]*out[0] + out[1]*out[1] + out[2]*out[2] + out[3]*out[3]);
        if (len > 1e-8f) {
            out[0] /= len;
            out[1] /= len;
            out[2] /= len;
            out[3] /= len;
        }
        return;
    }
    
    // Standard SLERP
    float theta0 = std::acos(dot);
    float theta = theta0 * t;
    float sinTheta = std::sin(theta);
    float sinTheta0 = std::sin(theta0);
    
    float s0 = std::cos(theta) - dot * sinTheta / sinTheta0;
    float s1 = sinTheta / sinTheta0;
    
    out[0] = s0 * q0[0] + s1 * q1f[0];
    out[1] = s0 * q0[1] + s1 * q1f[1];
    out[2] = s0 * q0[2] + s1 * q1f[2];
    out[3] = s0 * q0[3] + s1 * q1f[3];
}

/**
 * @brief Log-space interpolation for scale values
 * 
 * Computes: exp((1-t) * log(a) + t * log(b))
 * More stable than linear lerp for values spanning multiple orders of magnitude.
 * 
 * @param a Start value
 * @param b End value
 * @param t Interpolation factor [0, 1]
 * @return Interpolated value
 */
inline float logLerp(float a, float b, float t)
{
    constexpr float MIN_SCALE = 1e-8f;
    a = std::max(a, MIN_SCALE);
    b = std::max(b, MIN_SCALE);
    return std::exp((1.0f - t) * std::log(a) + t * std::log(b));
}

/**
 * @brief Linear interpolation
 */
inline float lerp(float a, float b, float t)
{
    return a + t * (b - a);
}

/**
 * @brief Interpolate between two SplatSets
 * 
 * Performs per-splat interpolation:
 * - Positions: linear lerp
 * - Rotations: SLERP with hemisphere consistency
 * - Scales: log-space lerp
 * - Opacities: linear lerp
 * - SH coefficients (f_dc, f_rest): linear lerp
 * 
 * Both input SplatSets must have the same number of splats.
 * The output is resized/initialized as needed.
 * 
 * @param out Output SplatSet (will be resized)
 * @param frame0 Start keyframe
 * @param frame1 End keyframe
 * @param t Interpolation factor [0, 1]
 * @return true if successful, false if splat counts don't match
 */
inline bool interpolateSplatSet(SplatSet& out, const SplatSet& frame0, const SplatSet& frame1, float t)
{
    const size_t count = frame0.size();
    if (count != frame1.size() || count == 0) {
        return false;
    }
    
    // Clamp t to valid range
    t = std::max(0.0f, std::min(1.0f, t));
    
    // Edge cases - just copy
    if (t <= 0.0f) {
        out = frame0;
        return true;
    }
    if (t >= 1.0f) {
        out = frame1;
        return true;
    }
    
    // Resize output arrays
    out.positions.resize(count * 3);
    out.rotation.resize(count * 4);
    out.scale.resize(count * 3);
    out.opacity.resize(count);
    out.f_dc.resize(count * 3);
    
    // Handle f_rest - use the larger SH degree
    const size_t shPerSplat0 = frame0.f_rest.size() / count;
    const size_t shPerSplat1 = frame1.f_rest.size() / count;
    const size_t shPerSplat = std::max(shPerSplat0, shPerSplat1);
    out.f_rest.resize(count * shPerSplat);
    
    // Interpolate all splats
    for (size_t i = 0; i < count; ++i) {
        // Positions - linear lerp
        out.positions[i*3 + 0] = lerp(frame0.positions[i*3 + 0], frame1.positions[i*3 + 0], t);
        out.positions[i*3 + 1] = lerp(frame0.positions[i*3 + 1], frame1.positions[i*3 + 1], t);
        out.positions[i*3 + 2] = lerp(frame0.positions[i*3 + 2], frame1.positions[i*3 + 2], t);
        
        // Rotations - SLERP
        slerpQuaternion(&frame0.rotation[i*4], &frame1.rotation[i*4], t, &out.rotation[i*4]);
        
        // Scales - log-space lerp
        out.scale[i*3 + 0] = logLerp(frame0.scale[i*3 + 0], frame1.scale[i*3 + 0], t);
        out.scale[i*3 + 1] = logLerp(frame0.scale[i*3 + 1], frame1.scale[i*3 + 1], t);
        out.scale[i*3 + 2] = logLerp(frame0.scale[i*3 + 2], frame1.scale[i*3 + 2], t);
        
        // Opacity - linear lerp
        out.opacity[i] = lerp(frame0.opacity[i], frame1.opacity[i], t);
        
        // f_dc (base color SH) - linear lerp
        out.f_dc[i*3 + 0] = lerp(frame0.f_dc[i*3 + 0], frame1.f_dc[i*3 + 0], t);
        out.f_dc[i*3 + 1] = lerp(frame0.f_dc[i*3 + 1], frame1.f_dc[i*3 + 1], t);
        out.f_dc[i*3 + 2] = lerp(frame0.f_dc[i*3 + 2], frame1.f_dc[i*3 + 2], t);
        
        // f_rest (higher-order SH) - linear lerp with padding
        for (size_t j = 0; j < shPerSplat; ++j) {
            float v0 = (j < shPerSplat0) ? frame0.f_rest[i * shPerSplat0 + j] : 0.0f;
            float v1 = (j < shPerSplat1) ? frame1.f_rest[i * shPerSplat1 + j] : 0.0f;
            out.f_rest[i * shPerSplat + j] = lerp(v0, v1, t);
        }
    }
    
    // Copy temporal data if present (don't interpolate - use frame0's motion vectors)
    out.has_time_data = frame0.has_time_data || frame1.has_time_data;
    if (out.has_time_data) {
        out.motion = frame0.motion;
        out.time = frame0.time;
        out.time_scale = frame0.time_scale;
        if (frame0.has_gate) {
            out.has_gate = true;
            out.gate = frame0.gate;
        }
        out.minTime = lerp(frame0.minTime, frame1.minTime, t);
        out.maxTime = lerp(frame0.maxTime, frame1.maxTime, t);
    }
    
    return true;
}

} // namespace vk_viewer
