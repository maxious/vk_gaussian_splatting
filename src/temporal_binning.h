#ifndef _TEMPORAL_BINNING_H_
#define _TEMPORAL_BINNING_H_

#include <vector>
#include <array>
#include <cmath>
#include <cstdint>
#include <algorithm>

namespace vk_viewer {

// Temporal binning structure for CPU-side culling (Phase 3.1)
struct TemporalBins {
    static constexpr int NUM_BINS = 128;
    
    // Per-bin: indices of splats whose t_center falls in this bin
    std::vector<std::vector<uint32_t>> binIndices;  // [NUM_BINS][variable]
    
    // Precomputed: bin_time_center, max_duration_in_bin (for overlap calc)
    std::array<float, NUM_BINS> binTimeCenter;
    std::array<float, NUM_BINS> binMaxDuration;
    
    // At runtime: which bins are active for current time t
    std::vector<uint32_t> activeBins;  // indices of bins to render

    TemporalBins() {
        binIndices.resize(NUM_BINS);
        binMaxDuration.fill(0.0f);
        // Initialize centers
        for(int i = 0; i < NUM_BINS; ++i) {
            binTimeCenter[i] = (float(i) + 0.5f) / float(NUM_BINS);
        }
    }

    void clear() {
        for(auto& bin : binIndices) bin.clear();
        binMaxDuration.fill(0.0f);
        activeBins.clear();
    }

    // Build bins from splat data
    void build(const std::vector<float>& times, const std::vector<float>& durations, float t_min = 0.0f, float t_max = 1.0f) {
        clear();
        if(times.empty()) return;

        float inv_range = 1.0f / (t_max - t_min + 1e-6f);

        for(size_t i = 0; i < times.size(); ++i) {
            float t_norm = (times[i] - t_min) * inv_range;
            t_norm = std::max(0.0f, std::min(1.0f, t_norm)); // Clamp [0, 1]
            
            int binIdx = std::min(int(t_norm * NUM_BINS), NUM_BINS - 1);
            
            binIndices[binIdx].push_back(static_cast<uint32_t>(i));
            
            // Track max duration in this bin for conservative culling
            // Duration is time_scale in splat data
            if(i < durations.size()) {
                binMaxDuration[binIdx] = std::max(binMaxDuration[binIdx], durations[i]);
            }
        }
    }

    // Find active bins for a given time
    const std::vector<uint32_t>& updateActiveBins(float current_time, float temporal_threshold = 0.01f) {
        activeBins.clear();
        
        // Check each bin
        for(int i = 0; i < NUM_BINS; ++i) {
            if(binIndices[i].empty()) continue;

            float t_center = binTimeCenter[i];
            float max_dur = binMaxDuration[i]; // time_scale (sigma)
            
            // Conservative check: if current time is within 3 sigma of bin center
            // (assuming splats in bin are clustered near center)
            // Gaussian 3-sigma rule: 99.7% of energy is within +/- 3*sigma
            // We use the MAX sigma in the bin to be safe
            // Also account for bin width itself (1/NUM_BINS)
            float bin_half_width = 0.5f / float(NUM_BINS);
            
            // Distance from bin center to current time
            float dt = std::abs(current_time - t_center);
            
            // Conservative threshold: if within 3 sigma + bin width
            if (dt < (3.0f * max_dur + bin_half_width)) {
                activeBins.push_back(i);
            }
        }
        return activeBins;
    }
    
    // Get total active splat count (for buffer resizing/stats)
    size_t getActiveSplatCount() const {
        size_t count = 0;
        for(uint32_t binIdx : activeBins) {
            count += binIndices[binIdx].size();
        }
        return count;
    }
};

} // namespace vk_viewer

#endif // _TEMPORAL_BINNING_H_