#pragma once
#include <string>
#include <map>
#include <vector>
#include <chrono>
#include <mutex>
#include <numeric>
#include <algorithm>

namespace vk_gaussian_splatting {

class PerfStats {
public:
    struct Metric {
        double current = 0.0;
        double min = 1e9;
        double max = -1e9;
        double avg = 0.0;
        uint64_t count = 0;
        std::vector<double> historyForPlotting;
        size_t historySize = 60;

        void update(double value) {
            current = value;
            min = std::min(min, value);
            max = std::max(max, value);
            count++;
            
            avg = avg + (value - avg) / count;
            
            historyForPlotting.push_back(value);
            if (historyForPlotting.size() > historySize) {
                historyForPlotting.erase(historyForPlotting.begin());
            }
        }
    };

    void update(const std::string& name, double value) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_metrics[name].update(value);
    }

    Metric getMetric(const std::string& name) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_metrics.find(name) != m_metrics.end()) {
            return m_metrics[name];
        }
        return Metric();
    }

    std::map<std::string, Metric> getAllMetrics() {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_metrics;
    }

private:
    std::map<std::string, Metric> m_metrics;
    std::mutex m_mutex;
};

}
