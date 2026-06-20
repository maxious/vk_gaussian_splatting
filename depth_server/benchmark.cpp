#include "benchmark.h"

#include "worker_pool.h"

#include <nvutils/logger.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace {

struct FrameData {
    std::vector<std::uint8_t> rgb;
    int width = 0;
    int height = 0;
};

std::optional<std::pair<int, int>> inferDimensions(std::size_t byte_size)
{
    if(byte_size == 0 || (byte_size % 3) != 0)
    {
        return std::nullopt;
    }

    const std::size_t pixel_count = byte_size / 3;
    const std::pair<int, int> common_sizes[] = {
        {640, 480}, {1280, 720}, {1920, 1080}, {3840, 2160}, {800, 600}, {1024, 768}, {1280, 960}, {1024, 1024}
    };
    for(const auto& [w, h] : common_sizes)
    {
        if(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) == pixel_count)
        {
            return std::pair<int, int>{w, h};
        }
    }

    std::pair<int, int> best{0, 0};
    double best_score = std::numeric_limits<double>::infinity();
    const double target_ratios[] = {16.0 / 9.0, 4.0 / 3.0, 1.0, 3.0 / 2.0, 5.0 / 4.0};

    for(std::size_t h = 1; h * h <= pixel_count; ++h)
    {
        if((pixel_count % h) != 0)
        {
            continue;
        }
        const std::size_t w = pixel_count / h;
        const double ratio = static_cast<double>(w) / static_cast<double>(h);
        double score = std::numeric_limits<double>::infinity();
        for(double target : target_ratios)
        {
            score = std::min(score, std::abs(ratio - target));
        }
        if(score < best_score)
        {
            best_score = score;
            best = {static_cast<int>(w), static_cast<int>(h)};
        }
    }

    if(best.first > 0 && best.second > 0)
    {
        return best;
    }

    return std::nullopt;
}

bool loadFrameFile(const std::filesystem::path& path, FrameData& out_frame)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if(!file)
    {
        LOGE("Benchmark: cannot open frame %s\n", path.string().c_str());
        return false;
    }

    const std::streamsize size = file.tellg();
    if(size <= 0)
    {
        LOGE("Benchmark: empty frame %s\n", path.string().c_str());
        return false;
    }

    const auto dims = inferDimensions(static_cast<std::size_t>(size));
    if(!dims)
    {
        LOGE("Benchmark: cannot infer frame dimensions from %s (%lld bytes)\n", path.string().c_str(),
             static_cast<long long>(size));
        return false;
    }

    out_frame.width = dims->first;
    out_frame.height = dims->second;
    out_frame.rgb.resize(static_cast<std::size_t>(size));
    file.seekg(0, std::ios::beg);
    if(!file.read(reinterpret_cast<char*>(out_frame.rgb.data()), size))
    {
        LOGE("Benchmark: failed reading frame %s\n", path.string().c_str());
        return false;
    }

    return true;
}

double percentile(const std::vector<double>& values, double p)
{
    if(values.empty())
    {
        return 0.0;
    }
    const double clamped = std::clamp(p, 0.0, 1.0);
    const std::size_t idx = static_cast<std::size_t>(std::ceil(clamped * static_cast<double>(values.size())));
    const std::size_t pos = std::min(idx == 0 ? std::size_t{0} : idx - 1, values.size() - 1);
    return values[pos];
}

}  // namespace

int runBenchmark(const char* model_path,
                 const char* frames_dir,
                 int num_workers,
                 const char* backend,
                 int repeat_count,
                 int warmup_frames,
                 const char* output_path)
{
    if(model_path == nullptr || model_path[0] == '\0' || frames_dir == nullptr || frames_dir[0] == '\0')
    {
        LOGE("Benchmark: --model and --benchmark-frames are required\n");
        return 1;
    }

    const std::filesystem::path frames_path(frames_dir);
    if(!std::filesystem::is_directory(frames_path))
    {
        LOGE("Benchmark: frames directory not found: %s\n", frames_dir);
        return 1;
    }

    std::vector<std::filesystem::path> frame_files;
    for(const auto& entry : std::filesystem::directory_iterator(frames_path))
    {
        if(entry.is_regular_file() && entry.path().extension() == ".rgb")
        {
            frame_files.push_back(entry.path());
        }
    }

    if(frame_files.empty())
    {
        LOGE("Benchmark: no .rgb files found in %s\n", frames_dir);
        return 1;
    }

    std::sort(frame_files.begin(), frame_files.end());

    std::vector<FrameData> frames;
    frames.reserve(frame_files.size());
    for(const auto& frame_path : frame_files)
    {
        FrameData frame;
        if(!loadFrameFile(frame_path, frame))
        {
            return 1;
        }
        frames.push_back(std::move(frame));
    }

    const int frame_width = frames.front().width;
    const int frame_height = frames.front().height;
    for(const auto& frame : frames)
    {
        if(frame.width != frame_width || frame.height != frame_height)
        {
            LOGE("Benchmark: mixed frame resolutions are not supported\n");
            return 1;
        }
    }

    LOGI("Benchmark: loaded %zu frames from %s\n", frames.size(), frames_dir);

    WorkerPool pool;
    const int worker_count = std::max(1, num_workers);
    if(!pool.initialize(model_path, worker_count, backend != nullptr ? backend : "cpu"))
    {
        LOGE("Benchmark: failed to initialize worker pool\n");
        return 1;
    }

    const int warm_count = std::min(std::max(0, warmup_frames), static_cast<int>(frames.size()));
    for(int i = 0; i < warm_count; ++i)
    {
        const FrameData& frame = frames[static_cast<std::size_t>(i) % frames.size()];
        if(pool.submitFrame(static_cast<std::uint32_t>(i), 0, frame.rgb.data(), static_cast<std::uint32_t>(frame.width),
                            static_cast<std::uint32_t>(frame.height)) < 0)
        {
            LOGE("Benchmark: warmup submit failed\n");
            pool.shutdown();
            return 1;
        }

        int frame_index = -1;
        std::vector<float> depth_data;
        int out_w = 0;
        int out_h = 0;
        float scale = 0.0f;
        float bias = 0.0f;
        float z_max = 0.0f;
        while(!pool.pollResult(frame_index, depth_data, out_w, out_h, scale, bias, z_max))
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    const int repeat_total = std::max(1, repeat_count);
    std::vector<double> timings_ms;
    timings_ms.reserve(static_cast<std::size_t>(repeat_total) * frames.size());

    for(int repeat = 0; repeat < repeat_total; ++repeat)
    {
        for(std::size_t i = 0; i < frames.size(); ++i)
        {
            const FrameData& frame = frames[i];
            const auto start = std::chrono::steady_clock::now();
            if(pool.submitFrame(static_cast<std::uint32_t>(repeat * frames.size() + i), 0, frame.rgb.data(),
                                static_cast<std::uint32_t>(frame.width), static_cast<std::uint32_t>(frame.height)) < 0)
            {
                LOGE("Benchmark: submit failed\n");
                pool.shutdown();
                return 1;
            }

            int frame_index = -1;
            std::vector<float> depth_data;
            int out_w = 0;
            int out_h = 0;
            float scale = 0.0f;
            float bias = 0.0f;
            float z_max = 0.0f;
            while(!pool.pollResult(frame_index, depth_data, out_w, out_h, scale, bias, z_max))
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            const auto end = std::chrono::steady_clock::now();
            const double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
            timings_ms.push_back(elapsed_ms);
        }
    }

    if(timings_ms.empty())
    {
        LOGE("Benchmark: no timings collected\n");
        pool.shutdown();
        return 1;
    }

    const double sum_ms = std::accumulate(timings_ms.begin(), timings_ms.end(), 0.0);
    const double mean_ms = sum_ms / static_cast<double>(timings_ms.size());
    const double total_time_sec = sum_ms / 1000.0;
    const double throughput_fps = total_time_sec > 0.0 ? static_cast<double>(timings_ms.size()) / total_time_sec : 0.0;

    std::vector<double> sorted = timings_ms;
    std::sort(sorted.begin(), sorted.end());

    double variance = 0.0;
    for(double value : timings_ms)
    {
        const double diff = value - mean_ms;
        variance += diff * diff;
    }
    const double stddev_ms = std::sqrt(variance / static_cast<double>(timings_ms.size()));

    char json[8192];
    const int json_len = std::snprintf(
        json,
        sizeof(json),
        "{\n"
        "  \"backend\": \"%s\",\n"
        "  \"model\": \"%s\",\n"
        "  \"num_workers\": %d,\n"
        "  \"frame_resolution\": {\"width\": %d, \"height\": %d},\n"
        "  \"warmup_frames\": %d,\n"
        "  \"repeat_per_frame\": %d,\n"
        "  \"total_frames_processed\": %zu,\n"
        "  \"total_time_sec\": %.3f,\n"
        "  \"throughput_fps\": %.2f,\n"
        "  \"per_frame_stats\": {\n"
        "    \"mean_ms\": %.2f,\n"
        "    \"median_ms\": %.2f,\n"
        "    \"p95_ms\": %.2f,\n"
        "    \"p99_ms\": %.2f,\n"
        "    \"min_ms\": %.2f,\n"
        "    \"max_ms\": %.2f,\n"
        "    \"stddev_ms\": %.2f\n"
        "  }\n"
        "}\n",
        backend != nullptr ? backend : "cpu",
        model_path,
        pool.activeWorkers(),
        frame_width,
        frame_height,
        warm_count,
        repeat_total,
        timings_ms.size(),
        total_time_sec,
        throughput_fps,
        mean_ms,
        percentile(sorted, 0.5),
        percentile(sorted, 0.95),
        percentile(sorted, 0.99),
        sorted.front(),
        sorted.back(),
        stddev_ms);

    if(json_len < 0 || static_cast<std::size_t>(json_len) >= sizeof(json))
    {
        LOGE("Benchmark: failed to format JSON output\n");
        pool.shutdown();
        return 1;
    }

    LOGI("Benchmark results:\n%s\n", json);

    if(output_path != nullptr && output_path[0] != '\0')
    {
        std::ofstream out(output_path, std::ios::binary);
        if(!out)
        {
            LOGE("Benchmark: cannot write output file %s\n", output_path);
            pool.shutdown();
            return 1;
        }
        out.write(json, json_len);
    }

    pool.shutdown();
    LOGI("Benchmark complete: %.1f FPS (mean %.1f ms, median %.1f ms)\n", throughput_fps, mean_ms,
         percentile(sorted, 0.5));
    return 0;
}
