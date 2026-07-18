#include "worker_monitor.h"
#include "benchmark.h"
#include "cloud_worker_pool.h"
#include "worker_pool.h"
#include "splat_worker_pool.h"
#include "tcp_server.h"
#include "model_downloader.h"

#include <nvutils/logger.hpp>

#include <atomic>
#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstring>
#include <memory>
#include <string>
#include <thread>

std::atomic<bool> g_shutdownRequested{false};

namespace {

void handleSignal(int sig)
{
    if(sig == SIGTERM || sig == SIGINT)
    {
        LOGI("Received signal %d, initiating graceful shutdown...\n", sig);
        g_shutdownRequested.store(true);
    }
}

void printHelp()
{
    LOGI("Usage: depth_server [options]\n");
    LOGI("Options:\n");
    LOGI("  --mode depth|splat   Operating mode (default: depth)\n");
    LOGI("  --port N             TCP listen port (default: 9000)\n");
    LOGI("  --model PATH         GGUF model path (depth mode)\n");
    LOGI("  --workers N          Number of worker processes (default: auto)\n");
    LOGI("  --backend cpu|cuda   Inference backend (default: cpu)\n");
    LOGI("  --splat-model PATH   FreeSplatter GGUF model path (splat mode)\n");
    LOGI("  --splat-backend cpu|vulkan|cuda  Splat inference backend (default: cpu)\n");
    LOGI("  --splat-workers N    Number of splat worker processes (default: 1)\n");
    LOGI("  --cloud-model PATH   DA3 GGUF model path for cloud streaming mode\n");
    LOGI("  --cloud-backend cpu|cuda  Cloud inference backend (default: cpu)\n");
    LOGI("  --cloud-workers N    Number of cloud worker processes (default: 1)\n");
    LOGI("  --cloud-max-frames N Max frames per job 2..200 (default: 64)\n");
    LOGI("  --cloud-chunk-size N Frames per sliding window 2..24 (default: 12)\n");
    LOGI("  --cloud-overlap N    Overlap between consecutive windows (default: 3)\n");
    LOGI("  --cloud-fuse          Enable TSDF voxel surface fusion (default: off)\n");
    LOGI("  --cloud-metric        Enable absolute-metre rescale (default: off)\n");
    LOGI("  --cloud-icp           Enable per-seam ICP refinement (default: off)\n");
    LOGI("  --cloud-loop-close    Enable loop-closure pose-graph (default: off)\n");
    LOGI("  --cloud-conf-pct F    Confidence percentile 0..100 (default: 55)\n");
    LOGI("  --cloud-point-size F  Per-point radius multiplier (default: 1.2)\n");
    LOGI("  --cloud-fuse-voxel-frac F  Voxel fraction of bbox diagonal (default: 0.004)\n");
    LOGI("  --cloud-fuse-trunc-mult F  Truncation as multiple of voxel (default: 4)\n");
    LOGI("  --benchmark          Run benchmark mode (no TCP server)\n");
    LOGI("  --benchmark-frames PATH  Directory of .rgb frames for benchmark\n");
    LOGI("  --benchmark-repeat N     Repeat each frame N times (default: 10)\n");
    LOGI("  --benchmark-warmup N     Warmup frames before measurement (default: 3)\n");
    LOGI("  --benchmark-output PATH  JSON output path\n");
    LOGI("  --list-models        Print available HuggingFace models\n");
    LOGI("  --help               Show this help and exit\n");
    LOGI("\n");
    LOGI("The --model and --splat-model arguments accept either a local file\n");
    LOGI("path or a HuggingFace reference of the form 'repo_id:filename'. HF\n");
    LOGI("references are auto-downloaded to ~/.cache/depth_server/ on first use.\n");
    LOGI("Run two instances (one --mode depth, one --mode splat on a different\n");
    LOGI("port) to serve both depth and splat requests simultaneously.\n");
}

}  // namespace

int main(int argc, char** argv)
{
    LOGI("depth_server v0.1 — distributed depth processing\n");

    signal(SIGTERM, handleSignal);
    signal(SIGINT, handleSignal);
    signal(SIGPIPE, SIG_IGN);

    std::string modelPath;
    std::string backend = "cpu";
    int numWorkers = 0;
    int port = 9000;
    bool benchmarkMode = false;
    std::string benchmarkFramesDir;
    int benchmarkRepeat = 10;
    int benchmarkWarmup = 3;
    std::string benchmarkOutputPath;
    std::string mode = "depth";       // "depth", "splat", or "cloud"
    std::string splatModelPath;
    std::string splatBackend = "cpu";
    int numSplatWorkers = 1;
    std::string cloudModelPath;
    std::string cloudBackend = "cpu";
    int numCloudWorkers = 1;

    for(int i = 1; i < argc; ++i)
    {
        if(std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0)
        {
            printHelp();
            return 0;
        }
        if(std::strcmp(argv[i], "--list-models") == 0)
        {
            printAvailableModels();
            return 0;
        }
        if(std::strcmp(argv[i], "--model") == 0 && i + 1 < argc)
        {
            modelPath = argv[++i];
            continue;
        }
        if(std::strcmp(argv[i], "--port") == 0 && i + 1 < argc)
        {
            port = std::stoi(argv[++i]);
            continue;
        }
        if(std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc)
        {
            backend = argv[++i];
            continue;
        }
        if(std::strcmp(argv[i], "--workers") == 0 && i + 1 < argc)
        {
            numWorkers = std::max(0, std::stoi(argv[++i]));
            continue;
        }
        if(std::strcmp(argv[i], "--benchmark") == 0)
        {
            benchmarkMode = true;
            continue;
        }
        if(std::strcmp(argv[i], "--benchmark-frames") == 0 && i + 1 < argc)
        {
            benchmarkFramesDir = argv[++i];
            continue;
        }
        if(std::strcmp(argv[i], "--benchmark-repeat") == 0 && i + 1 < argc)
        {
            benchmarkRepeat = std::max(1, std::stoi(argv[++i]));
            continue;
        }
        if(std::strcmp(argv[i], "--benchmark-warmup") == 0 && i + 1 < argc)
        {
            benchmarkWarmup = std::max(0, std::stoi(argv[++i]));
            continue;
        }
        if(std::strcmp(argv[i], "--benchmark-output") == 0 && i + 1 < argc)
        {
            benchmarkOutputPath = argv[++i];
            continue;
        }
        if(std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc)
        {
            const char* newMode = argv[++i];
            if(!mode.empty() && mode != newMode && ((mode == "splat" && std::strcmp(newMode, "cloud") == 0) || (mode == "cloud" && std::strcmp(newMode, "splat") == 0)))
            {
                LOGE("--mode cloud and --mode splat are mutually exclusive\n");
                return 1;
            }
            mode = newMode;
            if(mode != "depth" && mode != "splat" && mode != "cloud")
            {
                LOGE("--mode must be 'depth', 'splat', or 'cloud'\n");
                return 1;
            }
            continue;
        }
        if(std::strcmp(argv[i], "--splat-model") == 0 && i + 1 < argc)
        {
            splatModelPath = argv[++i];
            continue;
        }
        if(std::strcmp(argv[i], "--splat-backend") == 0 && i + 1 < argc)
        {
            splatBackend = argv[++i];
            continue;
        }
        if(std::strcmp(argv[i], "--splat-workers") == 0 && i + 1 < argc)
        {
            numSplatWorkers = std::max(1, std::stoi(argv[++i]));
            continue;
        }
        if(std::strcmp(argv[i], "--cloud-model") == 0 && i + 1 < argc)
        {
            cloudModelPath = argv[++i];
            continue;
        }
        if(std::strcmp(argv[i], "--cloud-backend") == 0 && i + 1 < argc)
        {
            cloudBackend = argv[++i];
            continue;
        }
        if(std::strcmp(argv[i], "--cloud-workers") == 0 && i + 1 < argc)
        {
            numCloudWorkers = std::max(1, std::stoi(argv[++i]));
            continue;
        }
    }

    if(mode != "splat" && !splatModelPath.empty())
    {
        LOGE("--splat-model requires --mode splat\n");
        return 1;
    }
    if(mode != "cloud" && !cloudModelPath.empty())
    {
        LOGE("--cloud-model requires --mode cloud\n");
        return 1;
    }

    std::string resolvedModel;
    if(mode == "depth")
    {
        if(modelPath.empty())
        {
            LOGE("--model PATH is required in --mode depth\n");
            return 1;
        }
        resolvedModel = resolveModelPath(modelPath);
        if(resolvedModel.empty())
        {
            LOGE("Failed to resolve model: %s\n", modelPath.c_str());
            return 1;
        }
    }

    std::unique_ptr<SplatWorkerPool> splatPool;
    if(mode == "splat")
    {
        if(splatModelPath.empty())
        {
            LOGE("--splat-model PATH is required in --mode splat\n");
            return 1;
        }
        std::string resolvedSplat = resolveModelPath(splatModelPath);
        if(resolvedSplat.empty())
        {
            LOGE("Failed to resolve splat model: %s\n", splatModelPath.c_str());
            return 1;
        }
        splatPool = std::make_unique<SplatWorkerPool>();
        if(!splatPool->initialize(resolvedSplat, numSplatWorkers, splatBackend))
        {
            LOGE("Failed to initialize splat worker pool\n");
            return 1;
        }
        LOGI("Splat mode initialized with %d worker(s), backend=%s\n", numSplatWorkers, splatBackend.c_str());
    }

    std::unique_ptr<CloudWorkerPool> cloudPool;
    if(mode == "cloud")
    {
        if(cloudModelPath.empty())
        {
            LOGE("--cloud-model PATH is required in --mode cloud\n");
            return 1;
        }
        std::string resolvedCloud = resolveModelPath(cloudModelPath);
        if(resolvedCloud.empty())
        {
            LOGE("Failed to resolve cloud model: %s\n", cloudModelPath.c_str());
            return 1;
        }
        cloudPool = std::make_unique<CloudWorkerPool>();
        if(!cloudPool->initialize(resolvedCloud, numCloudWorkers, cloudBackend))
        {
            LOGE("Failed to initialize cloud worker pool\n");
            return 1;
        }
        LOGI("Cloud mode initialized with %d worker(s), backend=%s\n", numCloudWorkers, cloudBackend.c_str());
    }

    if(benchmarkMode)
    {
        return runBenchmark(resolvedModel.c_str(), benchmarkFramesDir.c_str(), numWorkers, backend.c_str(),
                            benchmarkRepeat, benchmarkWarmup, benchmarkOutputPath.c_str());
    }

    WorkerPool pool;
    if(mode == "depth")
    {
        if(!pool.initialize(resolvedModel, numWorkers, backend))
        {
            LOGE("Failed to initialize worker pool\n");
            return 1;
        }
    }

    WorkerMonitor monitor(pool);
    if(mode == "depth")
    {
        if(!monitor.start(resolvedModel, backend))
        {
            LOGE("Failed to start worker monitor\n");
            pool.shutdown();
            return 1;
        }
    }

    TcpServer server(pool, splatPool ? splatPool.get() : nullptr);
    if(cloudPool)
    {
        server.setCloudPool(cloudPool.get());
    }
    if(!server.start(port))
    {
        LOGE("Failed to start TCP server\n");
        monitor.stop();
        pool.shutdown();
        if(splatPool) splatPool->shutdown();
        return 1;
    }
    if(cloudPool) server.setCloudPool(cloudPool.get());

    while(!g_shutdownRequested.load())
    {
        server.update();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    LOGI("Shutting down...\n");
    server.stop();
    monitor.stop();
    pool.shutdown();
    if(splatPool) splatPool->shutdown();
    return 0;
}
