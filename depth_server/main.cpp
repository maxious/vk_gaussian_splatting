#include "worker_monitor.h"
#include "benchmark.h"
#include "worker_pool.h"
#include "tcp_server.h"
#include "model_downloader.h"

#include <nvutils/logger.hpp>

#include <atomic>
#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstring>
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
    LOGI("  --port N             TCP listen port (default: 9000)\n");
    LOGI("  --model PATH         GGUF model path\n");
    LOGI("  --workers N          Number of worker processes (default: auto)\n");
    LOGI("  --backend cpu|cuda   Inference backend (default: cpu)\n");
    LOGI("  --benchmark          Run benchmark mode (no TCP server)\n");
    LOGI("  --benchmark-frames PATH  Directory of .rgb frames for benchmark\n");
    LOGI("  --benchmark-repeat N     Repeat each frame N times (default: 10)\n");
    LOGI("  --benchmark-warmup N     Warmup frames before measurement (default: 3)\n");
    LOGI("  --benchmark-output PATH  JSON output path\n");
    LOGI("  --list-models        Print available HuggingFace models\n");
    LOGI("  --help               Show this help and exit\n");
    LOGI("\n");
    LOGI("The --model argument accepts either a local file path or a\n");
    LOGI("HuggingFace reference of the form 'repo_id:filename'. HF references\n");
    LOGI("are auto-downloaded to ~/.cache/depth_server/ on first use.\n");
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
    }

    if(modelPath.empty())
    {
        LOGE("--model PATH is required\n");
        return 1;
    }

    std::string resolvedModel = resolveModelPath(modelPath);
    if(resolvedModel.empty())
    {
        LOGE("Failed to resolve model: %s\n", modelPath.c_str());
        return 1;
    }

    if(benchmarkMode)
    {
        return runBenchmark(resolvedModel.c_str(), benchmarkFramesDir.c_str(), numWorkers, backend.c_str(),
                            benchmarkRepeat, benchmarkWarmup, benchmarkOutputPath.c_str());
    }

    WorkerPool pool;
    if(!pool.initialize(resolvedModel, numWorkers, backend))
    {
        LOGE("Failed to initialize worker pool\n");
        return 1;
    }

    WorkerMonitor monitor(pool);
    if(!monitor.start(resolvedModel, backend))
    {
        LOGE("Failed to start worker monitor\n");
        pool.shutdown();
        return 1;
    }

    TcpServer server(pool);
    if(!server.start(port))
    {
        LOGE("Failed to start TCP server\n");
        monitor.stop();
        pool.shutdown();
        return 1;
    }

    while(!g_shutdownRequested.load())
    {
        server.update();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    LOGI("Shutting down...\n");
    server.stop();
    monitor.stop();
    pool.shutdown();
    return 0;
}
