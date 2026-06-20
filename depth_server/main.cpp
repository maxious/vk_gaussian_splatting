#include "worker_monitor.h"
#include "worker_pool.h"

#include <nvutils/logger.hpp>

#include <atomic>
#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstring>
#include <string>
#include <thread>

namespace {

std::atomic<bool> g_shutdownRequested{false};

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
    LOGI("  --help               Show this help and exit\n");
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

    for(int i = 1; i < argc; ++i)
    {
        if(std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0)
        {
            printHelp();
            return 0;
        }
        if(std::strcmp(argv[i], "--model") == 0 && i + 1 < argc)
        {
            modelPath = argv[++i];
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
    }

    if(modelPath.empty())
    {
        LOGI("TCP depth server skeleton. Use --help for options.\n");
        return 0;
    }

    WorkerPool pool;
    if(!pool.initialize(modelPath, numWorkers, backend))
    {
        LOGE("Failed to initialize worker pool\n");
        return 1;
    }

    WorkerMonitor monitor(pool);
    if(!monitor.start(modelPath, backend))
    {
        LOGE("Failed to start worker monitor\n");
        pool.shutdown();
        return 1;
    }

    while(!g_shutdownRequested.load())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    LOGI("Shutting down...\n");
    monitor.stop();
    pool.shutdown();
    return 0;
}
