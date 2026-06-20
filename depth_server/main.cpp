#include <nvutils/logger.hpp>

#include <cstring>

void printHelp() {
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

int main(int argc, char** argv) {
    LOGI("depth_server v0.1 — distributed depth processing\n");

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            printHelp();
            return 0;
        }
    }

    LOGI("TCP depth server skeleton. Use --help for options.\n");
    return 0;
}
