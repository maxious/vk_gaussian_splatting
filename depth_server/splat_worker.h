#pragma once

#include <string>

// SplatWorker - runs in a forked subprocess. Never returns.
//
// The worker reads jobs from a socketpair fd (length-prefixed binary IPC),
// runs FreeSplatterBuffer::run() on the supplied image bytes, writes the
// resulting .splat file to the cache directory, and sends a result message
// back to the parent process. Progress updates are also sent while a job is
// in flight.
class SplatWorker {
public:
    // Run the worker loop. Reads jobs from parent_fd (length-prefixed binary IPC),
    // calls FreeSplatterBuffer::run(), writes .splat to cache dir, sends result back.
    // Never returns under normal operation.
    // parent_fd: socketpair fd from the parent
    // model_path: path to .gguf file
    // backend: "cpu" | "vulkan" | "cuda"
    // splat_cache_dir: directory to write .splat output files
    static void run(int parent_fd, const std::string& model_path,
                    const std::string& backend, const std::string& splat_cache_dir);
};