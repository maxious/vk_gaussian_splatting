#pragma once

// Run benchmark mode. Returns 0 on success, 1 on failure.
// Loads frames from dir, processes them through WorkerPool, measures timing.
int runBenchmark(const char* model_path,
                 const char* frames_dir,
                 int num_workers,
                 const char* backend,
                 int repeat_count,
                 int warmup_frames,
                 const char* output_path);
