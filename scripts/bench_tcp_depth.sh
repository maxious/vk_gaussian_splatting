#!/bin/bash
set -e

MODEL_URL="https://huggingface.co/mudler/depth-anything.cpp-gguf/resolve/main/depth-anything-base-q8_0.gguf"
MODEL_PATH="/tmp/bench_da_base_q8_0.gguf"
FRAMES_DIR="/tmp/bench_frames"
CPU_JSON="/tmp/bench_cpu.json"
CUDA_JSON="/tmp/bench_cuda.json"
REPORT="./bench_report.md"
BINARY="${DEPTH_SERVER_BINARY:-./_bin/Release/depth_server}"
FRAME_COUNT="${BENCH_FRAME_COUNT:-100}"
BENCH_REPEAT="${BENCH_REPEAT:-3}"
BENCH_WARMUP="${BENCH_WARMUP:-3}"
FRAME_WIDTH="${BENCH_FRAME_WIDTH:-640}"
FRAME_HEIGHT="${BENCH_FRAME_HEIGHT:-480}"

if [ ! -x "$BINARY" ] && [ -x "./_bin/Debug/depth_server" ]; then
    BINARY="./_bin/Debug/depth_server"
fi

download_model() {
    if [ ! -f "$MODEL_PATH" ]; then
        echo "Downloading model..."
        curl -L --fail --retry 3 --retry-delay 2 -o "$MODEL_PATH" "$MODEL_URL"
    fi
}

generate_frames() {
    mkdir -p "$FRAMES_DIR"
    FRAME_COUNT_ENV="$FRAME_COUNT" FRAME_WIDTH_ENV="$FRAME_WIDTH" FRAME_HEIGHT_ENV="$FRAME_HEIGHT" python3 -c "
import numpy as np
import os
from pathlib import Path

out = Path('$FRAMES_DIR')
rng = np.random.default_rng(1234)
frame_count = int(os.environ['FRAME_COUNT_ENV'])
frame_width = int(os.environ['FRAME_WIDTH_ENV'])
frame_height = int(os.environ['FRAME_HEIGHT_ENV'])
for i in range(frame_count):
    frame = rng.integers(0, 256, size=(frame_height, frame_width, 3), dtype=np.uint8)
    frame.tofile(out / f'frame_{i:04d}.rgb')
print(f'Generated {frame_count} test frames')
"
}

run_benchmark() {
    backend="$1"
    output_json="$2"

    echo "=== Running ${backend^^} benchmark ==="
    if ! "$BINARY" --benchmark --model "$MODEL_PATH" --backend "$backend" \
        --benchmark-frames "$FRAMES_DIR" \
        --benchmark-repeat "$BENCH_REPEAT" --benchmark-warmup "$BENCH_WARMUP" \
        --benchmark-output "$output_json"; then
        echo "${backend^^} benchmark failed"
        rm -f "$output_json"
    fi
}

write_report() {
    FRAME_COUNT_ENV="$FRAME_COUNT" FRAME_WIDTH_ENV="$FRAME_WIDTH" FRAME_HEIGHT_ENV="$FRAME_HEIGHT" BENCH_REPEAT_ENV="$BENCH_REPEAT" BENCH_WARMUP_ENV="$BENCH_WARMUP" python3 - "$REPORT" "$CPU_JSON" "$CUDA_JSON" <<'PY'
import json
import os
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
cpu_path = Path(sys.argv[2])
cuda_path = Path(sys.argv[3])

def load(path):
    if not path.is_file():
        return None
    with path.open() as f:
        return json.load(f)

def pick(data, *keys, default="?"):
    cur = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

def row(name, data):
    if not data:
        return f"| {name} | - | - | - | - | unavailable |"
    workers = pick(data, "num_workers")
    fps = pick(data, "throughput_fps", default=None)
    mean_ms = pick(data, "per_frame_stats", "mean_ms", default=None)
    p95_ms = pick(data, "per_frame_stats", "p95_ms", default=None)
    notes = pick(data, "notes", default="")
    fps_txt = f"{fps:.1f}" if isinstance(fps, (int, float)) else str(fps)
    mean_txt = f"{mean_ms:.1f}" if isinstance(mean_ms, (int, float)) else str(mean_ms)
    p95_txt = f"{p95_ms:.1f}" if isinstance(p95_ms, (int, float)) else str(p95_ms)
    return f"| {name} | {workers} | {fps_txt} | {mean_txt} | {p95_txt} | {notes} |"

cpu = load(cpu_path)
cuda = load(cuda_path)

lines = [
    "# TCP Depth Server Benchmark Report",
    "",
    "## Configuration",
    "- Model: DA3-BASE q8_0 (~142MB)",
    f"- Resolution: {os.environ['FRAME_WIDTH_ENV']}x{os.environ['FRAME_HEIGHT_ENV']}",
    f"- Frames: {os.environ['FRAME_COUNT_ENV']} (repeat={os.environ['BENCH_REPEAT_ENV']}, warmup={os.environ['BENCH_WARMUP_ENV']})",
    "",
    "## Results",
    "| Backend | Workers | FPS | Mean(ms) | P95(ms) | Notes |",
    "|---------|---------|-----|----------|---------|-------|",
    row("CPU", cpu),
    row("CUDA", cuda),
]

if cpu and cuda:
    cpu_fps = pick(cpu, "throughput_fps", default=None)
    cuda_fps = pick(cuda, "throughput_fps", default=None)
    if isinstance(cpu_fps, (int, float)) and isinstance(cuda_fps, (int, float)) and cuda_fps:
        lines += ["", f"- CUDA speedup vs CPU: {cuda_fps / cpu_fps:.2f}x"]

report_path.write_text("\n".join(lines) + "\n")
print(f"Benchmark report generated: {report_path}")
PY
    cat "$REPORT"
}

download_model
generate_frames

if [ ! -x "$BINARY" ]; then
    echo "Benchmark binary not found: $BINARY"
    rm -f "$CPU_JSON" "$CUDA_JSON"
    write_report
    exit 0
fi

run_benchmark cpu "$CPU_JSON"

if command -v nvidia-smi >/dev/null 2>&1; then
    run_benchmark cuda "$CUDA_JSON"
else
    echo "CUDA benchmark skipped (nvidia-smi not available)"
    rm -f "$CUDA_JSON"
fi

write_report
