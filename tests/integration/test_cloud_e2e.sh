#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# Binary directory: from ctest (generator expression) or default
BIN="${1:-${ROOT}/_bin/Debug}"

TEST_PORT=9102
TMP_SPLAT="/tmp/cloud_e2e_test.splat"
TMP_PNG="/tmp/cloud_e2e_render.png"
FRAME_DIR=""
SERVER_PID=""

# --- cleanup ----------------------------------------------------------------
cleanup() {
    local ec=$?
    if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    pkill -f "depth_server.*${TEST_PORT}" 2>/dev/null || true
    rm -f "$TMP_SPLAT" "$TMP_PNG"
    if [ -n "$FRAME_DIR" ] && [ -d "$FRAME_DIR" ]; then
        rm -rf "$FRAME_DIR"
    fi
    exit $ec
}
trap cleanup EXIT INT TERM

# --- 1. Resolve test frames ------------------------------------------------
echo "=== Cloud E2E Test ==="
echo "[1/7] Resolving test frames..."

try_extract_video() {
    local video_path="$1"
    local label="$2"
    if [ ! -f "$video_path" ]; then
        return 1
    fi
    # Check if ffmpeg actually works (it might have missing shared libs)
    if ! ffmpeg -version &>/dev/null; then
        echo "  ffmpeg is broken, skipping $label"
        return 1
    fi
    echo "  Using $label..."
    FRAME_DIR=$(mktemp -d)
    if ! ffmpeg -y -loglevel error -i "$video_path" \
        -vf "fps=2,scale=640:-2" -frames:v 2 "${FRAME_DIR}/frame_%05d.jpg" 2>/dev/null; then
        echo "  ffmpeg extraction failed for $label, skipping"
        rm -rf "$FRAME_DIR" 2>/dev/null || true
        FRAME_DIR=""
        return 1
    fi
    TEST_FRAMES=$(ls "${FRAME_DIR}"/frame_*.jpg 2>/dev/null | sort | tr '\n' ',' | sed 's/,$//')
    if [ -z "$TEST_FRAMES" ]; then
        rm -rf "$FRAME_DIR" 2>/dev/null || true
        FRAME_DIR=""
        return 1
    fi
    return 0
}

TEST_FRAMES=""
if ! try_extract_video "${ROOT}/_downloaded_resources/lake_processed/rgb_sequence.mp4" "lake video"; then
    if ! try_extract_video "${ROOT}/_downloaded_resources/rocket/rocket.mp4" "rocket video"; then
        if [ -d "${ROOT}/tests/fixtures/cloud" ]; then
            echo "  Using pre-extracted frames from tests/fixtures/cloud/"
            TEST_FRAMES=$(ls "${ROOT}"/tests/fixtures/cloud/frame_*.jpg 2>/dev/null | sort | head -2 | tr '\n' ',' | sed 's/,$//')
        fi
    fi
fi

if [ -z "$TEST_FRAMES" ]; then
    echo "SKIP: No test frames available (no video or fixtures)"
    exit 0
fi
echo "  Got $(echo "$TEST_FRAMES" | tr ',' '\n' | wc -l) frames"

# --- 2. Resolve model -------------------------------------------------------
echo "[2/7] Resolving model..."
MODEL=""
MODEL_CANDIDATES=(
    "${HOME}/.cache/depth_server/mudler/depth-anything.cpp-gguf/depth-anything-base-q8_0.gguf"
    "${HOME}/.cache/depth_server/mudler/depth-anything.cpp-gguf/depth-anything-base-f16.gguf"
    "${ROOT}/models/depth-anything-base-q8_0.gguf"
    "${HOME}/.cache/depth_server/mudler/depth-anything.cpp-gguf/depth-anything-large-f32.gguf"
)
for candidate in "${MODEL_CANDIDATES[@]}"; do
    if [ -f "$candidate" ]; then
        MODEL="$candidate"
        break
    fi
done

if [ -z "$MODEL" ]; then
    echo "SKIP: No depth-anything model found (download one with model_downloader first)"
    exit 0
fi
echo "  Model: $(basename "$MODEL")"

# --- Verify binaries exist --------------------------------------------------
if [ ! -f "${BIN}/depth_server" ]; then
    echo "SKIP: depth_server not found at ${BIN}/depth_server"
    exit 0
fi
if [ ! -f "${BIN}/test_cloud_client" ]; then
    echo "SKIP: test_cloud_client not found at ${BIN}/test_cloud_client"
    exit 0
fi

# --- 3. Start depth_server --------------------------------------------------
echo "[3/7] Starting depth_server on port ${TEST_PORT}..."
"${BIN}/depth_server" --mode cloud --port "$TEST_PORT" \
    --cloud-model "$MODEL" --cloud-workers 1 &
SERVER_PID=$!

# Wait for port to accept connections
START_TS=$(date +%s)
while ! nc -z 127.0.0.1 "$TEST_PORT" 2>/dev/null; do
    if [ $(( $(date +%s) - START_TS )) -gt 30 ]; then
        echo "FAIL: depth_server did not start within 30s"
        if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "Server still running - killing"
            kill "$SERVER_PID" 2>/dev/null || true
        fi
        exit 1
    fi
    sleep 0.5
done
echo "  Server ready (pid=${SERVER_PID})"

# --- 4. Run test_cloud_client -----------------------------------------------
echo "[4/7] Running test_cloud_client (120s timeout)..."
if ! "${BIN}/test_cloud_client" --port "$TEST_PORT" --frames "$TEST_FRAMES" \
    --output "$TMP_SPLAT" --chunk-size 2 --overlap 1 --no-fuse --no-icp --no-loop-close --timeout-ms 120000; then
    echo "FAIL: test_cloud_client returned non-zero"
    exit 1
fi

# --- 5. Validate .splat -----------------------------------------------------
echo "[5/7] Validating .splat output..."
if [ ! -f "$TMP_SPLAT" ]; then
    echo "FAIL: No .splat file produced at ${TMP_SPLAT}"
    exit 1
fi

SPLAT_SIZE=$(stat -c%s "$TMP_SPLAT" 2>/dev/null || stat -f%z "$TMP_SPLAT" 2>/dev/null || echo 0)
if [ "$SPLAT_SIZE" -lt 1000 ]; then
    echo "FAIL: .splat file too small (${SPLAT_SIZE} bytes, expected >= 1000)"
    exit 1
fi
echo "  Size: ${SPLAT_SIZE} bytes OK"

# Validate gzip structure (.splat is gzip-compressed)
if command -v gunzip &>/dev/null; then
    if gunzip -t "$TMP_SPLAT" 2>/dev/null; then
        echo "  Gzip structure: OK"
    else
        echo "  Gzip structure: skipped (raw binary format)"
    fi
fi

# --- 6. Validate point count ------------------------------------------------
echo "[6/7] Validating point count..."
# Each .splat record is 32 bytes (kCloudSplatBytesPerRecord)
N_POINTS=$((SPLAT_SIZE / 32))
if [ "$N_POINTS" -lt 1000 ]; then
    echo "FAIL: Too few points (~${N_POINTS}, expected >= 1000)"
    exit 1
fi
echo "  Points: ~${N_POINTS} (OK >= 1000)"

# --- 7. Screenshot verification (optional) ----------------------------------
echo "[7/7] Visual verification..."
VIEWER="${BIN}/vk_viewer"
if [ -f "$VIEWER" ]; then
    if [ -f /opt/vulkan/1.4.357.0/setup-env.sh ]; then
        source /opt/vulkan/1.4.357.0/setup-env.sh 2>/dev/null || true
    fi
    if "${VIEWER}" --inputFile "$TMP_SPLAT" \
        --screenshotDelay 3.0 --screenshot "$TMP_PNG" \
        --size 800 600 --validation 0 2>/dev/null; then
        if [ -f "$TMP_PNG" ]; then
            echo "  Screenshot: ${TMP_PNG} OK"
        fi
    else
        echo "  Screenshot: skipped (no GPU or viewer failed)"
    fi
else
    echo "  Screenshot: skipped (vk_viewer not built)"
fi

echo ""
echo "=== Test PASSED ==="
exit 0
