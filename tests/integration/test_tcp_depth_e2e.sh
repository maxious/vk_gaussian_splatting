#!/bin/bash
set -e

# Test model URL (DA3-BASE q4_k, ~99MB)
MODEL_URL="https://huggingface.co/mudler/depth-anything.cpp-gguf/resolve/main/depth-anything-base-q4_k.gguf"
MODEL_PATH="/tmp/test_da_base_q4_k.gguf"
TEST_VIDEO="/tmp/test_tcp_video.mp4"
SCREENSHOT="/tmp/tcp_e2e_test.png"
BINARY_PATH="$1"

if [ -z "$BINARY_PATH" ]; then
    BINARY_PATH="./_bin/Release"
fi

# Download model if not cached
if [ ! -f "$MODEL_PATH" ]; then
    echo "Downloading test model..."
    curl -L -o "$MODEL_PATH" "$MODEL_URL"
fi

# Generate test video
python3 -c "
import numpy as np, cv2
frames = []
for i in range(30):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:,:,0] = (i * 8) % 256  # R channel varies
    frame[:,:,1] = 128
    frame[:,:,2] = 255 - (i * 8) % 256
    frames.append(frame)
out = cv2.VideoWriter('$TEST_VIDEO', cv2.VideoWriter_fourcc(*'mp4v'), 10, (640, 480))
for f in frames: out.write(f)
out.release()
"

# Start server
$BINARY_PATH/depth_server --model "$MODEL_PATH" --workers 1 --port 19000 &
SERVER_PID=$!

# Wait for server ready
sleep 5

# Run viewer
$BINARY_PATH/vk_viewer --inputFile "$TEST_VIDEO" \
    --tcp-depth-servers 127.0.0.1:19000 \
    --screenshotDelay 5.0 --screenshot "$SCREENSHOT" \
    --size 800 600 --validation 0 || true

# Verify screenshot
file "$SCREENSHOT" | grep -q "PNG image data" || {
    echo "FAIL: Screenshot not valid PNG"
    kill $SERVER_PID 2>/dev/null
    exit 1
}

echo "PASS: Integration test successful"
kill $SERVER_PID 2>/dev/null
exit 0
