#!/bin/bash
set -e

# PBR Screenshot Test
# Verifies the PBR rendering pipeline (--pipeline 2 --pbrEnabled 1) renders
# correctly when rendering with an HDR envmap.
# Usage: ./tests/run_pbr_screenshot_test.sh [Debug|Release]

BUILD_CONFIG="${1:-Release}"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SCENE="${SCRIPT_DIR}/_downloaded_resources/flowers_1/flowers_1.ply"
ENVMAP="${SCRIPT_DIR}/_downloaded_resources/studio_small_07_4k.hdr"
SCREENSHOT="/tmp/pbr_test.png"
BINARY_DIR="${SCRIPT_DIR}/_bin/${BUILD_CONFIG}"

echo "=== PBR Screenshot Test ==="
echo "Build config: ${BUILD_CONFIG}"
echo "Scene: ${SCENE}"
echo "Envmap: ${ENVMAP}"
echo "Output: ${SCREENSHOT}"

# Check test scene exists
if [ ! -f "$SCENE" ]; then
  echo "ERROR: Test scene not found at ${SCENE}"
  exit 1
fi

# Check envmap exists
if [ ! -f "$ENVMAP" ]; then
  echo "ERROR: Envmap file not found at ${ENVMAP}"
  exit 1
fi

# Build if binary doesn't exist
if [ ! -f "${BINARY_DIR}/vk_viewer" ]; then
  echo "Building viewer (${BUILD_CONFIG})..."
  cmake --build "${SCRIPT_DIR}/build" --config "${BUILD_CONFIG}"
fi

# Verify binary exists
if [ ! -f "${BINARY_DIR}/vk_viewer" ]; then
  echo "ERROR: vk_viewer binary not found at ${BINARY_DIR}/vk_viewer"
  exit 1
fi

# Source Vulkan SDK
if [ -f /opt/vulkan/1.4.357.0/setup-env.sh ]; then
  source /opt/vulkan/1.4.357.0/setup-env.sh
fi

# Run the viewer with PBR pipeline and envmap
"${BINARY_DIR}/vk_viewer" --inputFile "${SCENE}" \
  --pipeline 2 \
  --pbrEnabled 1 \
  --envmap "${ENVMAP}" \
  --screenshotDelay 3.0 \
  --screenshot "${SCREENSHOT}" \
  --size 800 600 \
  --validation 0

# Verify screenshot was created
if [ ! -f "$SCREENSHOT" ]; then
  echo "ERROR: Screenshot not created"
  exit 1
fi

# Verify PNG is valid
FILE_OUTPUT=$(file "$SCREENSHOT")
echo "Screenshot: ${FILE_OUTPUT}"

if ! echo "$FILE_OUTPUT" | grep -q "PNG image data"; then
  echo "FAIL: Screenshot is not a valid PNG"
  exit 1
fi

# Verify dimensions
echo "Verifying image dimensions..."
python3 -c "
from PIL import Image
img = Image.open('${SCREENSHOT}')
if img.size != (800, 600):
    print(f'FAIL: Expected 800x600, got {img.size[0]}x{img.size[1]}')
    exit(1)
print(f'Dimensions: {img.size[0]}x{img.size[1]} - PASS')
"

# Verify not all-black (check mean pixel brightness)
echo "Verifying image brightness..."
MEAN=$(python3 -c "
from PIL import Image
import numpy as np
img = Image.open('${SCREENSHOT}').convert('RGB')
arr = np.array(img, dtype=np.float32)
mean = arr.mean()
print(f'{mean:.1f}')
")

echo "Mean brightness: ${MEAN}"

if [ "$(echo "${MEAN} > 5" | bc)" -eq 1 ]; then
  echo "PASS: Screenshot has non-black pixels (mean=${MEAN})"
else
  echo "FAIL: Screenshot appears all-black (mean=${MEAN})"
  exit 1
fi

echo ""
echo "=== Test PASSED ==="
