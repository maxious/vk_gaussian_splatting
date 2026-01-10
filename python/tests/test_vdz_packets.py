"""Tests for VDZ packet encoding/decoding."""

from __future__ import annotations

import struct
import sys
import os
import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.utils.packets import (
    pack_depth_payload,
    quantize_depth,
    HEADER_STRUCT,
    HEADER_SIZE,
    DATA_TYPE_UINT16,
)


class TestQuantizeDepth:
    """Tests for depth quantization."""

    def test_quantize_basic(self):
        """Test basic depth quantization."""
        depth = np.array([0.0, 5.0, 10.0], dtype=np.float32)
        encoded, scale, bias = quantize_depth(depth, 0.0, 10.0)

        assert scale == 10.0 / 65535.0
        assert bias == 0.0
        assert len(encoded) == 3
        assert encoded[0] == 0
        # 5.0 / scale = 32767.5, rounded to 32768
        assert abs(int(encoded[1]) - 32768) <= 1
        assert encoded[2] == 65535

    def test_quantize_with_offset(self):
        """Test quantization with non-zero minimum."""
        depth = np.array([5.0, 10.0, 15.0], dtype=np.float32)
        encoded, scale, bias = quantize_depth(depth, 5.0, 15.0)

        assert bias == 5.0
        assert encoded[0] == 0
        # 10.0 maps to approximately 32767.5
        assert abs(int(encoded[1]) - 32768) <= 1
        assert encoded[2] == 65535

    def test_quantize_clamping(self):
        """Test that values outside range are clamped."""
        depth = np.array([-5.0, 5.0, 25.0], dtype=np.float32)
        encoded, scale, bias = quantize_depth(depth, 0.0, 10.0)

        assert encoded[0] == 0
        assert encoded[1] > 0
        assert encoded[2] == 65535

    def test_quantize_nan_handling(self):
        """Test that NaN values are handled (should be post-processed)."""
        depth = np.array([0.0, np.nan, 10.0], dtype=np.float32)
        encoded, scale, bias = quantize_depth(depth, 0.0, 10.0)

        assert len(encoded) == 3


class TestPackDepthPayload:
    """Tests for depth payload packing."""

    def test_pack_uncompressed(self):
        """Test packing without compression."""
        depth = np.array([[0.0, 5.0], [10.0, 15.0]], dtype=np.float32)
        payload = pack_depth_payload(depth, 1000.0, 0.0, 15.0, compress=False)

        assert payload.width == 2
        assert payload.height == 2
        assert payload.scale == 15.0 / 65535.0
        assert payload.bias == 0.0
        assert payload.z_max == 15.0
        assert payload.buffer[:HEADER_SIZE]
        # Data follows header - uint16 = 2 bytes per value * 4 values = 8 bytes
        assert len(payload.buffer) == HEADER_SIZE + 2 * 2 * 2

    def test_pack_compressed(self):
        """Test packing with compression."""
        depth = np.array([[0.0, 5.0], [10.0, 15.0]], dtype=np.float32)
        payload = pack_depth_payload(depth, 1000.0, 0.0, 15.0, compress=True)

        assert payload.width == 2
        assert payload.height == 2
        assert payload.buffer[:4] == b"VDZ2"

    def test_pack_header_format(self):
        """Test that header matches expected format."""
        depth = np.array([[1.0, 2.0]], dtype=np.float32)
        payload = pack_depth_payload(depth, 1234.5, 0.5, 10.0, compress=False)

        header_bytes = payload.buffer[:HEADER_SIZE]
        magic, version, data_type, timestamp, width, height, scale, bias, z_max = (
            HEADER_STRUCT.unpack(header_bytes)
        )

        assert magic == b"VDZ1"
        assert version == 1
        assert data_type == DATA_TYPE_UINT16
        assert timestamp == 1234
        assert width == 2
        assert height == 1
        assert scale == pytest.approx((10.0 - 0.5) / 65535.0)
        assert bias == 0.5
        assert z_max == 10.0

    def test_roundtrip(self):
        """Test that depth can be encoded and decoded correctly."""
        original = np.random.rand(100, 100).astype(np.float32) * 10.0
        z_min = 0.0
        z_max = 10.0

        payload = pack_depth_payload(original, 1000.0, z_min, z_max, compress=False)

        header_bytes = payload.buffer[:HEADER_SIZE]
        _, _, _, timestamp, width, height, scale, bias, z_max_parsed = HEADER_STRUCT.unpack(
            header_bytes
        )

        encoded = np.frombuffer(payload.buffer[HEADER_SIZE:], dtype=np.uint16)
        decoded = (encoded * scale + bias).astype(np.float32).reshape(height, width)

        np.testing.assert_allclose(decoded, original, rtol=0.01, atol=0.1)


class TestHeaderStruct:
    """Tests for header structure constants."""

    def test_header_size(self):
        """Verify header size is 32 bytes."""
        assert HEADER_SIZE == 32

    def test_header_format_string(self):
        """Verify header format string."""
        expected = "<4sHHIIIfff"
        assert HEADER_STRUCT.format == expected
