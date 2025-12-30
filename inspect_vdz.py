#!/usr/bin/env python3
import struct
import os

files = [
    "_downloaded_resources/test_frame_0.vdz",
    "_downloaded_resources/test_frame_1.vdz",
    "_downloaded_resources/test_frame_2.vdz",
    "_downloaded_resources/test_frame_3.vdz",
    "_downloaded_resources/test_frame_4.vdz",
]

for filename in files:
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        continue

    with open(filename, "rb") as f:
        # Read header
        magic = f.read(4).decode("ascii")
        version = struct.unpack("<H", f.read(2))[0]
        dataType = struct.unpack("<H", f.read(2))[0]
        timestamp = struct.unpack("<I", f.read(4))[0]
        width = struct.unpack("<I", f.read(4))[0]
        height = struct.unpack("<I", f.read(4))[0]
        scale = struct.unpack("<f", f.read(4))[0]
        bias = struct.unpack("<f", f.read(4))[0]
        zMax = struct.unpack("<f", f.read(4))[0]

        # Get file size
        f.seek(0, 2)
        fileSize = f.tell()
        f.seek(32, 0)  # Back to data start

        print(f"{filename}:")
        print(f"  Magic: {magic}")
        print(f"  Version: {version}")
        print(f"  DataType: {dataType}")
        print(f"  Timestamp: {timestamp} ms")
        print(f"  Width: {width}")
        print(f"  Height: {height}")
        print(f"  Scale: {scale:.6f}")
        print(f"  Bias: {bias:.6f}")
        print(f"  ZMax: {zMax:.6f}")

        # Read a few samples to check for variation
        if dataType == 1 and fileSize == 32 + width * height * 2:  # Uncompressed uint16
            data = f.read(width * height * 2)

            # Scan all for min/max
            import array

            arr = array.array("H")
            arr.frombytes(data)

            min_val = 65535
            max_val = 0
            for val in arr:
                if val < min_val:
                    min_val = val
                if val > max_val:
                    max_val = val

            print(f"  Data Range:")
            print(f"    Raw Min: {min_val} -> {min_val * scale + bias:.4f}m")
            print(f"    Raw Max: {max_val} -> {max_val * scale + bias:.4f}m")

            # Sample center, and 4 corners
            indices = [
                0,  # Top-Left
                width - 1,  # Top-Right
                (height - 1) * width,  # Bottom-Left
                width * height - 1,  # Bottom-Right
                (height // 2) * width + (width // 2),  # Center
            ]
            print("  Specific Samples:")
            for idx in indices:
                if idx < len(arr):
                    val = arr[idx]
                    meters = val * scale + bias
                    print(f"    Index {idx}: {val} -> {meters:.4f}m")

        print(f"  File Size: {fileSize} bytes")
        print()
