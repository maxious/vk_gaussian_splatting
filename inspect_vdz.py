#!/usr/bin/env python3
import struct
import os

files = [
    "_downloaded_resources/test_frame_0.vdz",
    "_downloaded_resources/test_frame_1.vdz",
    "_downloaded_resources/test_frame_2.vdz",
    "_downloaded_resources/test_frame_3.vdz",
    "_downloaded_resources/test_frame_4.vdz"
]

for filename in files:
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        continue
    
    with open(filename, 'rb') as f:
        # Read header
        magic = f.read(4).decode('ascii')
        version = struct.unpack('<H', f.read(2))[0]
        dataType = struct.unpack('<H', f.read(2))[0]
        timestamp = struct.unpack('<I', f.read(4))[0]
        width = struct.unpack('<I', f.read(4))[0]
        height = struct.unpack('<I', f.read(4))[0]
        scale = struct.unpack('<f', f.read(4))[0]
        bias = struct.unpack('<f', f.read(4))[0]
        zMax = struct.unpack('<f', f.read(4))[0]
        
        # Get file size
        f.seek(0, 2)
        fileSize = f.tell()
        
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
        print(f"  File Size: {fileSize} bytes")
        print(f"  Header Size: 32 bytes")
        print(f"  Data Size: {fileSize - 32} bytes")
        print(f"  Expected Data (uncompressed): {width * height * 2} bytes")
        print()
