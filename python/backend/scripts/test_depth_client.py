#!/usr/bin/env python3
"""Test client for the depth streaming backend.

This client:
1. Uploads a video file to create a session
2. Connects to the WebSocket stream
3. Requests depth frames at specified timestamps
4. Receives and optionally saves the depth data
"""

import asyncio
import json
import struct
import argparse
from pathlib import Path
from typing import Optional

import aiohttp
import numpy as np


# VDZ header format: magic (4 bytes) + version (2) + data_type (2) + timestamp_ms (4)
# + width (4) + height (4) + scale (4) + bias (4) + z_max (4)
# Total: 32 bytes
VDZ_HEADER_FORMAT = "<4sHHIIIfff"
VDZ_HEADER_SIZE = 32
VDZ_MAGIC = b"VDZ1"


def parse_vdz_header(data: bytes):
    """Parse VDZ frame header."""
    magic, version, data_type, timestamp_ms, width, height, scale, bias, z_max = struct.unpack(
        VDZ_HEADER_FORMAT, data[:VDZ_HEADER_SIZE]
    )
    if magic != VDZ_MAGIC:
        raise ValueError(f"Invalid VDZ magic: {magic}")
    return {
        "version": version,
        "data_type": data_type,
        "timestamp_ms": timestamp_ms,
        "width": width,
        "height": height,
        "scale": scale,
        "bias": bias,
        "z_max": z_max,
    }


class DepthClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.session_id: Optional[str] = None
        self.width: int = 0
        self.height: int = 0
        self.fps: float = 0
        self.duration_ms: float = 0

    async def create_session(self, video_path: Path) -> str:
        """Upload video and create a session."""
        print(f"Uploading {video_path}...")
        async with aiohttp.ClientSession() as session:
            with open(video_path, "rb") as f:
                data = aiohttp.FormData()
                data.add_field(
                    "file",
                    f,
                    filename=video_path.name,
                    content_type="video/mp4",
                )
                async with session.post(f"{self.base_url}/api/sessions", data=data) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"Failed to create session: {resp.status}")
                    result = await resp.json()
                    self.session_id = result["session_id"]
                    self.width = result["width"]
                    self.height = result["height"]
                    self.fps = result["fps"]
                    self.duration_ms = result["duration_ms"]
                    print(f"Session created: {self.session_id}")
                    print(
                        f"  Video: {self.width}x{self.height} @ {self.fps:.2f}fps, {self.duration_ms / 1000:.2f}s"
                    )
                    return self.session_id

    async def stream_depth_frames(
        self,
        timestamps_ms: list[float],
        save_dir: Optional[Path] = None,
        verbose: bool = True,
        use_multi_device: bool = False,  # Enable multi-device mode
    ) -> list[dict]:
        """Connect to WebSocket and request depth frames."""
        if not self.session_id:
            raise RuntimeError("No session created. Call create_session first.")

        save_dir = save_dir or Path("tmp/depth_output")
        save_dir.mkdir(parents=True, exist_ok=True)

        uri = f"{self.base_url.replace('http', 'ws')}/api/sessions/{self.session_id}/stream"
        if verbose:
            print(f"Connecting to {uri}...")

        results = []
        received_count = 0

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(uri) as ws:
                if verbose:
                    print("Connected to WebSocket")

                # Send requests for each timestamp
                for ts in timestamps_ms:
                    request = {"time_ms": ts}
                    await ws.send_json(request)
                    if verbose:
                        print(f"  Requested frame at {ts:.1f}ms")

                # Receive responses
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.BINARY:
                        header = parse_vdz_header(msg.data)
                        depth_data = msg.data[VDZ_HEADER_SIZE:]

                        # Parse depth values (uint16, encoded with scale and bias)
                        # depth = encoded * scale + bias
                        encoded = np.frombuffer(depth_data, dtype=np.uint16)
                        depth_array = (encoded * header["scale"] + header["bias"]).astype(
                            np.float32
                        )
                        depth_array = depth_array.reshape(header["height"], header["width"])

                        # Calculate z_min from bias (since encoded starts at 0)
                        z_min = header["bias"]
                        z_max = header["z_max"]

                        result = {
                            "timestamp_ms": header["timestamp_ms"],
                            "z_min": z_min,
                            "z_max": z_max,
                            "width": header["width"],
                            "height": header["height"],
                            "depth_array": depth_array,
                        }
                        results.append(result)
                        received_count += 1

                        if verbose:
                            print(
                                f"  Received frame at {header['timestamp_ms']:.1f}ms "
                                f"({header['width']}x{header['height']}, "
                                f"z range: [{z_min:.2f}, {z_max:.2f}])"
                            )

                        # Save depth array as numpy file
                        np.save(
                            save_dir / f"depth_{header['timestamp_ms']:.0f}.npy",
                            depth_array,
                        )

                        # Also save as PNG (normalized for visualization)
                        depth_normalized = ((depth_array - z_min) / (z_max - z_min) * 255).astype(
                            np.uint8
                        )
                        from PIL import Image

                        Image.fromarray(depth_normalized).save(
                            save_dir / f"depth_{header['timestamp_ms']:.0f}.png"
                        )

                        # Early exit if we received all requested frames
                        if received_count >= len(timestamps_ms):
                            break
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        raise RuntimeError(f"WebSocket error: {msg.data}")

        if verbose:
            print(f"\nReceived {received_count}/{len(timestamps_ms)} frames")
        return results

    async def request_all_frames(self, save_dir: Optional[Path] = None) -> list[dict]:
        """Request depth frames for the entire video at regular intervals."""
        if not self.session_id:
            raise RuntimeError("No session created. Call create_session first.")

        # Request frames at regular intervals (every 1 second by default)
        interval_ms = 1000  # 1 second
        timestamps = list(range(0, int(self.duration_ms), int(interval_ms)))
        return await self.stream_depth_frames(timestamps, save_dir)

    async def close(self):
        """Delete the session."""
        if self.session_id:
            async with aiohttp.ClientSession() as session:
                async with session.delete(
                    f"{self.base_url}/api/sessions/{self.session_id}"
                ) as resp:
                    if resp.status == 200:
                        print(f"Session {self.session_id} deleted")
                    else:
                        print(f"Failed to delete session: {resp.status}")


async def main():
    parser = argparse.ArgumentParser(description="Test client for depth streaming backend")
    parser.add_argument("video", type=Path, help="Path to video file")
    parser.add_argument("--url", default="http://localhost:8000", help="Backend URL")
    parser.add_argument(
        "--timestamps", type=str, help="Comma-separated timestamps in ms (e.g., '0,1000,2000')"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("tmp/depth_output"), help="Output directory"
    )
    parser.add_argument(
        "--all", action="store_true", help="Request all frames at 1-second intervals"
    )
    parser.add_argument("--keep", action="store_true", help="Keep session after test")
    args = parser.parse_args()

    client = DepthClient(args.url)

    try:
        # Create session
        await client.create_session(args.video)

        # Determine which timestamps to request
        if args.timestamps:
            timestamps = [float(ts) for ts in args.timestamps.split(",")]
        elif args.all:
            timestamps = list(range(0, int(client.duration_ms), 1000))
        else:
            # Default: request first few frames
            timestamps = [0, 1000, 2000] if client.duration_ms > 2000 else [0]

        # Request depth frames
        results = await client.stream_depth_frames(timestamps, args.output)

        print(f"\nDone! Received {len(results)} depth frames")
        print(f"Output saved to: {args.output}")

    finally:
        if not args.keep and client.session_id:
            await client.close()


if __name__ == "__main__":
    asyncio.run(main())
