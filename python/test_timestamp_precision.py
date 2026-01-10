#!/usr/bin/env python3
"""Test script to verify torchcodec timestamp precision.

This script tests that:
1. VideoDataset captures precise timestamps from FrameBatch.recording_times_sec
2. Both iterate_batches() and iterate_batches_streaming() provide correct timestamps
3. Timestamps are monotonically increasing for sequential access
"""

import sys
from pathlib import Path

# Add python/ to path
sys.path.insert(0, str(Path(__file__).parent))

from common.datasets import VideoDataset


def test_timestamp_precision(video_path: Path, batch_size: int = 8, max_frames: int = 100) -> None:
    print("=" * 60)
    print(f"TESTING TIMESTAMP PRECISION: {video_path.name}")
    print("=" * 60)

    dataset = VideoDataset(video_path, frame_indices=list(range(max_frames)))
    total_frames = dataset._metadata.num_frames
    print(f"Video: {total_frames} frames at {dataset._metadata.average_fps:.2f} FPS")
    print(f"Testing first {max_frames} frames")

    # Test iterate_batches() - captures timestamps from FrameBatch
    print("\n--- Testing iterate_batches() ---")
    all_timestamps_batch: list[float] = []
    for frames, timestamps_ms, indices in dataset.iterate_batches(batch_size):
        all_timestamps_batch.extend(timestamps_ms)
        print(f"  Batch: {len(frames)} frames, timestamps: {timestamps_ms[:3]}... ms")

    print(f"  Total timestamps collected: {len(all_timestamps_batch)}")
    print(f"  First timestamp: {all_timestamps_batch[0]:.3f} ms")
    print(f"  Last timestamp: {all_timestamps_batch[-1]:.3f} ms")

    # Check monotonicity
    is_monotonic_batch = all(
        all_timestamps_batch[i] <= all_timestamps_batch[i + 1]
        for i in range(len(all_timestamps_batch) - 1)
    )
    print(f"  Monotonically increasing: {is_monotonic_batch}")

    # Test iterate_batches_streaming() - forward-only streaming
    print("\n--- Testing iterate_batches_streaming() ---")
    dataset2 = VideoDataset(video_path, frame_indices=list(range(max_frames)))
    all_timestamps_stream: list[float] = []

    for frames, timestamps_ms, indices in dataset2.iterate_batches_streaming(batch_size):
        all_timestamps_stream.extend(timestamps_ms)
        print(f"  Batch: {len(frames)} frames, timestamps: {timestamps_ms[:3]}... ms")

    print(f"  Total timestamps collected: {len(all_timestamps_stream)}")
    print(f"  First timestamp: {all_timestamps_stream[0]:.3f} ms")
    print(f"  Last timestamp: {all_timestamps_stream[-1]:.3f} ms")

    # Check monotonicity
    is_monotonic_stream = all(
        all_timestamps_stream[i] <= all_timestamps_stream[i + 1]
        for i in range(len(all_timestamps_stream) - 1)
    )
    print(f"  Monotonically increasing: {is_monotonic_stream}")

    # Compare timestamps from both methods
    print("\n--- Comparing Methods ---")
    if len(all_timestamps_batch) == len(all_timestamps_stream):
        max_diff = max(abs(a - b) for a, b in zip(all_timestamps_batch, all_timestamps_stream))
        print(f"  Max difference between methods: {max_diff:.6f} ms")

        if max_diff < 0.001:  # Less than 1 microsecond difference
            print("  ✓ Both methods produce identical timestamps!")
        else:
            print("  ✗ Timestamps differ between methods!")
    else:
        print(
            f"  ✗ Different number of timestamps: batch={len(all_timestamps_batch)}, stream={len(all_timestamps_stream)}"
        )

    # Verify timestamps match expected FPS-based timing
    print("\n--- Verifying Timestamp Precision ---")
    expected_fps = dataset._metadata.average_fps
    if expected_fps is None:
        print("  Cannot verify: FPS not available")
        return
    expected_interval_ms = 1000.0 / expected_fps

    # Check first few intervals
    print(f"  Expected interval: {expected_interval_ms:.3f} ms ({expected_fps:.2f} FPS)")
    for i in range(min(5, len(all_timestamps_stream) - 1)):
        actual_interval = all_timestamps_stream[i + 1] - all_timestamps_stream[i]
        deviation = abs(actual_interval - expected_interval_ms)
        status = "✓" if deviation < 1.0 else "?"
        print(
            f"    Frame {i}->{i + 1}: {actual_interval:.3f} ms (deviation: {deviation:.3f} ms) {status}"
        )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total frames: {len(dataset)}")
    print(f"iterate_batches() monotonic: {is_monotonic_batch}")
    print(f"iterate_batches_streaming() monotonic: {is_monotonic_stream}")

    if is_monotonic_batch and is_monotonic_stream:
        print("\n✓ All timestamp tests passed!")
    else:
        print("\n✗ Timestamp monotonicity test failed!")

    dataset.close()
    dataset2.close()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Test torchcodec timestamp precision")
    parser.add_argument("video", type=Path, help="Path to video file")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for testing")
    parser.add_argument("--max-frames", type=int, default=100, help="Max frames to test")

    args = parser.parse_args()

    if not args.video.exists():
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    test_timestamp_precision(args.video, args.batch_size, args.max_frames)


if __name__ == "__main__":
    main()
