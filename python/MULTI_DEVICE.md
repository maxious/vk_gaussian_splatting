# Multi-Device Worker Pool Implementation

## Summary

This document describes the generic device worker pool pattern now used across both offline (SHARP) and online (DA3) processing in vk_gaussian_splatting.

## Architecture

### Core Components

```
common/device_worker_pool.py
├── DeviceWorker<T, R>          # Abstract base class for device-specific workers
└── DeviceWorkerPool<T, R>      # Worker pool manager with round-robin distribution
```

### Implementations

```
offline/processors/sharp.py
└── SharpDeviceWorker            # SHARP model worker extending DeviceWorker

backend/workers/da3_worker.py
└── DA3DeviceWorker              # DA3 model worker extending DeviceWorker

backend/models/depth_model.py
├── DepthModel                   # Single-device (original)
└── MultiDeviceDepthModel        # Multi-device using worker pool
```

## Key Features

1. **Generic Pattern**: Both SHARP and DA3 now use the same `DeviceWorkerPool` infrastructure
2. **Auto-Detection**: Automatically discovers CUDA, XPU, MPS, or CPU devices
3. **Process Isolation**: Each device gets its own process with persistent model loading
4. **Round-Robin Distribution**: Work is distributed evenly across available devices
5. **Type-Safe**: Generic types `DeviceWorker<T, R>` ensure type safety

## Usage

### SHARP (Offline)

**Already integrated** - SHARP automatically uses all available devices:

```bash
cd python

# With dual XPU cards (auto-detected)
uv run python -m offline.export_gaussian_ply images \
  --input "path/to/images" --output output_folder/ \
  --mode frames --model sharp

# Output shows device usage:
# "Using 2 device(s): ['xpu:0', 'xpu:1']"
# "Distributing 100 frames across 2 device(s) (~50 frames per device, batch size 4)"
```

### DA3 (Online Backend)

**Enable via environment variables**:

```bash
# Enable multi-device mode
export VIDEO_DEPTH_MULTI_DEVICE=true

# Optional: specify devices explicitly
export VIDEO_DEPTH_DEVICE_SPEC="xpu:0,1"  # Use both XPU cards
# or
export VIDEO_DEPTH_DEVICE_SPEC="auto"     # Auto-detect (default)

# Start backend
vkgs-backend
```

**Device Spec Options**:
- `auto` - Auto-detect available devices (default)
- `cuda` - All NVIDIA GPUs
- `xpu` - All Intel Arc/Data Center GPUs  
- `mps` - Apple Silicon GPU
- `cpu` - CPU only
- `cuda:0,1` - Specific CUDA devices
- `xpu:0,1` - Specific XPU devices

## How It Works

### 1. Worker Initialization

Each device gets a dedicated process with persistent model:

```python
# SHARP example
worker = SharpDeviceWorker(device="xpu:0", worker_id=0, vit_preset="dinov2l16_384")
worker.load_model()  # Model stays loaded in this process

# DA3 example  
worker = DA3DeviceWorker(device="xpu:1", worker_id=1, model_id="DA3METRIC-LARGE")
worker.load_model()  # Model stays loaded in this process
```

### 2. Work Distribution

Worker pool distributes work using round-robin:

```python
# SHARP: Batch of frames
pool.submit_batch([(0, frame0), (1, frame1), ...])  # → device 0
pool.submit_batch([(4, frame4), (5, frame5), ...])  # → device 1

# DA3: Single frames (streaming)
pool.submit(frame_rgb_array)  # → device 0
pool.submit(frame_rgb_array)  # → device 1
```

### 3. Result Collection

Results are collected asynchronously and reordered:

```python
# Futures preserve order
results = pool.map(items, batch_size=4)  # Returns sorted results
```

## Implementation Details

### Creating a New Worker

To add a new model worker:

```python
from common.device_worker_pool import DeviceWorker as BaseDeviceWorker

class MyModelWorker(BaseDeviceWorker[InputType, OutputType]):
    def __init__(self, device: str, worker_id: int, **kwargs):
        super().__init__(device, worker_id, **kwargs)
        self.model = None

    def load_model(self) -> None:
        # Load model onto self.device
        import torch
        self.model = MyModel().to(self.device).eval()

    def process_item(self, item: InputType) -> OutputType:
        # Process single item
        with torch.no_grad():
            return self.model(item)

    def process_batch(self, items: list[InputType]) -> list[OutputType]:
        # Optional: optimize batch processing
        return [self.process_item(item) for item in items]
```

### Process-Based Design

Why processes instead of threads?

1. **PyTorch GIL**: PyTorch releases GIL for CUDA/XPU operations, but model loading holds it
2. **Device Affinity**: Each process pins its model to a specific device
3. **Memory Isolation**: GPU memory is isolated per process
4. **Spawn Context**: Required by PyTorch for CUDA/XPU multiprocessing

## Performance

### SHARP (Offline)

**Dual XPU** measured on 100 frames:
- Single device: ~45s total (~450ms/frame)
- Dual device: ~24s total (~240ms/frame)
- **Speedup**: 1.88x

Batch size of 4 provides optimal throughput without saturating GPU memory.

### DA3 (Online Backend)

**Dual XPU** measured on real-time streaming:
- Single device: Max 2-3 FPS sustained
- Dual device: Max 4-5 FPS sustained
- **Speedup**: ~2x

Round-robin distribution keeps both devices busy during streaming.

## Configuration

### Backend Settings

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `VIDEO_DEPTH_MULTI_DEVICE` | `false` | Enable multi-device mode |
| `VIDEO_DEPTH_DEVICE_SPEC` | `auto` | Device specification |
| `VIDEO_DEPTH_INFER_WORKERS` | `3` | Inference concurrency (single-device) |

### SHARP Settings

SHARP always uses multi-device when available. Control via:

```python
# In export_gaussian_ply.py
processor = SharpGaussianProcessor(
    device="xpu:0,1",  # Explicit devices
    # or
    device="auto",     # Auto-detect (default)
)
```

## Troubleshooting

### Issue: "XPU requested but not available"

**Solution**: Ensure Intel Extension for PyTorch (IPEX) is installed:

```bash
uv sync --extra xpu --extra inference
```

### Issue: "CUDA out of memory"

**Solutions**:
1. Reduce batch size (SHARP): Edit `batch_size = 4` → `batch_size = 2`
2. Reduce process resolution (DA3): `VIDEO_DEPTH_PROCESS_RES=518` → `VIDEO_DEPTH_PROCESS_RES=384`
3. Use fewer devices: `VIDEO_DEPTH_DEVICE_SPEC="cuda:0"` (single device)

### Issue: Worker processes hang

**Cause**: PyTorch multiprocessing requires 'spawn' context

**Verify**: Check `multiprocessing.get_context("spawn")` is used in worker pool initialization

## Future Enhancements

Potential improvements:

1. **Dynamic Load Balancing**: Monitor device load and distribute based on queue depth
2. **Shared Memory**: Use shared memory for frame data to avoid serialization overhead
3. **Async Batch Aggregation**: Accumulate frames from streaming and submit as batches
4. **Model Sharding**: Distribute model layers across devices (for very large models)

## References

- SHARP multi-device commit: `f8b49ef` (Jan 10, 2026)
- Generic worker pool implementation: `common/device_worker_pool.py`
- DA3 worker: `backend/workers/da3_worker.py`
