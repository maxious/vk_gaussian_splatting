# HDR Display Support

This document describes the HDR (High Dynamic Range) display support implementation for the Vulkan Gaussian Splatting application.

## Overview

The application now automatically detects HDR-capable displays and outputs HDR content directly to the screen when available. This is achieved through Vulkan's display color space extension mechanisms.

## Implementation Details

### 1. Swapchain HDR Detection (`nvpro_core2/nvvk/swapchain.cpp/.hpp`)

#### Changes Made:
- **Added color space tracking**: Store the selected color space in `m_colorSpace` member variable
- **HDR format prioritization**: The `selectSwapSurfaceFormat()` method now:
  1. First attempts to select HDR-capable formats in this order:
     - `VK_FORMAT_R16G16B16A16_SFLOAT` with `VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT`
     - `VK_FORMAT_A2B10G10R10_UNORM_PACK32` with `VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT`
     - `VK_FORMAT_R16G16B16A16_SFLOAT` with `VK_COLOR_SPACE_BT2020_LINEAR_EXT`
     - `VK_FORMAT_A2B10G10R10_UNORM_PACK32` with `VK_COLOR_SPACE_BT2020_LINEAR_EXT`
  2. Falls back to SDR formats if no HDR format is available:
     - `VK_FORMAT_B8G8R8A8_UNORM` with `VK_COLOR_SPACE_SRGB_NONLINEAR_KHR`
     - `VK_FORMAT_R8G8B8A8_UNORM` with `VK_COLOR_SPACE_SRGB_NONLINEAR_KHR`

#### New Public Methods:
```cpp
VkColorSpaceKHR getColorSpace() const;      // Returns the swapchain color space
bool isHDRAvailable() const;                // Returns true if HDR color space is active
```

### 2. HDR Utility Class (`src/hdr_support.h`)

A new utility class `HDRSupport` provides helpers for HDR handling:

```cpp
class HDRSupport {
public:
  // Check if color space is HDR
  static bool isHDRColorSpace(VkColorSpaceKHR colorSpace);
  
  // Get human-readable color space name
  static const char* getColorSpaceName(VkColorSpaceKHR colorSpace);
  
  // Log HDR status at startup
  static void logHDRStatus(VkColorSpaceKHR colorSpace, VkFormat format);
};
```

### 3. Application Integration (`nvpro_core2/nvapp/application.hpp`)

Added public getter to expose swapchain:
```cpp
inline const nvvk::Swapchain& getSwapchain() const { return m_swapchain; }
```

### 4. Startup Logging (`src/gaussian_splatting.cpp`)

In the `onAttach()` method, HDR status is logged at startup:
```
Display Configuration: HDR ENABLED
  Color Space: Extended sRGB Linear (HDR)
  Format: X
```

Or for SDR:
```
Display Configuration: SDR
  Color Space: sRGB (SDR)
  Format: X
```

## Supported HDR Color Spaces

The implementation supports the following HDR color spaces:
- `VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT`: Extended sRGB with linear gamma
- `VK_COLOR_SPACE_DCI_P3_LINEAR_EXT`: DCI-P3 color space with linear gamma
- `VK_COLOR_SPACE_BT2020_LINEAR_EXT`: BT.2020 color space with linear gamma
- `VK_COLOR_SPACE_EXTENDED_SRGB_NONLINEAR_EXT`: Extended sRGB with non-linear gamma

## Usage

No changes are required to use HDR support. The application automatically:

1. **Detects** HDR-capable displays during swapchain initialization
2. **Selects** an appropriate HDR format if available
3. **Logs** the display configuration at startup
4. **Outputs** directly to HDR displays when available

### Accessing HDR Status in Code

You can check HDR status anywhere in the rendering pipeline:

```cpp
// In any code with access to the application:
const auto& swapchain = app->getSwapchain();
if (swapchain.isHDRAvailable()) {
  LOGI("HDR is enabled!");
  VkColorSpaceKHR colorSpace = swapchain.getColorSpace();
  // ... handle HDR rendering
}

// Or use the HDR utility:
if (HDRSupport::isHDRColorSpace(colorSpace)) {
  LOGI("Display supports: %s\n", HDRSupport::getColorSpaceName(colorSpace));
}
```

## Rendering Considerations

### Current State

The application renders in linear color space by default. When HDR is available:
- **Output format**: 16-bit per channel (R16G16B16A16_SFLOAT) or 10-bit packed format
- **Color space**: Linear for proper tone mapping
- **Display**: Direct to HDR framebuffer

### Future Enhancements

To fully leverage HDR displays, consider:

1. **Tone mapping**: Implement proper tone mapping shaders that preserve extended dynamic range
2. **Exposure control**: Add UI controls for exposure adjustment in HDR mode
3. **Metadata**: Support SMPTE 2086 mastering display metadata for peak brightness hints
4. **HDR texture formats**: Use HDR texture formats for input data when available

### Example: Checking in Shaders

To detect HDR in rendering code, check the color format and adjust tone mapping accordingly:

```glsl
// In fragment shader
#define HDR_ENABLED (colorFormat == VK_FORMAT_R16G16B16A16_SFLOAT)

if (HDR_ENABLED) {
  // Use extended range values directly
  outColor = vec4(vec3(value), 1.0);  // Values can exceed 1.0
} else {
  // Tone map for SDR display
  outColor = vec4(toneMap(value), 1.0);
}
```

## Technical Notes

### Why Extended sRGB Linear?

The implementation prioritizes `VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT` because:
- Supports the full range of sRGB colors plus extended range
- Linear gamma is preferred for rendering calculations
- Widely supported on modern hardware
- Compatible with existing rendering pipelines

### Format Selection Priority

1. **16-bit float** (`R16G16B16A16_SFLOAT`): Maximum precision, best for tone mapping
2. **10-bit packed** (`A2B10G10R10_UNORM_PACK32`): Good precision, memory efficient
3. **8-bit SDR** (fallback): For displays without HDR support

### Performance Impact

- **Negligible**: HDR format selection is one-time operation at initialization
- **Memory**: Slightly increased if using 16-bit formats (but still less than full FP32)
- **Bandwidth**: May be reduced with 10-bit packed format

## Testing HDR Support

To verify HDR is working:

1. **Check log output** at startup for color space and format information
2. **Use HDR monitor tools** to confirm extended color range values are being output
3. **Compare rendering** on HDR vs. SDR displays to see extended dynamic range

### Linux: Verify HDR Display

```bash
# Check available color spaces
vulkaninfo | grep -i "color space"
```

### Windows: NVIDIA Driver

Ensure NVIDIA driver is up to date and HDR output is enabled in settings.

### macOS: Metal Integration

Note: Vulkan on macOS (via Metal) has limited HDR support. Check platform documentation for requirements.

## References

- [NVIDIA HDR Programming Guide](https://developer.nvidia.com/high-dynamic-range-display-development)
- [Khronos Vulkan Spec - Surface Color Spaces](https://www.khronos.org/registry/vulkan/specs/1.3/html/vkspec.html#VkColorSpaceKHR)
- [UHD Color for Games](https://developer.nvidia.com/sites/default/files/akamai/gameworks/hdr/UHDColorForGames.pdf)

## Files Modified

- `nvpro_core2/nvvk/swapchain.hpp` - Added color space tracking and HDR query methods
- `nvpro_core2/nvvk/swapchain.cpp` - Implemented HDR format selection logic
- `nvpro_core2/nvapp/application.hpp` - Exposed swapchain getter
- `src/gaussian_splatting.cpp` - Added HDR status logging at startup
- `src/hdr_support.h` - New utility class for HDR handling

## Files Created

- `src/hdr_support.h` - HDR utility class and helpers
- `doc/HDR_SUPPORT.md` - This documentation
