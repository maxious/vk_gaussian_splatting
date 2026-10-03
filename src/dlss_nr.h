// Native DLSS 5 Neural Rendering (OpenDLSS-NR) for the viewer.
//
// Owns the model, the kernels and the 71-block graph, a dedicated RGBA16F history pair and a dedicated RGBA8
// output, and records the whole pass (input features -> network -> temporal composite) into the viewer's own
// command buffer. The network runs in GLSL/SPIR-V on cooperative matrices: see 3rdparty/OpenDLSS-NR/VENDORED.md
// for why the PTX route is compiled out.
//
// Proxy space: the scene input is the viewer's display-referred COLOR_MAIN and the history stores the same space,
// so nothing here tone maps or sRGB encodes. See shaders/dlss_nr/.
//
// Threading: init / loadModel / resize / record / destroy are main-thread with no frame in flight. record() runs
// inside the viewer's command buffer, outside a render pass.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vulkan/vulkan.h>

class DlssNrPass {
public:
  DlssNrPass();
  ~DlssNrPass();
  DlssNrPass(const DlssNrPass&) = delete;
  DlssNrPass& operator=(const DlssNrPass&) = delete;

  // An image the viewer owns. `view` is only needed for the images the pass samples; `layout` is the viewer's
  // current tracking value, used as the barrier's old layout.
  struct GpuImage {
    VkImage     image  = VK_NULL_HANDLE;
    VkImageView view   = VK_NULL_HANDLE;
    VkFormat    format = VK_FORMAT_UNDEFINED;
    VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL;
    bool valid() const { return image != VK_NULL_HANDLE; }
  };

  // Per-frame inputs, decided on the main thread. `motionScale` / `motionBias` turn the viewer's stored motion
  // into a uv offset:  uvHistory = uv + motion * motionScale + motionBias
  //   pixel motion: (currNdc - prevNdc) * viewport * 0.5 + dlssJitter
  //                 -> motionScale = -1 / viewport, motionBias = dlssJitter / viewport
  //   NDC motion:   (currNdc - prevNdc) -> motionScale = (-0.5, -0.5), motionBias = (0, 0)
  struct Frame {
    bool     enabled      = false;
    bool     historyValid = false;
    float    motionScale[2] = {-0.5F, -0.5F};
    float    motionBias[2]  = {0.0F, 0.0F};
    float    intensity      = 1.0F;
    float    localTone      = 1.0F;
    float    localStructure = 1.0F;
    float    skinStructure  = -1.0F;
    bool     autoMask       = true;
    int32_t  style          = 0;
    uint32_t seed           = 0;
  };

  // Creates the device context, the samplers, the pipelines and the descriptors. No model yet, so this is cheap
  // and safe to call from onAttach. `shaderDir` holds dlss_nr_preprocess.spv and dlss_nr_composite.spv.
  bool init(VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device, uint32_t queueFamily,
            uint32_t queueIndex, const std::string& shaderDir);

  // Loads the model directory (manifest.json + stage files) and the 13 network kernels from `kernelDir`.
  // Expensive: seconds, dominated by the stage SHA-256 verification.
  bool loadModel(const std::string& modelDir, const std::string& kernelDir);

  // init() succeeded but loadModel() may not have been called yet.
  bool initialized() const;
  // usable: initialized, model loaded, and a size was fitted.
  bool ready() const;
  const std::string& error() const;

  // Rebuilds the graph's activations, the images and the command buffers for a new size.
  bool resize(uint32_t width, uint32_t height);
  uint32_t width() const;
  uint32_t height() const;
  // The padded field the network actually runs on.
  uint32_t fullWidth() const;
  uint32_t fullHeight() const;

  // Records the pass into the viewer's command buffer, which must be outside a render pass.
  //   color  : the scene, sampled (R8G8B8A8_UNORM, display-referred) and then overwritten with the NR result
  //   motion : the viewer's motion target (R16G16_SFLOAT)
  // Returns false when the pass is not ready, or when the inputs do not fit the fitted size.
  bool record(VkCommandBuffer cmd, const Frame& frame, const GpuImage& color, const GpuImage& motion);

  // Timings of the last recorded frame, milliseconds. Not implemented yet; always 0.
  double networkMs() const;
  const std::string& deviceName() const;

  void destroy();

private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;
};
