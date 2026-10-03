// Native DLSS 5 Neural Rendering (OpenDLSS-NR) for the RTX paths.
//
// This lives outside vk_viewer_render.cpp so it compiles as its own translation unit rather than being pulled in
// by the unity build. Anything not built with WITH_DLSS_NR is compiled out.

#include "vk_viewer.h"

#ifdef WITH_DLSS_NR

#include <filesystem>

#include <nvutils/file_operations.hpp>

namespace vk_viewer {

bool VkViewer::renderDlssNr(VkCommandBuffer cmd)
{
  if(!prmDlssNr.dlssNrEnabled || prmDlssNr.dlssNrModel.empty())
    return false;
  if(m_viewSize.x <= 0 || m_viewSize.y <= 0)
    return false;
  if(m_dlssNrState < 0)
    return false;

  const std::filesystem::path shaderDir = nvutils::getExecutablePath().parent_path() / "shaders_dlss_nr";

  if(m_dlssNrState == 0)
  {
    if(!m_dlssNr.init(m_app->getInstance(), m_app->getPhysicalDevice(), m_app->getDevice(),
                      m_app->getQueue(0).familyIndex, m_app->getQueue(0).queueIndex, shaderDir.string()))
    {
      LOGW("DLSS-NR disabled: %s\n", m_dlssNr.error().c_str());
      m_dlssNrState = -1;
      return false;
    }
    if(!m_dlssNr.loadModel(prmDlssNr.dlssNrModel.string(), shaderDir.string()))
    {
      LOGW("DLSS-NR disabled: %s\n", m_dlssNr.error().c_str());
      m_dlssNrState = -1;
      return false;
    }
    m_dlssNrState = 1;
  }

  const uint32_t width = uint32_t(m_viewSize.x);
  const uint32_t height = uint32_t(m_viewSize.y);
  if(!m_dlssNr.ready() || m_dlssNr.width() != width || m_dlssNr.height() != height)
  {
    if(!m_dlssNr.resize(width, height))
    {
      LOGW("DLSS-NR disabled: %s\n", m_dlssNr.error().c_str());
      m_dlssNrState = -1;
      return false;
    }
    m_dlssNrFrames = 0;  // the history was just cleared
  }

  // COLOR_MAIN is the scene the RTX pass wrote (R8G8B8A8_UNORM) and COLOR_MOTION the motion target
  // (R16G16_SFLOAT); both stay in VK_IMAGE_LAYOUT_GENERAL for the G-buffer's lifetime. NR samples COLOR_MAIN and
  // then replaces it with its own result, so the ImGui presentation and the XR copy are untouched.
  DlssNrPass::GpuImage color;
  color.image = m_gBuffers.getColorImage(COLOR_MAIN);
  color.view = m_gBuffers.getColorImageView(COLOR_MAIN);
  color.format = VK_FORMAT_R8G8B8A8_UNORM;
  color.layout = VK_IMAGE_LAYOUT_GENERAL;

  DlssNrPass::GpuImage motion;
  motion.image = m_gBuffers.getColorImage(COLOR_MOTION);
  motion.view = m_gBuffers.getColorImageView(COLOR_MOTION);
  motion.format = VK_FORMAT_R16G16_SFLOAT;
  motion.layout = VK_IMAGE_LAYOUT_GENERAL;

  DlssNrPass::Frame frame;
  frame.enabled = true;
  frame.historyValid = prmDlssNr.dlssNrTemporal && m_dlssNrFrames > 0;
  frame.intensity = prmDlssNr.dlssNrIntensity;
  frame.style = prmDlssNr.dlssNrStyle;
  frame.seed = m_dlssNrFrames;
  // The ray generator stores screen-space (pixel) motion for the desktop path, with the jitter already added
  // (vk_viewer_rtx.cpp sets useNdcMotion = 0 outside XR; threedgrt_raytrace.rgen.slang:658-671), so undo both here.
  frame.motionScale[0] = -1.0f / float(width);
  frame.motionScale[1] = -1.0f / float(height);
  frame.motionBias[0] = prmFrame.dlssJitter.x / float(width);
  frame.motionBias[1] = prmFrame.dlssJitter.y / float(height);

  if(!m_dlssNr.record(cmd, frame, color, motion))
    return false;
  ++m_dlssNrFrames;
  return true;
}

}  // namespace vk_viewer

#endif  // WITH_DLSS_NR
