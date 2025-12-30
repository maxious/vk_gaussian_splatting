#pragma once
#include "gaussian_splatting_ui.h"

namespace vk_gaussian_splatting {

// Helper to draw property editor for depth stream
void drawDepthStreamProperties(GaussianSplattingUI* ui);

// Helper to draw property editor for video export
void drawVideoExportProperties(GaussianSplattingUI* ui);

// Helper to draw property editor for rendering settings
void drawRenderingProperties(GaussianSplattingUI* ui);

// Helper to draw property editor for scene statistics
void drawSceneStatistics(GaussianSplattingUI* ui);

} // namespace vk_gaussian_splatting
