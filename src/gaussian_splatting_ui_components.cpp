/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gaussian_splatting_ui.h"
#include <imgui/imgui.h>
#include <imgui/imgui_internal.h>
#include <nvgui/property_editor.hpp>

namespace vk_gaussian_splatting {

void GaussianSplattingUI::guiDrawDepthStreamProperties()
{
  namespace PE = nvgui::PropertyEditor;

  if(ImGui::CollapsingHeader("Depth Streaming", ImGuiTreeNodeFlags_DefaultOpen))
  {
    PE::begin("##Depth Streaming");

    static char hostBuffer[256] = "192.168.1.200";
    PE::entry("Host IP", [&]() {
      return ImGui::InputText("##Host", hostBuffer, sizeof(hostBuffer));
    });

    PE::entry("Connect", [&]() {
      if(m_enableDepthRendering)
      {
        if(ImGui::Button("Disconnect"))
        {
          m_enableDepthRendering = false;
        }
      }
      else
      {
        if(ImGui::Button("Connect"))
        {
          enableDepthRendering(hostBuffer);
        }
      }
      return false;
    });

    if(m_enableDepthRendering)
    {
      PE::entry("Depth Scale", [&]() {
        return ImGui::DragFloat("##Scale", &m_depthScale, 0.01f, 0.1f, 10.0f);
      });
      
      PE::entry("Depth Bias", [&]() {
        return ImGui::DragFloat("##Bias", &m_depthBias, 0.01f, -5.0f, 5.0f);
      });
    }

    PE::end();
  }
}

void GaussianSplattingUI::guiDrawPerformancePanel()
{
    if (ImGui::Begin("Performance Telemetry")) {
        auto metrics = m_perfStats.getAllMetrics();
        
        if (ImGui::BeginTable("Metrics", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
            ImGui::TableSetupColumn("Name");
            ImGui::TableSetupColumn("Current");
            ImGui::TableSetupColumn("Avg");
            ImGui::TableSetupColumn("Min/Max");
            ImGui::TableHeadersRow();

            for (const auto& [name, metric] : metrics) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%s", name.c_str());
                
                ImGui::TableNextColumn();
                ImGui::Text("%.2f", metric.current);
                
                ImGui::TableNextColumn();
                ImGui::Text("%.2f", metric.avg);
                
                ImGui::TableNextColumn();
                ImGui::Text("%.2f / %.2f", metric.min, metric.max);
            }
            ImGui::EndTable();
        }
        
        for (const auto& [name, metric] : metrics) {
            if (!metric.historyForPlotting.empty()) {
                std::vector<float> values(metric.historyForPlotting.begin(), metric.historyForPlotting.end());
                ImGui::PlotLines(name.c_str(), values.data(), (int)values.size(), 0, nullptr, FLT_MAX, FLT_MAX, ImVec2(0, 50));
            }
        }
    }
    ImGui::End();
}

}
