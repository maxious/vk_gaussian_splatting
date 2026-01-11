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

// This file is included from vk_viewer_ui.cpp - do not compile separately
// Contains: Depth streaming properties UI (guiDrawDepthStreamProperties)

void VkViewerUI::guiDrawDepthStreamProperties()
{
  namespace PE = nvgui::PropertyEditor;

  if(ImGui::CollapsingHeader("Depth Streaming", ImGuiTreeNodeFlags_DefaultOpen))
  {
    PE::begin("##Depth Streaming");

    // Connection settings
    static char hostBuffer[256] = "192.168.1.200";
    PE::entry("Host", [&]() {
      return ImGui::InputText("##Host", hostBuffer, sizeof(hostBuffer));
    });

    static int port = 8000;
    PE::entry("Port", [&]() {
      return ImGui::InputInt("##Port", &port, 1, 100, ImGuiInputTextFlags_CharsDecimal);
    });

    static std::filesystem::path videoPath;
    static bool backendConnected = false;
    static bool connectionAttempted = false;
    static bool connectionFailed = false;
    static DepthStreamClient::SessionInfo currentSession;
    static bool uploadInProgress = false;
    static bool uploadFailed = false;

    // Initialize backend manager if not already done
    if (!m_backendManager)
    {
      m_backendManager = std::make_unique<BackendProcessManager>();
    }

    // Local backend management
    PE::entry("Local Backend", [&]() {
      bool backendRunning = m_backendManager->isRunning();
      bool backendManaged = m_backendManager->isManaged();

      if (backendRunning)
      {
        ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "✓ Backend running (PID managed)");
        ImGui::SameLine();

        if (ImGui::Button("Stop##StopBackend"))
        {
          m_backendManager->stop();
          m_localBackendStarted = false;
          strncpy(hostBuffer, "192.168.1.200", sizeof(hostBuffer) - 1);
          backendConnected = false;
        }
      }
      else
      {
        if (ImGui::Button("Start Local Backend (XPU)"))
        {
          if (m_backendManager->start())
          {
            strncpy(hostBuffer, "127.0.0.1", sizeof(hostBuffer) - 1);
            port = 8000;

            if (m_depthClient)
            {
              m_depthClient->setBackendAddress(hostBuffer, port);
            }

            m_localBackendStarted = true;

            if (!m_depthClient)
            {
              m_depthClient = std::make_unique<DepthStreamClient>(hostBuffer, port);
            }
            connectionAttempted = true;
            backendConnected = m_depthClient->testConnection();
            connectionFailed = !backendConnected;
          }
        }

        if (backendManaged && !backendRunning)
        {
          ImGui::SameLine();
          ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "(exited)");
        }
      }

      return false;
    });

    ImGui::Separator();

    if(connectionAttempted)
    {
      if(backendConnected)
      {
        ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "✓ Connected to backend");
      }
      else if(connectionFailed)
      {
        ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "✗ Failed to connect to backend");
      }
    }

    if(uploadInProgress)
    {
      ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "⟳ Uploading video...");
    }
    else if(uploadFailed)
    {
      ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "✗ Upload failed");
    }
    else if(m_enableDepthRendering)
    {
      ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "✓ Streaming active - Session: %s", currentSession.sessionId.c_str());
    }

    if(backendConnected)
    {

      PE::entry("Video File", [&]() {
        std::string displayText = videoPath.empty() ? "No file selected" : videoPath.filename().string();
        ImGui::Text("%s", displayText.c_str());

        if(ImGui::Button("Select Video/Image File..."))
        {
          auto path = nvgui::windowOpenFileDialog(m_app->getWindowHandle(), "Select depth video/image file",
                                                  "Video/Image Files|*.mp4;*.avi;*.mov;*.mkv;*.jpg;*.jpeg;*.png|All Files|*.*");
          if(!path.empty())
          {
            videoPath = path;
            return true;
          }
        }
        return false;
      });

      // Upload/Connect button
      PE::entry("Upload Video", [&]() {
        if(m_enableDepthRendering)
        {
          if(ImGui::Button("Disconnect"))
          {
            m_enableDepthRendering = false;
            backendConnected = false;
            uploadFailed = false;
            uploadInProgress = false;
            if (m_depthClient) {
              m_depthClient->disconnectWebSocket();
            }
          }
        }
        else
        {
          if(ImGui::Button("Upload & Start") && !videoPath.empty() && !uploadInProgress)
          {
            uploadInProgress = true;
            uploadFailed = false;

            std::string ext = videoPath.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            bool isImage = (ext == ".jpg" || ext == ".jpeg" || ext == ".png");

            if (isImage)
            {
                // Image processing logic
                std::vector<uint8_t> plyData;
                
                // Use processImagePath if file is local and backend is on localhost, otherwise upload
                bool success = false;
                if (std::string(hostBuffer) == "127.0.0.1" || std::string(hostBuffer) == "localhost")
                {
                    success = m_depthClient->processImagePath(videoPath, plyData);
                }
                else
                {
                    success = m_depthClient->uploadImage(videoPath, plyData);
                }

                if (success && !plyData.empty())
                {
                    // Write PLY data to temp file for loading via existing file-based API
                    std::filesystem::path tempPlyPath = std::filesystem::temp_directory_path() / "temp_gs_image.ply";
                    std::ofstream out(tempPlyPath, std::ios::binary);
                    out.write(reinterpret_cast<const char*>(plyData.data()), plyData.size());
                    out.close();

                    prmScene.sceneToLoadFilename = tempPlyPath;
                    prmScene.addSceneToExisting = false;
                    uploadInProgress = false;
                    
                    LOGI("Loaded 3DGS from image: %s\n", videoPath.string().c_str());
                }
                else
                {
                    LOGE("Failed to process image\n");
                    uploadFailed = true;
                    uploadInProgress = false;
                }
            }
            else
            {
                // Video processing logic
                if (m_depthClient && m_depthClient->uploadVideo(videoPath, currentSession))
                {
                  if (m_depthClient->connectWebSocket(currentSession.sessionId))
                  {
                    enableDepthRendering(hostBuffer, port, videoPath.string());
                    uploadInProgress = false;
                  }
                  else
                  {
                    LOGE("Failed to connect WebSocket\n");
                    uploadFailed = true;
                    uploadInProgress = false;
                  }
                }
                else
                {
                    LOGE("Failed to upload video\n");
                    uploadFailed = true;
                    uploadInProgress = false;
                }
            }
          }
        }
        return false;
      });
    }
    else
    {
      // Connect button
      PE::entry("Connect", [&]() {
        if(ImGui::Button("Connect to Backend"))
        {
          if (!m_depthClient) {
            m_depthClient = std::make_unique<DepthStreamClient>();
          }
          m_depthClient->setBackendAddress(hostBuffer, port);
          connectionAttempted = true;
          backendConnected = m_depthClient->testConnection();
          connectionFailed = !backendConnected;
        }
        return false;
      });
    }

    // Playback controls (visible when streaming is active - NOT offline mode)
    if(m_enableDepthRendering && !m_videoDepthPlaybackMode)
    {
      ImGui::SeparatorText("Playback");

      // Get current stats
      DepthStreamClient::ClientStats stats = m_depthClient ? m_depthClient->getStats() : DepthStreamClient::ClientStats{};

      // Frame counter
      ImGui::Text("Frames: %zu", static_cast<size_t>(stats.totalFrames));

      // Playback buttons
      ImGui::PushID("PlaybackControls");
      if(ImGui::Button(ICON_MS_PLAY_ARROW "##play"))
      {
        m_playbackPaused = false;
      }
      ImGui::SameLine();
      if(ImGui::Button(ICON_MS_PAUSE "##pause"))
      {
        m_playbackPaused = true;
      }
      ImGui::SameLine();
      if(ImGui::Button(ICON_MS_RESTART_ALT "##restart"))
      {
        m_playbackTimeOffset = 0.0;
        m_playbackStartTime = std::chrono::steady_clock::now();
        m_playbackPaused = false;
      }
      ImGui::PopID();

      // Timeline scrubber
      float durationSec = currentSession.durationMs / 1000.0f;
      if(durationSec > 0)
      {
        float currentTimeSec = m_playbackTimeOffset / 1000.0f;
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
        if(ImGui::SliderFloat("##timeline", &currentTimeSec, 0.0f, durationSec))
        {
          // Seeking - restart playback from this position
          m_playbackStartTime = std::chrono::steady_clock::now();
          m_playbackTimeOffset = currentTimeSec * 1000.0f;
          m_playbackPaused = false;
        }
        ImGui::SameLine();
        ImGui::Text("%.1fs / %.1fs", currentTimeSec, durationSec);
      }

      ImGui::SeparatorText("Performance");

      // Backend telemetry from depth client
      if(ImGui::BeginTable("BackendTelemetry", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg))
      {
        ImGui::TableSetupColumn("Metric");
        ImGui::TableSetupColumn("Value");
        ImGui::TableHeadersRow();

        // Inference time
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("Inference");
        ImGui::TableNextColumn();
        ImGui::Text("%.1f ms", stats.inferTimeMs);

        // Decode time
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("Decode");
        ImGui::TableNextColumn();
        ImGui::Text("%.1f ms", stats.decodeTimeMs);

        // Pack time
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("Pack");
        ImGui::TableNextColumn();
        ImGui::Text("%.1f ms", stats.packTimeMs);

        // Queue wait time
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("Queue Wait");
        ImGui::TableNextColumn();
        ImGui::Text("%.1f ms", stats.queueWaitTimeMs);

        // RTT and jitter
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("RTT");
        ImGui::TableNextColumn();
        ImGui::Text("%.1f ms", stats.rttMs);

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("Jitter");
        ImGui::TableNextColumn();
        ImGui::Text("%.1f ms", stats.jitterMs);

        // Depth FPS
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("Depth FPS");
        ImGui::TableNextColumn();
        ImGui::Text("%.1f", stats.fps);

        // Dropped frames
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("Dropped");
        ImGui::TableNextColumn();
        ImGui::Text("%zu", static_cast<size_t>(stats.droppedFrames));

        ImGui::EndTable();
      }

      // Video texture toggle for streaming mode
      ImGui::SeparatorText("Display");
      bool useVideoTexture = prmFrame.vdzUseVideoTexture != 0;
      if(ImGui::Checkbox("Use Video Texture", &useVideoTexture))
      {
        prmFrame.vdzUseVideoTexture = useVideoTexture ? 1 : 0;
      }
      ImGui::SameLine();
      nvgui::tooltip("Show video RGB when enabled, depth colormap when disabled");
    }

    if(m_enableDepthRendering && !m_videoDepthPlaybackMode)
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

  if(ImGui::CollapsingHeader("Offline Video+Depth", m_videoDepthPlaybackMode ? ImGuiTreeNodeFlags_DefaultOpen : 0))
  {
    PE::begin("##Offline Video+Depth");

    if(!m_videoDepthPlaybackMode && !m_hlsPlaybackMode)
    {
      PE::entry("Load Video+Depth", [this]() {
        static std::filesystem::path videoPath;

        if(ImGui::Button("Load Video..."))
        {
          videoPath = nvgui::windowOpenFileDialog(m_app->getWindowHandle(), "Select Video File", "Video Files|*.mp4;*.avi;*.mov;*.mkv");
        }
        ImGui::SameLine();
        ImGui::Text("%s", videoPath.empty() ? "(none)" : videoPath.filename().string().c_str());

        return false;
      });

      PE::entry("Load HLS Stream", [this]() {
        static std::filesystem::path hlsPath;

        if(ImGui::Button("Select metadata.json..."))
        {
          hlsPath = nvgui::windowOpenFileDialog(m_app->getWindowHandle(), "Select HLS Metadata", "Metadata Files|metadata.json");
        }
        ImGui::SameLine();
        ImGui::Text("%s", hlsPath.empty() ? "(none)" : hlsPath.filename().string().c_str());

        if(!hlsPath.empty() && ImGui::Button("Load HLS"))
        {
          enableHlsPlayback(hlsPath.string());
        }

        return false;
      });

      PE::entry("Generate HLS from Video", [this]() {
        static std::filesystem::path videoPath;
        static bool uploading = false;
        static bool generating = false;
        static DepthStreamClient::HlsGenerationStatus hlsStatus;
        static DepthStreamClient::SessionInfo hlsSession;
        static double lastStatusPoll = 0.0;
        static bool hlsReady = false;

        if(ImGui::Button("Select Video..."))
        {
          videoPath = nvgui::windowOpenFileDialog(m_app->getWindowHandle(), "Select Video File", "Video Files|*.mp4;*.avi;*.mov;*.mkv");
          uploading = false;
          generating = false;
          hlsReady = false;
        }
        ImGui::SameLine();
        ImGui::Text("%s", videoPath.empty() ? "(none)" : videoPath.filename().string().c_str());

        if(!videoPath.empty() && !hlsReady)
        {
          if(!uploading && !generating)
          {
            if(ImGui::Button("Upload & Generate HLS"))
            {
              uploading = true;
              generating = false;
              hlsReady = false;

              if(m_depthClient->createHlsSession(videoPath, hlsSession))
              {
                if(m_depthClient->startHlsGeneration(hlsSession.sessionId, hlsSession.fps, 640))
                {
                  generating = true;
                  uploading = false;
                  lastStatusPoll = 0.0;
                }
                else
                {
                  LOGE("Failed to start HLS generation\n");
                  uploading = false;
                  generating = false;
                }
              }
              else
              {
                LOGE("Failed to create HLS session\n");
                uploading = false;
              }
            }
          }
        }

        // Display upload/generation status with ETA
        if(uploading)
        {
          ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Uploading video...");
        }

        if(generating)
        {
          auto now = ImGui::GetTime();
          if(now - lastStatusPoll > 0.5)
          {
            m_depthClient->getHlsStatus(hlsSession.sessionId, hlsStatus);
            lastStatusPoll = now;

            if(hlsStatus.isReady())
            {
              generating = false;
              hlsReady = true;
            }
            else if(hlsStatus.hasError())
            {
              generating = false;
              ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Error: %s", hlsStatus.errorMessage.c_str());
            }
          }

          if(generating)
          {
            ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Generating HLS stream...");

            // Progress bar
            ImGui::ProgressBar(hlsStatus.progress, ImVec2(-FLT_MIN, 0.0f));

            // Progress details
            ImGui::Text("Frames: %d / %d", hlsStatus.frameCount, hlsStatus.totalFrames);
            ImGui::SameLine();

            // ETA display
            if(hlsStatus.etaSeconds > 0 && hlsStatus.etaSeconds < 3600)
            {
              int etaMinutes = static_cast<int>(hlsStatus.etaSeconds / 60);
              int etaSeconds = static_cast<int>(hlsStatus.etaSeconds) % 60;
              ImGui::Text("ETA: %d:%02d", etaMinutes, etaSeconds);
            }
            else if(hlsStatus.etaSeconds >= 3600)
            {
              int etaHours = static_cast<int>(hlsStatus.etaSeconds / 3600);
              int etaMinutes = static_cast<int>(hlsStatus.etaSeconds) % 3600 / 60;
              ImGui::Text("ETA: %d:%02d", etaHours, etaMinutes);
            }
            else
            {
              ImGui::Text("ETA: calculating...");
            }

            ImGui::SameLine();
            ImGui::Text("(%.1f FPS)", hlsStatus.framesPerSecond);
          }
        }

        // Show load button when HLS is ready
        if(hlsReady)
        {
          ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "HLS stream ready!");

          std::string metadataPath = m_depthClient->getHlsMetadataPath(hlsSession.sessionId);
          ImGui::SameLine();
          if(ImGui::Button("Load"))
          {
            enableHlsPlayback(metadataPath);
            hlsReady = false;
            generating = false;
          }
        }

        return false;
      });
    }
    else
    {
#ifdef WITH_VIDEO_DECODER
      if(m_videoDepthManager)
      {
        const auto& metadata = m_videoDepthManager->getMetadata();
        
        float bufferedRatio = m_videoDepthManager->getBufferedRatio();
        double bufferedSec = m_videoDepthManager->getBufferedDuration();
        double totalSec = m_videoDepthManager->getDuration();
        
        if(m_videoDepthManager->isBuffering())
        {
          ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Buffering: %.0f%%", bufferedRatio * 100.0f);
          ImGui::ProgressBar(bufferedRatio, ImVec2(-FLT_MIN, 0.0f));
        }
        else
        {
          ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "✓ Buffered");
        }
        
        ImGui::Text("Frames: %d / %d", m_depthFrameCounter, metadata.frameCount);
        
        double currentTime = m_videoDepthManager->getCurrentTime();
        float sliderMax = static_cast<float>(bufferedSec > 0.0 ? bufferedSec : totalSec);
        float tFloat = static_cast<float>(currentTime);
        
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 100.0f);
        if(ImGui::SliderFloat("##offline_timeline", &tFloat, 0.0f, sliderMax, ""))
        {
          m_videoDepthManager->seek(static_cast<double>(tFloat));
          m_lastVdzFrameIndex = SIZE_MAX;
        }
        ImGui::SameLine();
        ImGui::Text("%.1fs / %.1fs", currentTime, totalSec);
      }
#endif
      
      bool useVideo = prmFrame.vdzUseVideoTexture != 0;
      if(PE::Checkbox("Use Video Texture", &useVideo,
                      "When enabled, uses the video RGB texture.\n"
                      "When disabled, shows depth as a colormap."))
      {
        prmFrame.vdzUseVideoTexture = useVideo ? 1 : 0;
      }
      
      PE::entry("Playback", [this]() {
        bool changed = false;
#ifdef WITH_VIDEO_DECODER
        if(!m_videoDepthManager) return false;
        
        bool atEnd = m_videoDepthManager->isAtEnd();
        
        if(atEnd && !m_playbackPaused)
        {
          ImGui::Text("Finished");
          ImGui::SameLine();
        }
        
        if(m_playbackPaused)
        {
          if(ImGui::Button(ICON_MS_PLAY_ARROW " Play"))
          {
            m_playbackPaused = false;
            m_videoDepthManager->play();
            changed = true;
          }
        }
        else
        {
          if(ImGui::Button(ICON_MS_PAUSE " Pause"))
          {
            m_playbackPaused = true;
            m_videoDepthManager->pause();
            changed = true;
          }
        }
        
        ImGui::SameLine();
        if(ImGui::Button(ICON_MS_RESTART_ALT " Restart"))
        {
          m_videoDepthManager->seek(0.0);
          m_videoDepthManager->play();
          m_lastVdzFrameIndex = SIZE_MAX;
          m_playbackPaused = false;
          changed = true;
        }
#endif
        return changed;
      });

      if(ImGui::Button("Stop Playback"))
      {
        m_enableDepthRendering = false;
        m_videoDepthPlaybackMode = false;
        m_hlsPlaybackMode = false;
#ifdef WITH_VIDEO_DECODER
        if(m_videoDepthManager)
        {
          m_videoDepthManager->close();
          m_videoDepthManager.reset();
        }
        if(m_hlsPlayer)
        {
          m_hlsPlayer->stop();
          m_hlsPlayer.reset();
        }
#endif
      }
    }

    PE::end();
  }



  if(m_enableDepthRendering && ImGui::CollapsingHeader("Depth Mesh Settings", ImGuiTreeNodeFlags_DefaultOpen))
  {
    PE::begin("##Depth Mesh Settings");

    PE::SliderFloat("Z Scale", &prmFrame.vdzZScale, 0.0f, 10.0f, "%.2f", 0,
                    "Depth scale multiplier");
    PE::SliderFloat("Z Bias", &prmFrame.vdzZBias, -5.0f, 5.0f, "%.2f", 0,
                    "Global Z offset added after scaling");
    PE::SliderFloat("Z Gamma", &prmFrame.vdzZGamma, 0.1f, 5.0f, "%.2f", 0,
                    "Gamma correction for depth");
    PE::SliderFloat("Z Max Clip", &prmFrame.vdzZMaxClip, 0.0f, 10.0f, "%.2f", 0,
                    "Maximum depth clipping threshold");
    PE::SliderFloat("Plane Scale", &prmFrame.vdzPlaneScale, 0.1f, 10.0f, "%.2f", 0,
                    "Scale of the view-aligned plane");
    PE::SliderFloat("Aspect Ratio", &prmFrame.vdzAspect, 0.5f, 3.0f, "%.3f", 0,
                    "Aspect ratio (width/height) of the depth texture");
    PE::SliderFloat("Edge Threshold", &prmFrame.vdzEdgeThreshold, 0.0f, 1.0f, "%.3f", 0,
                    "Depth gradient threshold for edge detection");

    bool useVideoTexture = prmFrame.vdzUseVideoTexture != 0;
    if(PE::Checkbox("Use Video Texture", &useVideoTexture,
                    "Show video RGB when enabled, depth colormap when disabled."))
    {
      prmFrame.vdzUseVideoTexture = useVideoTexture ? 1 : 0;
    }

    // Parallax rendering controls
    if(ImGui::TreeNode("Parallax Rendering"))
    {
      // Control mode indicator
      bool hasSplats = m_splatSet.positions.size() > 0 || m_splatLoader.getStatus() == SplatLoaderAsync::State::STATE_READY;
      bool hasMeshes = !m_meshSetVk.instances.empty();
      bool isDepthOnly = m_enableDepthRendering && !hasSplats && !hasMeshes;

      if(isDepthOnly)
      {
        ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 200, 100, 255));
        ImGui::Text(ICON_MS_VIDEOCAM " Depth Video Mode");
        ImGui::PopStyleColor();

        // Desktop controls help
        ImGui::BeginDisabled();
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
        ImGui::InputTextWithHint("##controls", "LMB: parallax | WASD: timeline | Space: play/pause",
                                 nullptr, ImGuiInputTextFlags_ReadOnly);
        ImGui::EndDisabled();
        ImGui::SameLine();
        if(ImGui::SmallButton(ICON_MS_HELP "?##parallaxHelp"))
        {
          ImGui::OpenPopup("parallax_controls_help");
        }
        if(ImGui::BeginPopup("parallax_controls_help"))
        {
          ImGui::TextUnformatted("Desktop Controls:");
          ImGui::Separator();
          ImGui::TextWrapped("LMB drag: Adjust parallax (simulates head movement)");
          ImGui::TextWrapped("Mouse wheel: Adjust focus plane");
          ImGui::TextWrapped("WASD / Arrows: Scrub timeline");
          ImGui::TextWrapped("Home/End: Jump to start/end");
          ImGui::TextWrapped("Space: Play/Pause");
          ImGui::EndPopup();
        }

        // VR controls display
#ifdef WITH_OPENXR
        if(m_xrInitialized)
        {
          ImGui::Spacing();
          ImGui::Text("VR Controls:");
          ImGui::SameLine();
          ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(100, 200, 255, 255));
          ImGui::Text(ICON_MS_HEADSET " Left stick: tilt | Right stick: parallax");
          ImGui::PopStyleColor();
        }
#endif
      }
      else
      {
        ImGui::TextColored(ImVec4(0.4f, 0.8f, 0.4f, 1.0f), ICON_MS_SCENE " Scene Mode");
      }

      ImGui::Separator();

      PE::SliderFloat("Strength", &prmFrame.vdzParallaxStrength, 0.0f, 2.0f, "%.2f", 0,
                      "Parallax intensity multiplier (0.0 = disabled)");
      PE::SliderFloat("Focus Plane", &prmFrame.vdzParallaxFocus, 0.0f, 1.0f, "%.2f", 0,
                      "Depth of the focus plane (0=near, 1=far)");
      PE::SliderFloat("Edge Softness", &prmFrame.vdzParallaxEdgeSoftness, 0.0f, 0.1f, "%.3f", 0,
                      "Edge softening factor to reduce artifacts at borders");
      
      // Display parallax offset (for debugging)
      ImGui::Text("Offset: %.3f, %.3f", prmFrame.vdzParallaxOffset.x, prmFrame.vdzParallaxOffset.y);
      
      ImGui::TreePop();
    }

    // Hybrid rendering controls (mesh + POM)
    if(ImGui::TreeNode("Hybrid Rendering (Mesh + POM)"))
    {
      const char* modeNames[] = {"Divided Mesh", "POM Only", "Hybrid (Mesh + POM)"};
      int currentMode = prmFrame.vdzHybridMode;
      
      if(ImGui::Combo("Rendering Mode", &currentMode, modeNames, 3))
      {
        prmFrame.vdzHybridMode = currentMode;
        
        // Reinitialize mesh based on mode
        if(currentMode == 1)  // POM Only - use simple quad
        {
          getVdzMesh().generateQuad();
        }
        else if(currentMode == 2)  // Hybrid - use lower resolution grid
        {
          // Hybrid mode uses lower resolution grid (32x18) for better performance
          getVdzMesh().generateHybridGrid(32, 18);
        }
        else  // Divided Mesh - use standard resolution
        {
          getVdzMesh().reinitialize(128, 72);
        }
      }
      
      nvgui::tooltip("Divided Mesh: Original approach with high-resolution grid\n"
                     "POM Only: Simple quad with parallax occlusion mapping\n"
                     "Hybrid: Low-res mesh + POM for best quality/performance");
      
      if(prmFrame.vdzHybridMode == 2)  // Hybrid mode controls
      {
        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Hybrid Settings:");
        
        PE::SliderFloat("Mesh Strength", &prmFrame.vdzHybridMeshStrength, 0.0f, 1.0f, "%.2f", 0,
                        "Strength of mesh vertex displacement (0-1)");
        PE::SliderFloat("POM Strength", &prmFrame.vdzHybridPomStrength, 0.0f, 1.0f, "%.2f", 0,
                        "Strength of POM effect (0-1)");
        PE::SliderInt("POM Layers", &prmFrame.vdzHybridPomLayers, 4, 32, "%d", 0,
                      "Number of POM ray-march layers (higher = better quality, more GPU cost)");
        PE::SliderFloat("POM Depth Scale", &prmFrame.vdzHybridPomDepthScale, 0.001f, 0.1f, "%.3f", 0,
                        "Height scale for POM in UV space");
      }
      
      ImGui::TreePop();
    }

    // Gap filling controls
    if(ImGui::TreeNode("Gap Filling"))
    {
      bool gapFillEnabled = prmFrame.vdzGapFillEnabled != 0;
      if(PE::Checkbox("Enable Gap Fill", &gapFillEnabled,
                      "Fill invalid depth regions instead of discarding"))
      {
        prmFrame.vdzGapFillEnabled = gapFillEnabled ? 1 : 0;
      }
      
      if(gapFillEnabled)
      {
        float gapFillColor[3] = {prmFrame.vdzGapFillColor[0], prmFrame.vdzGapFillColor[1], prmFrame.vdzGapFillColor[2]};
        if(ImGui::ColorEdit3("Fill Color", gapFillColor, ImGuiColorEditFlags_Float))
        {
          prmFrame.vdzGapFillColor[0] = gapFillColor[0];
          prmFrame.vdzGapFillColor[1] = gapFillColor[1];
          prmFrame.vdzGapFillColor[2] = gapFillColor[2];
        }
        
        bool gapFillNeighbor = prmFrame.vdzGapFillNeighbor != 0;
        if(PE::Checkbox("Neighbor Average", &gapFillNeighbor,
                        "Use average of neighboring pixels instead of solid color"))
        {
          prmFrame.vdzGapFillNeighbor = gapFillNeighbor ? 1 : 0;
        }
        
        if(gapFillNeighbor)
        {
          PE::SliderInt("Neighbor Radius", &prmFrame.vdzGapFillRadius, 1, 8, "%d", 0,
                        "Radius for neighbor sampling (in pixels)");
        }
        
        PE::SliderFloat("Edge Feather", &prmFrame.vdzGapFeather, 0.0f, 2.0f, "%.2f", 0,
                        "Edge feathering amount for smooth transitions");
      }
      
      ImGui::TreePop();
    }

    // Bilateral filter controls
    if(ImGui::TreeNode("Bilateral Filter"))
    {
      bool bilateralEnabled = prmFrame.vdzBilateralEnabled != 0;
      if(PE::Checkbox("Enable Filter", &bilateralEnabled,
                      "Apply edge-preserving smoothing to depth"))
      {
        prmFrame.vdzBilateralEnabled = bilateralEnabled ? 1 : 0;
      }
      
      if(bilateralEnabled)
      {
        PE::SliderFloat("Spatial Sigma", &prmFrame.vdzBilateralSigmaSpace, 0.5f, 10.0f, "%.1f", 0,
                        "Spatial sigma for bilateral filter (larger = smoother)");
        PE::SliderFloat("Depth Sigma", &prmFrame.vdzBilateralSigmaDepth, 0.01f, 1.0f, "%.3f", 0,
                        "Depth range sigma for preserving edges");
      }
      
      ImGui::TreePop();
    }

    bool worldSpace = prmFrame.vdzWorldSpaceMode != 0;
    if(PE::Checkbox("World Space / VR Mode", &worldSpace,
                    "Detach mesh from camera and place it in the world."))
    {
      prmFrame.vdzWorldSpaceMode = worldSpace ? 1 : 0;
      m_requestUpdateShaders = true;
    }

    PE::end();
  }
}
