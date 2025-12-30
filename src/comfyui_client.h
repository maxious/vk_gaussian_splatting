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

#ifndef _COMFYUI_CLIENT_H_
#define _COMFYUI_CLIENT_H_

#ifdef WITH_COMFYUI

#include <string>
#include <functional>
#include <thread>
#include <mutex>
#include <atomic>
#include <filesystem>
#include <queue>

#include <ixwebsocket/IXWebSocket.h>
#include <ixwebsocket/IXHttpClient.h>
#include <tinygltf/json.hpp>

namespace vk_gaussian_splatting {

class ComfyUIClient
{
public:
    enum class State
    {
        Disconnected,
        Connecting,
        Connected,
        Running,
        Completed,
        Error
    };

    struct WorkflowResult
    {
        bool        success = false;
        std::string plyPath;
        std::string errorMessage;
    };

    using ProgressCallback = std::function<void(int current, int total, const std::string& nodeName)>;
    using CompletionCallback = std::function<void(const WorkflowResult& result)>;

public:
    ComfyUIClient();
    ~ComfyUIClient();

    bool connect(const std::string& host = "192.168.1.200", uint16_t port = 8188);
    void disconnect();

    bool queueWorkflow(const std::filesystem::path& workflowPath,
                       const std::string& positivePrompt,
                       const std::string& negativePrompt = "");

    void setProgressCallback(ProgressCallback callback) { m_progressCallback = callback; }
    void setCompletionCallback(CompletionCallback callback) { m_completionCallback = callback; }

    State getState() const { return m_state.load(); }
    std::string getLastError() const { std::lock_guard<std::mutex> lock(m_mutex); return m_lastError; }
    std::string getOutputPlyPath() const { std::lock_guard<std::mutex> lock(m_mutex); return m_outputPlyPath; }

    int getProgressCurrent() const { return m_progressCurrent.load(); }
    int getProgressTotal() const { return m_progressTotal.load(); }

    void update();

private:
    void onMessage(const ix::WebSocketMessagePtr& msg);
    
    // Helper function for sending HTTP POST requests
    bool sendHttpPost(const std::string& endpoint, const std::string& body, std::string& response);
    
    nlohmann::json loadAndModifyWorkflow(const std::filesystem::path& workflowPath,
                                         const std::string& positivePrompt,
                                         const std::string& negativePrompt);

    std::string generateClientId();

private:
    ix::WebSocket                   m_webSocket;
    std::atomic<State>              m_state{State::Disconnected};
    std::atomic<bool>               m_shouldRun{false};

    std::string                     m_host;
    uint16_t                        m_port{8188};
    std::string                     m_clientId;
    std::string                     m_currentPromptId;

    mutable std::mutex              m_mutex;
    std::string                     m_lastError;
    std::string                     m_outputPlyPath;

    std::atomic<int>                m_progressCurrent{0};
    std::atomic<int>                m_progressTotal{0};

    ProgressCallback                m_progressCallback;
    CompletionCallback              m_completionCallback;

    std::queue<WorkflowResult>      m_pendingResults;
};

}  // namespace vk_gaussian_splatting

#endif  // WITH_COMFYUI

#endif  // _COMFYUI_CLIENT_H_
