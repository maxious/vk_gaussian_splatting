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

#ifdef WITH_COMFYUI

#include "comfyui_client.h"

#include <fstream>
#include <sstream>
#include <random>
#include <iomanip>

#include <nvutils/logger.hpp>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netdb.h>
#endif

namespace vk_gaussian_splatting {

ComfyUIClient::ComfyUIClient()
{
    m_clientId = generateClientId();

    m_client.clear_access_channels(websocketpp::log::alevel::all);
    m_client.clear_error_channels(websocketpp::log::elevel::all);

    m_client.init_asio();

    m_client.set_message_handler([this](auto hdl, auto msg) { onMessage(hdl, msg); });
    m_client.set_open_handler([this](auto hdl) { onOpen(hdl); });
    m_client.set_close_handler([this](auto hdl) { onClose(hdl); });
    m_client.set_fail_handler([this](auto hdl) { onFail(hdl); });
}

ComfyUIClient::~ComfyUIClient()
{
    disconnect();
}

std::string ComfyUIClient::generateClientId()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 15);

    std::stringstream ss;
    for (int i = 0; i < 32; ++i)
    {
        ss << std::hex << dis(gen);
    }
    return ss.str();
}

bool ComfyUIClient::connect(const std::string& host, uint16_t port)
{
    if (m_state.load() != State::Disconnected)
    {
        disconnect();
    }

    m_host = host;
    m_port = port;
    m_state.store(State::Connecting);
    m_shouldRun.store(true);

    try
    {
        std::string uri = "ws://" + host + ":" + std::to_string(port) + "/ws?clientId=" + m_clientId;

        websocketpp::lib::error_code ec;
        auto con = m_client.get_connection(uri, ec);

        if (ec)
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_lastError = "Connection error: " + ec.message();
            m_state.store(State::Error);
            return false;
        }

        m_client.connect(con);

        m_ioThread = std::thread(&ComfyUIClient::ioThreadFunc, this);

        return true;
    }
    catch (const std::exception& e)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_lastError = std::string("Exception: ") + e.what();
        m_state.store(State::Error);
        return false;
    }
}

void ComfyUIClient::disconnect()
{
    m_shouldRun.store(false);

    if (m_state.load() == State::Connected || m_state.load() == State::Running)
    {
        try
        {
            m_client.close(m_connectionHdl, websocketpp::close::status::normal, "Client disconnecting");
        }
        catch (...) {}
    }

    m_client.stop();

    if (m_ioThread.joinable())
    {
        m_ioThread.join();
    }

    m_state.store(State::Disconnected);
}

void ComfyUIClient::ioThreadFunc()
{
    try
    {
        m_client.run();
    }
    catch (const std::exception& e)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_lastError = std::string("IO thread exception: ") + e.what();
        m_state.store(State::Error);
    }
}

void ComfyUIClient::onOpen(websocketpp::connection_hdl hdl)
{
    m_connectionHdl = hdl;
    m_state.store(State::Connected);
    LOGI("ComfyUI: WebSocket connected\n");
}

void ComfyUIClient::onClose(websocketpp::connection_hdl hdl)
{
    std::string reason;
    try
    {
        auto con = m_client.get_con_from_hdl(hdl);
        auto code = con->get_remote_close_code();
        auto closeReason = con->get_remote_close_reason();
        reason = "Code " + std::to_string(code);
        if (!closeReason.empty())
        {
            reason += ": " + closeReason;
        }
    }
    catch (...) { reason = "Unknown reason"; }

    std::lock_guard<std::mutex> lock(m_mutex);
    m_lastError = "Connection closed - " + reason;
    m_state.store(State::Disconnected);
    LOGI("ComfyUI: WebSocket closed (%s)\n", reason.c_str());
}

void ComfyUIClient::onFail(websocketpp::connection_hdl hdl)
{
    std::string errorDetail;
    try
    {
        auto con = m_client.get_con_from_hdl(hdl);
        auto ec = con->get_ec();
        auto httpStatus = con->get_response_code();
        auto uri = con->get_uri()->str();

        if (ec)
        {
            errorDetail = ec.message();
            if (ec == websocketpp::error::make_error_code(websocketpp::error::value::invalid_uri))
                errorDetail += " (invalid URI format)";
            else if (ec.value() == 111)  // Connection refused
                errorDetail = "Connection refused - is ComfyUI running at " + uri + "?";
            else if (ec.value() == 110)  // Connection timed out
                errorDetail = "Connection timed out - check if " + uri + " is reachable";
            else if (ec.value() == 113)  // No route to host
                errorDetail = "No route to host - check network connection";
        }
        else if (httpStatus != 0)
        {
            errorDetail = "HTTP error " + std::to_string(httpStatus);
            if (httpStatus == 404)
                errorDetail += " - ComfyUI websocket endpoint not found";
            else if (httpStatus == 403)
                errorDetail += " - Access forbidden";
            else if (httpStatus == 503)
                errorDetail += " - ComfyUI service unavailable";
        }
        else
        {
            errorDetail = "Unknown connection failure to " + uri;
        }
    }
    catch (...)
    {
        errorDetail = "Connection failed (unable to get details)";
    }

    std::lock_guard<std::mutex> lock(m_mutex);
    m_lastError = errorDetail;
    m_state.store(State::Error);
    LOGE("ComfyUI: %s\n", errorDetail.c_str());
}

void ComfyUIClient::onMessage(websocketpp::connection_hdl hdl, WsClient::message_ptr msg)
{
    try
    {
        auto payload = msg->get_payload();
        auto data = nlohmann::json::parse(payload);

        std::string msgType = data.value("type", "");

        if (msgType == "progress")
        {
            auto progressData = data["data"];
            int current = progressData.value("value", 0);
            int total = progressData.value("max", 1);

            m_progressCurrent.store(current);
            m_progressTotal.store(total);

            if (m_progressCallback)
            {
                m_progressCallback(current, total, "");
            }
        }
        else if (msgType == "executing")
        {
            auto execData = data["data"];
            std::string nodeId = execData.value("node", "");

            if (nodeId.empty())
            {
                std::string promptId = execData.value("prompt_id", "");
                if (promptId == m_currentPromptId)
                {
                    m_state.store(State::Completed);

                    WorkflowResult result;
                    result.success = true;
                    {
                        std::lock_guard<std::mutex> lock(m_mutex);
                        result.plyPath = m_outputPlyPath;
                    }

                    {
                        std::lock_guard<std::mutex> lock(m_mutex);
                        m_pendingResults.push(result);
                    }
                }
            }
        }
        else if (msgType == "executed")
        {
            auto execData = data["data"];
            if (execData.contains("output"))
            {
                auto output = execData["output"];
                if (output.contains("ply_path"))
                {
                    std::lock_guard<std::mutex> lock(m_mutex);
                    m_outputPlyPath = output["ply_path"].get<std::string>();
                    LOGI("ComfyUI: PLY output path: %s\n", m_outputPlyPath.c_str());
                }
            }
        }
        else if (msgType == "execution_error")
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_lastError = data.value("message", "Unknown execution error");
            m_state.store(State::Error);

            WorkflowResult result;
            result.success = false;
            result.errorMessage = m_lastError;
            m_pendingResults.push(result);
        }
    }
    catch (const std::exception& e)
    {
        LOGE("ComfyUI: Error parsing message: %s\n", e.what());
    }
}

bool ComfyUIClient::sendHttpPost(const std::string& endpoint, const std::string& body, std::string& response)
{
#ifdef _WIN32
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0)
    {
        return false;
    }
#endif

    int sock = static_cast<int>(socket(AF_INET, SOCK_STREAM, 0));
    if (sock < 0)
    {
#ifdef _WIN32
        WSACleanup();
#endif
        return false;
    }

    struct sockaddr_in server{};
    server.sin_family = AF_INET;
    server.sin_port = htons(m_port);
    inet_pton(AF_INET, m_host.c_str(), &server.sin_addr);

    if (::connect(sock, (struct sockaddr*)&server, sizeof(server)) < 0)
    {
#ifdef _WIN32
        closesocket(sock);
        WSACleanup();
#else
        close(sock);
#endif
        return false;
    }

    std::stringstream request;
    request << "POST " << endpoint << " HTTP/1.1\r\n";
    request << "Host: " << m_host << ":" << m_port << "\r\n";
    request << "Content-Type: application/json\r\n";
    request << "Content-Length: " << body.size() << "\r\n";
    request << "Connection: close\r\n\r\n";
    request << body;

    std::string requestStr = request.str();
    send(sock, requestStr.c_str(), static_cast<int>(requestStr.size()), 0);

    char buffer[4096];
    response.clear();
    int bytesReceived;
    while ((bytesReceived = recv(sock, buffer, sizeof(buffer) - 1, 0)) > 0)
    {
        buffer[bytesReceived] = '\0';
        response += buffer;
    }

#ifdef _WIN32
    closesocket(sock);
    WSACleanup();
#else
    close(sock);
#endif

    return true;
}

nlohmann::json ComfyUIClient::loadAndModifyWorkflow(const std::filesystem::path& workflowPath,
                                                     const std::string& positivePrompt,
                                                     const std::string& negativePrompt)
{
    std::ifstream file(workflowPath);
    if (!file.is_open())
    {
        throw std::runtime_error("Cannot open workflow file: " + workflowPath.string());
    }

    nlohmann::json workflow;
    file >> workflow;
    file.close();

    nlohmann::json apiWorkflow;

    if (workflow.contains("nodes"))
    {
        for (const auto& node : workflow["nodes"])
        {
            std::string nodeId = std::to_string(node["id"].get<int>());
            std::string nodeType = node["type"].get<std::string>();

            nlohmann::json nodeData;
            nodeData["class_type"] = nodeType;
            nodeData["inputs"] = nlohmann::json::object();

            if (node.contains("widgets_values") && !node["widgets_values"].is_null())
            {
                auto& widgets = node["widgets_values"];

                if (nodeType == "CLIPTextEncode")
                {
                    if (nodeId == "29" && !positivePrompt.empty())
                    {
                        nodeData["inputs"]["text"] = positivePrompt;
                    }
                    else if (nodeId == "50" && !negativePrompt.empty())
                    {
                        nodeData["inputs"]["text"] = negativePrompt;
                    }
                    else if (widgets.is_array() && !widgets.empty())
                    {
                        nodeData["inputs"]["text"] = widgets[0];
                    }
                }
            }

            if (node.contains("inputs"))
            {
                for (const auto& input : node["inputs"])
                {
                    if (input.contains("link") && !input["link"].is_null())
                    {
                        int linkId = input["link"].get<int>();
                        std::string inputName = input["name"].get<std::string>();

                        if (workflow.contains("links"))
                        {
                            for (const auto& link : workflow["links"])
                            {
                                if (link[0].get<int>() == linkId)
                                {
                                    int sourceNodeId = link[1].get<int>();
                                    int sourceSlot = link[2].get<int>();
                                    nodeData["inputs"][inputName] = nlohmann::json::array({std::to_string(sourceNodeId), sourceSlot});
                                    break;
                                }
                            }
                        }
                    }
                }
            }

            apiWorkflow[nodeId] = nodeData;
        }
    }

    return apiWorkflow;
}

bool ComfyUIClient::queueWorkflow(const std::filesystem::path& workflowPath,
                                   const std::string& positivePrompt,
                                   const std::string& negativePrompt)
{
    if (m_state.load() != State::Connected)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_lastError = "Not connected to ComfyUI server";
        return false;
    }

    try
    {
        auto apiWorkflow = loadAndModifyWorkflow(workflowPath, positivePrompt, negativePrompt);

        nlohmann::json prompt;
        prompt["prompt"] = apiWorkflow;
        prompt["client_id"] = m_clientId;

        std::string response;
        if (!sendHttpPost("/prompt", prompt.dump(), response))
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_lastError = "Failed to send prompt to ComfyUI";
            return false;
        }

        auto bodyStart = response.find("\r\n\r\n");
        if (bodyStart != std::string::npos)
        {
            std::string body = response.substr(bodyStart + 4);
            auto responseJson = nlohmann::json::parse(body);

            if (responseJson.contains("prompt_id"))
            {
                m_currentPromptId = responseJson["prompt_id"].get<std::string>();
                m_state.store(State::Running);
                m_progressCurrent.store(0);
                m_progressTotal.store(0);

                {
                    std::lock_guard<std::mutex> lock(m_mutex);
                    m_outputPlyPath.clear();
                }

                LOGI("ComfyUI: Workflow queued with prompt_id: %s\n", m_currentPromptId.c_str());
                return true;
            }
            else if (responseJson.contains("error"))
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                m_lastError = responseJson["error"].get<std::string>();
                return false;
            }
        }

        std::lock_guard<std::mutex> lock(m_mutex);
        m_lastError = "Invalid response from ComfyUI";
        return false;
    }
    catch (const std::exception& e)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_lastError = std::string("Exception: ") + e.what();
        return false;
    }
}

void ComfyUIClient::update()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    while (!m_pendingResults.empty())
    {
        auto result = m_pendingResults.front();
        m_pendingResults.pop();

        if (m_completionCallback)
        {
            m_completionCallback(result);
        }
    }
}

}  // namespace vk_gaussian_splatting

#endif  // WITH_COMFYUI
