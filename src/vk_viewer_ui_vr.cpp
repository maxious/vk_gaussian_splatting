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
// Contains: VR/XR hand mesh rendering and wrist button handling

#ifdef WITH_OPENXR

void VkViewerUI::onWristButtonPressed()
{
    m_showVrMenu = !m_showVrMenu;
}

void VkViewerUI::onXrInitialized()
{
    // Initialize hand meshes now that XR session is ready with hand trackers
    if (initHandMeshes()) {
        LOGI("Hand meshes initialized successfully\n");
    }
}

bool VkViewerUI::initHandMeshes()
{
    if (!m_xr || !m_xr->handsSupported())
        return false;

    // Define vertex structure for GPU
    struct HandVertex {
        glm::vec3 position;
        glm::vec3 normal;
        glm::vec2 uv;
        glm::ivec4 blendIndices;
        glm::vec4 blendWeights;
    };

    // Initialize each hand mesh
    for (int handIdx = 0; handIdx < 2; ++handIdx) {
        GsOpenXr::Hand hand = (handIdx == 0) ? GsOpenXr::Hand::Left : GsOpenXr::Hand::Right;
        VkViewerUI::HandMeshVk& mesh = (handIdx == 0) ? m_leftHandMesh : m_rightHandMesh;

        XrHandTrackerEXT tracker = m_xr->getHandTracker(hand);
        if (tracker == XR_NULL_HANDLE) {
            LOGW("Hand tracker not available for %s hand\n", hand == GsOpenXr::Hand::Left ? "left" : "right");
            continue;
        }

        // First call to get buffer sizes
        XrHandTrackingMeshFB handMesh{XR_TYPE_HAND_TRACKING_MESH_FB};
        XrResult result = m_xr->getHandMeshFB(tracker, &handMesh);
        if (result != XR_SUCCESS) {
            LOGW("Failed to get hand mesh info for %s hand: %d\n", hand == GsOpenXr::Hand::Left ? "left" : "right", (int)result);
            continue;
        }

        // Allocate CPU buffers
        mesh.positions.resize(handMesh.vertexCountOutput);
        mesh.normals.resize(handMesh.vertexCountOutput);
        mesh.uvs.resize(handMesh.vertexCountOutput);
        mesh.blendIndices.resize(handMesh.vertexCountOutput);
        mesh.blendWeights.resize(handMesh.vertexCountOutput);
        mesh.indices.resize(handMesh.indexCountOutput);

        // Set capacities and pointers for second call
        handMesh.vertexCapacityInput = static_cast<uint32_t>(mesh.positions.size());
        handMesh.indexCapacityInput = static_cast<uint32_t>(mesh.indices.size());
        handMesh.jointCapacityInput = XR_HAND_JOINT_COUNT_EXT;
        handMesh.vertexPositions = mesh.positions.data();
        handMesh.vertexNormals = mesh.normals.data();
        handMesh.vertexUVs = mesh.uvs.data();
        handMesh.vertexBlendIndices = mesh.blendIndices.data();
        handMesh.vertexBlendWeights = mesh.blendWeights.data();
        handMesh.indices = reinterpret_cast<int16_t*>(mesh.indices.data());
        handMesh.jointBindPoses = mesh.jointBindPoses.data();
        handMesh.jointRadii = mesh.jointRadii.data();
        handMesh.jointParents = mesh.jointParents.data();

        // Second call to fill data
        result = m_xr->getHandMeshFB(tracker, &handMesh);
        if (result != XR_SUCCESS) {
            LOGW("Failed to get hand mesh data for %s hand: %d\n", hand == GsOpenXr::Hand::Left ? "left" : "right", (int)result);
            continue;
        }

        // Copy bind poses
        std::copy(handMesh.jointBindPoses, handMesh.jointBindPoses + XR_HAND_JOINT_COUNT_EXT, mesh.jointBindPoses.begin());

        // Create GPU vertex buffer
        std::vector<HandVertex> vertices(handMesh.vertexCountOutput);
        for (uint32_t i = 0; i < handMesh.vertexCountOutput; ++i) {
            vertices[i].position = glm::vec3(mesh.positions[i].x, mesh.positions[i].y, mesh.positions[i].z);
            vertices[i].normal = glm::vec3(mesh.normals[i].x, mesh.normals[i].y, mesh.normals[i].z);
            vertices[i].uv = glm::vec2(mesh.uvs[i].x, mesh.uvs[i].y);
            vertices[i].blendIndices = glm::ivec4(mesh.blendIndices[i].x, mesh.blendIndices[i].y, mesh.blendIndices[i].z, mesh.blendIndices[i].w);
            vertices[i].blendWeights = glm::vec4(mesh.blendWeights[i].x, mesh.blendWeights[i].y, mesh.blendWeights[i].z, mesh.blendWeights[i].w);
        }

        VkDeviceSize vertexBufferSize = vertices.size() * sizeof(HandVertex);
        m_alloc.createBuffer(mesh.vertexBuffer, vertexBufferSize,
                                         VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                         VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        m_uploader.appendBuffer(mesh.vertexBuffer, 0, std::span(vertices));
        NVVK_DBG_NAME(mesh.vertexBuffer.buffer);

        // Create GPU index buffer
        VkDeviceSize indexBufferSize = mesh.indices.size() * sizeof(uint16_t);
        m_alloc.createBuffer(mesh.indexBuffer, indexBufferSize,
                                         VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                         VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        m_uploader.appendBuffer(mesh.indexBuffer, 0, std::span(mesh.indices));
        NVVK_DBG_NAME(mesh.indexBuffer.buffer);

        // Create joint matrices buffer (dynamic, updated each frame)
        // Use CPU-visible memory with persistent mapping for per-frame updates
        VkDeviceSize jointBufferSize = XR_HAND_JOINT_COUNT_EXT * sizeof(glm::mat4);
        m_alloc.createBuffer(mesh.jointMatricesBuffer, jointBufferSize,
                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                         VMA_MEMORY_USAGE_CPU_TO_GPU,
                                         VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
        NVVK_DBG_NAME(mesh.jointMatricesBuffer.buffer);

        mesh.initialized = true;
        LOGI("Initialized hand mesh for %s hand: %d vertices, %d indices\n",
             hand == GsOpenXr::Hand::Left ? "left" : "right", (int)handMesh.vertexCountOutput, (int)handMesh.indexCountOutput);

    }

    // Upload the appended buffer data using a temporary command buffer
    VkCommandBuffer cmd = m_app->createTempCmdBuffer();
    m_uploader.cmdUploadAppended(cmd);
    m_app->submitAndWaitTempCmdBuffer(cmd);

    // Delay rendering for a few frames to ensure all resources are synchronized
    m_handMeshReadyFrameDelay = 5;
    
    return true;
}

void VkViewerUI::destroyHandMeshes()
{
    for (int hand = 0; hand < 2; ++hand) {
        VkViewerUI::HandMeshVk& mesh = (hand == 0) ? m_leftHandMesh : m_rightHandMesh;

        m_alloc.destroyBuffer(mesh.vertexBuffer);
        m_alloc.destroyBuffer(mesh.indexBuffer);
        m_alloc.destroyBuffer(mesh.jointMatricesBuffer);

        mesh.initialized = false;
    }
}

void VkViewerUI::updateHandMeshes()
{
    if (!m_xr || !m_xr->handsSupported())
        return;

    // Helper to convert XrPosef to glm::mat4
    auto poseToMatrix = [](const XrPosef& pose) -> glm::mat4 {
        glm::quat q(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
        glm::vec3 t(pose.position.x, pose.position.y, pose.position.z);
        return glm::translate(glm::mat4(1.0f), t) * glm::mat4_cast(q);
    };

    for (int handIdx = 0; handIdx < 2; ++handIdx) {
        GsOpenXr::Hand hand = (handIdx == 0) ? GsOpenXr::Hand::Left : GsOpenXr::Hand::Right;
        VkViewerUI::HandMeshVk& mesh = (handIdx == 0) ? m_leftHandMesh : m_rightHandMesh;

        if (!mesh.initialized)
            continue;

        const auto& handInput = m_xr->getHandInput(hand);
        if (!handInput.tracked)
            continue;

        // Compute joint matrices following Meta's skinning approach
        // Get wrist as root for relative transforms
        glm::mat4 wristMatrix = poseToMatrix(handInput.jointPoses[XR_HAND_JOINT_WRIST_EXT]);
        glm::mat4 wristInverse = glm::inverse(wristMatrix);

        for (uint32_t i = 0; i < XR_HAND_JOINT_COUNT_EXT; ++i) {
            const XrPosef& currentPose = handInput.jointPoses[i];
            const XrPosef& bindPose = mesh.jointBindPoses[i];

            glm::mat4 currentMatrix = poseToMatrix(currentPose);
            glm::mat4 bindMatrix = poseToMatrix(bindPose);
            glm::mat4 inverseBind = glm::inverse(bindMatrix);

            // Transform relative to wrist, then apply inverse bind pose
            glm::mat4 modelFromRoot = wristInverse * currentMatrix;
            mesh.jointMatrices[i] = modelFromRoot * inverseBind;
        }

        // Upload joint matrices to GPU (persistently mapped buffer, direct memcpy)
        if (mesh.jointMatricesBuffer.mapping) {
            memcpy(mesh.jointMatricesBuffer.mapping, mesh.jointMatrices.data(), XR_HAND_JOINT_COUNT_EXT * sizeof(glm::mat4));
        }
    }
}

void VkViewerUI::renderHandMesh(VkCommandBuffer cmd, const VkViewerUI::HandMeshVk& mesh, const glm::mat4& wristTransform)
{
    if (!mesh.initialized || !mesh.visible)
        return;

    // Validate all required resources exist
    if (m_graphicsPipelineHandMesh == VK_NULL_HANDLE) {
        LOGD("[Hand] renderHandMesh: pipeline is null\n");
        return;
    }
    if (m_descriptorSet == VK_NULL_HANDLE || m_pipelineLayout == VK_NULL_HANDLE) {
        LOGD("[Hand] renderHandMesh: descriptor set or pipeline layout is null\n");
        return;
    }
    if (mesh.jointMatricesBuffer.buffer == VK_NULL_HANDLE) {
        LOGD("[Hand] renderHandMesh: joint matrices buffer is null\n");
        return;
    }
    if (mesh.vertexBuffer.buffer == VK_NULL_HANDLE || mesh.indexBuffer.buffer == VK_NULL_HANDLE) {
        LOGD("[Hand] renderHandMesh: vertex or index buffer is null\n");
        return;
    }

    // Update descriptor set with this hand's joint matrices buffer
    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = mesh.jointMatricesBuffer.buffer;
    bufferInfo.offset = 0;
    bufferInfo.range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = m_descriptorSet;
    write.dstBinding = BINDING_JOINT_MATRICES;
    write.dstArrayElement = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.descriptorCount = 1;
    write.pBufferInfo = &bufferInfo;

    vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);

    // Bind hand mesh pipeline
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipelineHandMesh);

    // Rebind descriptor set after updating (with dynamic offsets for UBO)
    uint32_t dynamicOffsets[] = {0, 0};  // frameInfo UBO offset, indirect buffer offset
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &m_descriptorSet, 2, dynamicOffsets);

    // Enable depth test and write
    vkCmdSetDepthTestEnable(cmd, VK_TRUE);
    vkCmdSetDepthWriteEnable(cmd, VK_TRUE);

    // Push constants for model matrix (wrist transform positions the hand in world space)
    shaderio::PushConstant pc{};
    pc.modelMatrix = wristTransform;
    pc.modelMatrixInverse = glm::inverse(wristTransform);
    pc.modelMatrixRotScaleInverse = glm::inverse(glm::mat4(glm::mat3(wristTransform)));
    pc.objIndex = 0;
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);

    // Bind vertex buffer
    VkBuffer vertexBuffers[] = {mesh.vertexBuffer.buffer};
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffers, offsets);

    // Bind index buffer
    vkCmdBindIndexBuffer(cmd, mesh.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT16);

    // Draw the mesh
    vkCmdDrawIndexed(cmd, static_cast<uint32_t>(mesh.indices.size()), 1, 0, 0, 0);
}

void VkViewerUI::onRenderMultiviewExtra(VkCommandBuffer cmd)
{
    if (!m_xr || !m_xr->handsSupported() || !m_xrInitialized || m_descriptorSet == VK_NULL_HANDLE)
        return;
    if (m_handMeshReadyFrameDelay > 0)
        return;
    if (m_graphicsPipelineHandMeshMultiview == VK_NULL_HANDLE)
        return;

    auto poseToMatrix = [](const XrPosef& pose) -> glm::mat4 {
        glm::quat q(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
        glm::vec3 t(pose.position.x, pose.position.y, pose.position.z);
        return glm::translate(glm::mat4(1.0f), t) * glm::mat4_cast(q);
    };

    const auto& leftHand = m_xr->getHandInput(GsOpenXr::Hand::Left);
    if (leftHand.tracked) {
        glm::mat4 wristTransform = poseToMatrix(leftHand.jointPoses[XR_HAND_JOINT_WRIST_EXT]);
        renderHandMeshMultiview(cmd, m_leftHandMesh, wristTransform);
    }
    else if (m_leftHandMesh.initialized && m_debugForceRenderHands) {
        glm::mat4 debugTransform = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -0.5f));
        renderHandMeshMultiview(cmd, m_leftHandMesh, debugTransform);
    }

    const auto& rightHand = m_xr->getHandInput(GsOpenXr::Hand::Right);
    if (rightHand.tracked) {
        glm::mat4 wristTransform = poseToMatrix(rightHand.jointPoses[XR_HAND_JOINT_WRIST_EXT]);
        renderHandMeshMultiview(cmd, m_rightHandMesh, wristTransform);
    }
    else if (m_rightHandMesh.initialized && m_debugForceRenderHands) {
        glm::mat4 debugTransform = glm::translate(glm::mat4(1.0f), glm::vec3(0.2f, 0.0f, -0.5f));
        renderHandMeshMultiview(cmd, m_rightHandMesh, debugTransform);
    }
}

void VkViewerUI::renderHandMeshMultiview(VkCommandBuffer cmd, const VkViewerUI::HandMeshVk& mesh, const glm::mat4& wristTransform)
{
    if (!mesh.initialized || !mesh.visible)
        return;

    if (m_graphicsPipelineHandMeshMultiview == VK_NULL_HANDLE)
        return;
    if (m_descriptorSet == VK_NULL_HANDLE || m_pipelineLayout == VK_NULL_HANDLE)
        return;
    if (mesh.jointMatricesBuffer.buffer == VK_NULL_HANDLE)
        return;
    if (mesh.vertexBuffer.buffer == VK_NULL_HANDLE || mesh.indexBuffer.buffer == VK_NULL_HANDLE)
        return;

    // Update descriptor set with this hand's joint matrices buffer
    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = mesh.jointMatricesBuffer.buffer;
    bufferInfo.offset = 0;
    bufferInfo.range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = m_descriptorSet;
    write.dstBinding = BINDING_JOINT_MATRICES;
    write.dstArrayElement = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.descriptorCount = 1;
    write.pBufferInfo = &bufferInfo;

    vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);

    // Bind hand mesh multiview pipeline
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipelineHandMeshMultiview);

    // Rebind descriptor set after updating (with dynamic offsets for UBO)
    uint32_t dynamicOffsets[] = {m_lastFrameInfoOffset, 0};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &m_descriptorSet, 2, dynamicOffsets);

    // Enable depth test and write
    vkCmdSetDepthTestEnable(cmd, VK_TRUE);
    vkCmdSetDepthWriteEnable(cmd, VK_TRUE);

    // Push constants for model matrix
    shaderio::PushConstant pc{};
    pc.modelMatrix = wristTransform;
    pc.modelMatrixInverse = glm::inverse(wristTransform);
    pc.modelMatrixRotScaleInverse = glm::inverse(glm::mat4(glm::mat3(wristTransform)));
    pc.objIndex = 0;
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);

    // Bind vertex buffer
    VkBuffer vertexBuffers[] = {mesh.vertexBuffer.buffer};
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffers, offsets);

    // Bind index buffer
    vkCmdBindIndexBuffer(cmd, mesh.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT16);

    // Draw the mesh
    vkCmdDrawIndexed(cmd, static_cast<uint32_t>(mesh.indices.size()), 1, 0, 0, 0);
}

#endif  // WITH_OPENXR
