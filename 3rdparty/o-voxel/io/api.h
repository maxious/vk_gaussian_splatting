/*
 * Efficient Sparse Voxel storage as Sparse Voxel Zip files (.svz)
 *
 * Copyright (C) 2025, Jianfeng XIANG <belljig@outlook.com>
 * All rights reserved.
 *
 * Licensed under The MIT License [see LICENSE for details]
 *
 * Written by Jianfeng XIANG
 */

#pragma once
#include <cstdint>
#include <vector>

/**
 * Encode a list of sparse voxel morton codes into a sparse voxel octree
 * NOTE: The input indices must be sorted in ascending order
 * 
 * @param codes    [N] vector containing the morton codes
 * @param depth    The depth of the sparse voxel octree
 * 
 * @return         vector containing the sparse voxel octree (uint8)
 */
std::vector<uint8_t> encode_sparse_voxel_octree_cpu(
    const std::vector<uint32_t>& codes,
    const uint32_t depth
);


/**
 * Decode a sparse voxel octree into a list of sparse voxel morton codes
 * 
 * @param octree   vector containing the sparse voxel octree (uint8)
 * @param depth    The depth of the sparse voxel octree
 * 
 * @return         [N] vector containing the morton codes (uint32)
 *                 The codes are sorted in ascending order
 */
std::vector<uint32_t> decode_sparse_voxel_octree_cpu(
    const std::vector<uint8_t>& octree,
    const uint32_t depth
);



/**
 * Encode the attribute of a sparse voxel octree into deltas from its parent node.
 * 
 * @param octree        vector containing the sparse voxel octree (uint8)
 * @param depth         The depth of the sparse voxel octree
 * @param attr          [N * C] vector containing the attribute of each sparse voxel (flattened)
 * @param num_channels  Number of channels C
 * 
 * @return              vector containing the deltas (uint8)
 */
std::vector<uint8_t> encode_sparse_voxel_octree_attr_parent_cpu(
    const std::vector<uint8_t>& octree,
    const uint32_t depth,
    const std::vector<uint8_t>& attr,
    const uint32_t num_channels
);


/**
 * Decode the attribute of a sparse voxel octree from its parent node and its deltas.
 * 
 * @param octree        vector containing the sparse voxel octree (uint8)
 * @param depth         The depth of the sparse voxel octree
 * @param delta         vector containing the deltas (uint8)
 * @param num_channels  Number of channels C
 * 
 * @return              [N * C] vector containing the attribute of each sparse voxel
 */
std::vector<uint8_t> decode_sparse_voxel_octree_attr_parent_cpu(
    const std::vector<uint8_t>& octree,
    const uint32_t depth,
    const std::vector<uint8_t>& delta,
    const uint32_t num_channels
);


/**
 * Encode the attribute of a sparse voxel octree into deltas from its neighbors.
 * 
 * @param coord         [N * 3] vector containing the coordinates of each sparse voxel (flattened)
 * @param res           The resolution of the sparse voxel grid
 * @param attr          [N * C] vector containing the attribute of each sparse voxel (flattened)
 * @param num_channels  Number of channels C
 * 
 * @return              vector containing the deltas (uint8)
 */
std::vector<uint8_t> encode_sparse_voxel_octree_attr_neighbor_cpu(
    const std::vector<int32_t>& coord,
    const uint32_t res,
    const std::vector<uint8_t>& attr,
    const uint32_t num_channels
);


/**
 * Decode the attribute of a sparse voxel octree from its neighbors and deltas.
 * 
 * @param coord         [N * 3] vector containing the coordinates of each sparse voxel (flattened)
 * @param res           The resolution of the sparse voxel grid
 * @param delta         [N * C] vector containing the deltas (flattened)
 * @param num_channels  Number of channels C
 * 
 * @return              [N * C] vector containing the attribute of each sparse voxel
 */
std::vector<uint8_t> decode_sparse_voxel_octree_attr_neighbor_cpu(
    const std::vector<int32_t>& coord,
    const uint32_t res,
    const std::vector<uint8_t>& delta,
    const uint32_t num_channels
);
