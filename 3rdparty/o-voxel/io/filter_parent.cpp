#include "api.h"
#include "lut.h"

#include <cstdint>
#include <cmath>
#include <vector>


std::vector<uint8_t> encode_recursive(
    const uint8_t* svo,
    const uint32_t depth,
    const uint8_t* attr,
    const size_t C,
    uint32_t& svo_ptr,
    uint32_t& attr_ptr,
    uint32_t& delta_ptr,
    uint32_t self_delta_ptr,
    uint32_t cur_depth,
    uint8_t* delta
) {
    std::vector<uint8_t> node_attr(C, 0);
    if (cur_depth == depth) {
        // Leaf node
        for (size_t i = 0; i < C; i++) {
            node_attr[i] = attr[attr_ptr + i];
            if (self_delta_ptr != 0 || cur_depth == 0) {
                delta[self_delta_ptr + i] = node_attr[i];
            }
        }
        attr_ptr += C;
    }
    else {
        // Internal node
        uint8_t node = svo[svo_ptr];
        uint32_t child_delta_ptr = delta_ptr;
        uint8_t cnt = lut_1cnt[node];
        svo_ptr++;
        delta_ptr += (uint32_t)(C * (cnt - 1));
        for (uint8_t i = 0; i < cnt; i++) {
            auto child_attr = encode_recursive(
                svo, depth, attr, C, svo_ptr, attr_ptr, delta_ptr, i == cnt-1 ? 0 : child_delta_ptr+(uint32_t)(i*C), cur_depth+1, delta
            );
            for (size_t j = 0; j < C; j++) {
                if (i == 0) {
                    node_attr[j] = child_attr[j];
                }
                else {
                    delta[child_delta_ptr + (i-1)*C + j] = child_attr[j] - delta[child_delta_ptr + (i-1)*C + j];
                }
            }
        }
        if (self_delta_ptr != 0 || cur_depth == 0) {
            for (size_t i = 0; i < C; i++) {
                delta[self_delta_ptr + i] = node_attr[i];
            }
        }
    }
    return node_attr;
}


/**
 * Encode the attribute of a sparse voxel octree into deltas from its parent node.
 * 
 * @param octree   vector containing the sparse voxel octree
 * @param depth    The depth of the sparse voxel octree
 * @param attr     [N * C] vector containing the attribute of each sparse voxel
 * @param num_channels Number of channels C
 * 
 * @return         vector containing the deltas
 */
std::vector<uint8_t> encode_sparse_voxel_octree_attr_parent_cpu(
    const std::vector<uint8_t>& octree,
    const uint32_t depth,
    const std::vector<uint8_t>& attr,
    const uint32_t num_channels
) {
    size_t C = num_channels;
    size_t N_leaf = attr.size() / C;
    // size_t N_node = octree.size(); // Unused
    const uint8_t* octree_data = octree.data();
    const uint8_t* attr_data = attr.data();

    std::vector<uint8_t> delta(N_leaf * C, 0);
    uint32_t svo_ptr = 0;
    uint32_t attr_ptr = 0;
    uint32_t delta_ptr = (uint32_t)C;
    encode_recursive(octree_data, depth, attr_data, C, svo_ptr, attr_ptr, delta_ptr, 0, 0, delta.data());

    return delta;
}


void decode_recursive(
    const uint8_t* svo,
    const uint32_t depth,
    const uint8_t* delta,
    const size_t C,
    uint32_t& svo_ptr,
    uint32_t& attr_ptr,
    uint32_t& delta_ptr,
    uint32_t cur_depth,
    uint8_t* cur_attr,
    uint8_t* attr
) {
    if (cur_depth == depth) {
        // Leaf node
        for (size_t i = 0; i < C; i++) {
            attr[attr_ptr + i] = cur_attr[i];
        }
        attr_ptr += C;
    }
    else {
        // Internal node
        uint8_t node = svo[svo_ptr];
        uint32_t child_delta_ptr = delta_ptr;
        std::vector<uint8_t> child_attr(cur_attr, cur_attr + C);
        uint8_t cnt = lut_1cnt[node];
        svo_ptr++;
        delta_ptr += (uint32_t)(C * (cnt - 1));
        for (uint8_t i = 0; i < cnt; i++) {
            for (size_t j = 0; j < C; j++) {
                if (i > 0) {
                    child_attr[j] += delta[child_delta_ptr + (i-1)*C + j];
                }
            }
            decode_recursive(
                svo, depth, delta, C, svo_ptr, attr_ptr, delta_ptr, cur_depth+1, child_attr.data(), attr
            );
        }
    }
}


/**
 * Decode the attribute of a sparse voxel octree from its parent node and its deltas.
 * 
 * @param octree   vector containing the sparse voxel octree
 * @param depth    The depth of the sparse voxel octree
 * @param delta    vector containing the deltas
 * @param num_channels Number of channels C
 * 
 * @return         [N * C] vector containing the attribute of each sparse voxel
 */
std::vector<uint8_t> decode_sparse_voxel_octree_attr_parent_cpu(
    const std::vector<uint8_t>& octree,
    const uint32_t depth,
    const std::vector<uint8_t>& delta,
    const uint32_t num_channels
) {
    // size_t N_node = octree.size(); // Unused
    size_t C = num_channels;
    size_t N_leaf = delta.size() / C;
    const uint8_t* octree_data = octree.data();
    const uint8_t* delta_data = delta.data();

    std::vector<uint8_t> attr(N_leaf * C, 0);
    uint32_t svo_ptr = 0;
    uint32_t attr_ptr = 0;
    uint32_t delta_ptr = (uint32_t)C;

    // Recursively decode the attribute
    // Note: In decode_recursive, the 9th argument is `uint8_t* cur_attr`. 
    // In original code, it was passed `delta_data` as initial `cur_attr`.
    // However, `cur_attr` is modified in `child_attr` vector which copies from it.
    // The initial call passes `delta_data` which seems to be the root attribute (as delta[0..C] is root attr?)
    // Yes, see encode_recursive: if self_delta_ptr == 0 (root), it writes node_attr to delta[0..C].
    // So delta[0..C] holds the root attribute values.
    decode_recursive(
        octree_data, depth, delta_data, C, svo_ptr, attr_ptr, delta_ptr, 0, (uint8_t*)delta_data, attr.data()
    );

    return attr;
}
