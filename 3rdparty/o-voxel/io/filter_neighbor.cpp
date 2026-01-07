#include "api.h"

#include <cstdint>
#include <cmath>
#include <vector>


/**
 * Encode the attribute of a sparse voxel octree into deltas from its neighbors.
 * 
 * @param coord    [N * 3] vector containing the coordinates of each sparse voxel
 * @param res      The resolution of the sparse voxel grid
 * @param attr     [N * C] vector containing the attribute of each sparse voxel
 * @param num_channels Number of channels C
 * 
 * @return         vector containing the deltas
 */
std::vector<uint8_t> encode_sparse_voxel_octree_attr_neighbor_cpu(
    const std::vector<int32_t>& coord,
    const uint32_t res,
    const std::vector<uint8_t>& attr,
    const uint32_t num_channels
) {
    size_t N = coord.size() / 3;
    size_t C = num_channels;
    const int32_t* coord_data = coord.data();
    const uint8_t* attr_data = attr.data();
    std::vector<uint8_t> buffer(res * res * res * (C + 1), 0);
    
    // Densify the coordinates
    for (size_t i = 0; i < N; i++) {
        int x = coord_data[i * 3 + 0];
        int y = coord_data[i * 3 + 1];
        int z = coord_data[i * 3 + 2];
        size_t ptr = (z * res * res + y * res + x) * (C + 1);
        buffer[ptr + C] = 1;
        for (size_t c = 0; c < C; c++) {
            buffer[ptr + c] = attr_data[i * C + c];
        }
    }
    
    // Compute the deltas
    for (int z = (int)res-1; z >= 0; z--) {
        for (int y = (int)res-1; y >= 0; y--) {
            for (int x = (int)res-1; x >= 0; x--) {
                size_t ptr = (z * res * res + y * res + x) * (C + 1);
                int neignbor_ptr = -1;
                size_t tmp_ptr;
                if (!buffer[ptr + C]) continue;
                // x
                tmp_ptr = (z * res * res + y * res + (x - 1)) * (C + 1);
                if (x > 0 && buffer[tmp_ptr + C]) neignbor_ptr = (int)tmp_ptr;
                // y
                tmp_ptr = (z * res * res + (y - 1) * res + x) * (C + 1);
                if (y > 0 && buffer[tmp_ptr + C]) neignbor_ptr = (int)tmp_ptr;
                // z
                tmp_ptr = ((z - 1) * res * res + y * res + x) * (C + 1);
                if (z > 0 && buffer[tmp_ptr + C]) neignbor_ptr = (int)tmp_ptr;
                // xy
                tmp_ptr = (z * res * res + (y - 1) * res + (x - 1)) * (C + 1);
                if (y > 0 && x > 0 && buffer[tmp_ptr + C]) neignbor_ptr = (int)tmp_ptr;
                // xz
                tmp_ptr = ((z - 1) * res * res + y * res + (x - 1)) * (C + 1);
                if (z > 0 && x > 0 && buffer[tmp_ptr + C]) neignbor_ptr = (int)tmp_ptr;
                // yz
                tmp_ptr = ((z - 1) * res * res + (y - 1) * res + x) * (C + 1);
                if (z > 0 && y > 0 && buffer[tmp_ptr + C]) neignbor_ptr = (int)tmp_ptr;
                // xyz
                tmp_ptr = ((z - 1) * res * res + (y - 1) * res + (x - 1)) * (C + 1);
                if (z > 0 && y > 0 && x > 0 && buffer[tmp_ptr + C]) neignbor_ptr = (int)tmp_ptr;
                if (neignbor_ptr >= 0) {
                    for (size_t c = 0; c < C; c++) {
                        buffer[ptr + c] -= buffer[neignbor_ptr + c];
                    }
                }
            }
        }
    }

    // Pack the deltas into a uint8 vector
    std::vector<uint8_t> delta(N * C, 0);
    uint8_t* delta_data = delta.data();
    for (size_t i = 0; i < N; i++) {
        int x = coord_data[i * 3 + 0];
        int y = coord_data[i * 3 + 1];
        int z = coord_data[i * 3 + 2];
        size_t ptr = (z * res * res + y * res + x) * (C + 1);
        for (size_t c = 0; c < C; c++) {
            delta_data[i * C + c] = buffer[ptr + c];
        }
    }
    return delta;
}


/**
 * Decode the attribute of a sparse voxel octree from its neighbors and deltas.
 * 
 * @param coord    [N * 3] vector containing the coordinates of each sparse voxel
 * @param res      The resolution of the sparse voxel grid
 * @param delta    [N * C] vector containing the deltas
 * @param num_channels Number of channels C
 * 
 * @return         [N * C] vector containing the attribute of each sparse voxel
 */
std::vector<uint8_t> decode_sparse_voxel_octree_attr_neighbor_cpu(
    const std::vector<int32_t>& coord,
    const uint32_t res,
    const std::vector<uint8_t>& delta,
    const uint32_t num_channels
) {
    size_t N = coord.size() / 3;
    size_t C = num_channels;
    const int32_t* coord_data = coord.data();
    const uint8_t* delta_data = delta.data();
    std::vector<uint8_t> buffer(res * res * res * (C + 1), 0);
    
    // Densify the coordinates
    for (size_t i = 0; i < N; i++) {
        int x = coord_data[i * 3 + 0];
        int y = coord_data[i * 3 + 1];
        int z = coord_data[i * 3 + 2];
        size_t ptr = (z * res * res + y * res + x) * (C + 1);
        buffer[ptr + C] = 1;
        for (size_t c = 0; c < C; c++) {
            buffer[ptr + c] = delta_data[i * C + c];
        }
    }
    
    // Reconstruct the attribute
    for (int z = 0; z < (int)res; z++) {
        for (int y = 0; y < (int)res; y++) {
            for (int x = 0; x < (int)res; x++) {
                size_t ptr = (z * res * res + y * res + x) * (C + 1);
                int neignbor_ptr = -1;
                size_t tmp_ptr;
                if (!buffer[ptr + C]) continue;
                // x
                tmp_ptr = (z * res * res + y * res + (x - 1)) * (C + 1);
                if (x > 0 && buffer[tmp_ptr + C]) neignbor_ptr = (int)tmp_ptr;
                // y
                tmp_ptr = (z * res * res + (y - 1) * res + x) * (C + 1);
                if (y > 0 && buffer[tmp_ptr + C]) neignbor_ptr = (int)tmp_ptr;
                // z
                tmp_ptr = ((z - 1) * res * res + y * res + x) * (C + 1);
                if (z > 0 && buffer[tmp_ptr + C]) neignbor_ptr = (int)tmp_ptr;
                // xy
                tmp_ptr = (z * res * res + (y - 1) * res + (x - 1)) * (C + 1);
                if (y > 0 && x > 0 && buffer[tmp_ptr + C]) neignbor_ptr = (int)tmp_ptr;
                // xz
                tmp_ptr = ((z - 1) * res * res + y * res + (x - 1)) * (C + 1);
                if (z > 0 && x > 0 && buffer[tmp_ptr + C]) neignbor_ptr = (int)tmp_ptr;
                // yz
                tmp_ptr = ((z - 1) * res * res + (y - 1) * res + x) * (C + 1);
                if (z > 0 && y > 0 && buffer[tmp_ptr + C]) neignbor_ptr = (int)tmp_ptr;
                // xyz
                tmp_ptr = ((z - 1) * res * res + (y - 1) * res + (x - 1)) * (C + 1);
                if (z > 0 && y > 0 && x > 0 && buffer[tmp_ptr + C]) neignbor_ptr = (int)tmp_ptr;
                if (neignbor_ptr >= 0) {
                    for (size_t c = 0; c < C; c++) {
                        buffer[ptr + c] += buffer[neignbor_ptr + c];
                    }
                }
            }
        }
    }

    // Pack the attribute into a uint8 vector
    std::vector<uint8_t> attr(N * C, 0);
    uint8_t* attr_data = attr.data();
    for (size_t i = 0; i < N; i++) {
        int x = coord_data[i * 3 + 0];
        int y = coord_data[i * 3 + 1];
        int z = coord_data[i * 3 + 2];
        size_t ptr = (z * res * res + y * res + x) * (C + 1);
        for (size_t c = 0; c < C; c++) {
            attr_data[i * C + c] = buffer[ptr + c];
        }
    }
    return attr;
}
