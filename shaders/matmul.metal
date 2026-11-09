#include <metal_stdlib>
using namespace metal;

kernel void matmul(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& dimension [[buffer(3)]],
    uint2 position [[thread_position_in_grid]]
) {
    const uint row = position.y;
    const uint column = position.x;

    if (row >= dimension || column >= dimension) {
        return;
    }

    float sum = 0.0f;
    for (uint k = 0; k < dimension; ++k) {
        sum += a[row * dimension + k] * b[k * dimension + column];
    }

    c[row * dimension + column] = sum;
}

// Every 16 x 16 group cooperatively loads two tiles, reusing each loaded
// value across 16 dot products. Out-of-range lanes still reach both barriers.
kernel void matmul_tiled(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& dimension [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 local [[thread_position_in_threadgroup]]
) {
    constexpr uint tile_size = 16;
    threadgroup float tile_a[tile_size][tile_size];
    threadgroup float tile_b[tile_size][tile_size];
    const uint row = group.y * tile_size + local.y;
    const uint column = group.x * tile_size + local.x;
    float sum = 0.0f;

    for (uint base = 0; base < dimension; base += tile_size) {
        const uint a_column = base + local.x;
        const uint b_row = base + local.y;
        tile_a[local.y][local.x] =
            row < dimension && a_column < dimension ? a[row * dimension + a_column] : 0.0f;
        tile_b[local.y][local.x] =
            b_row < dimension && column < dimension ? b[b_row * dimension + column] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < tile_size; ++k) {
            sum += tile_a[local.y][k] * tile_b[k][local.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < dimension && column < dimension) {
        c[row * dimension + column] = sum;
    }
}
