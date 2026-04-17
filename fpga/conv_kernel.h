#ifndef CONV2D_KERNEL_H
#define CONV2D_KERNEL_H

#include <adf.h>
#include <cstdint>

using namespace adf;

// ---- Tile geometry ----
constexpr int TILE_H    = 8;
constexpr int TILE_W    = 32;
constexpr int IC_BLOCK  = 32;
constexpr int OC_BLOCK  = 32;

constexpr int IN_H_3x3  = TILE_H + 2;   
constexpr int IN_W_3x3  = TILE_W + 2;   

// ---- Buffer sizes (Total Elements) ----
constexpr int IN_BUF_3x3  = IN_H_3x3 * IN_W_3x3 * IC_BLOCK;   // 10,880
constexpr int WT_BUF_3x3  = IC_BLOCK * 9 * OC_BLOCK;          //  9,216
constexpr int OUT_BUF     = TILE_H * TILE_W * OC_BLOCK;       //  8,192

constexpr int IN_BUF_1x1  = TILE_H * TILE_W * IC_BLOCK;       //  8,192
constexpr int WT_BUF_1x1  = IC_BLOCK * OC_BLOCK;              //  1,024

// ---- Kernel function declarations (Using Buffer API) ----

void conv2d_3x3_tile(
    input_buffer<int8, extents<IN_BUF_3x3>>&    in_act,
    input_buffer<int8, extents<WT_BUF_3x3>>&    in_wt,
    input_buffer<int32, extents<OC_BLOCK>>&     in_bias,
    input_buffer<int32, extents<OC_BLOCK>>&     in_requant_mult,
    input_buffer<int8, extents<OC_BLOCK>>&      in_requant_shift,
    output_buffer<int8, extents<OUT_BUF>>&      out_act
);

void conv2d_1x1_tile(
    input_buffer<int8, extents<IN_BUF_1x1>>&    in_act,
    input_buffer<int8, extents<WT_BUF_1x1>>&    in_wt,
    input_buffer<int32, extents<OC_BLOCK>>&     in_bias,
    input_buffer<int32, extents<OC_BLOCK>>&     in_requant_mult,
    input_buffer<int8, extents<OC_BLOCK>>&      in_requant_shift,
    output_buffer<int8, extents<OUT_BUF>>&      out_act
);

void elem_add_tile(
    input_buffer<int8, extents<OUT_BUF>>&   in_a,
    input_buffer<int8, extents<OUT_BUF>>&   in_b,
    output_buffer<int8, extents<OUT_BUF>>&  out_sum
);

#endif