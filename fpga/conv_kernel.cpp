#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include "conv_kernel.h"

using namespace adf;

// ---- Internal GEMM Engine ----
template <int M_TOTAL, int K_TOTAL, int N_TOTAL>
inline void gemm_int8(const int8_t* __restrict A, const int8_t* __restrict B, int32_t* __restrict C) {
    constexpr int MMUL_M = 4; constexpr int MMUL_K = 8; constexpr int MMUL_N = 4;
    for (int mt = 0; mt < M_TOTAL / MMUL_M; ++mt) {
        for (int nt = 0; nt < N_TOTAL / MMUL_N; ++nt) {
            aie::mmul<MMUL_M, MMUL_K, MMUL_N, int8, int8> mm;
            for (int kt = 0; kt < K_TOTAL / MMUL_K; ++kt) {
                // Use memcpy for fast tile building (avoids alignment issues)
                alignas(32) int8_t a_tile[32];
                alignas(32) int8_t b_tile[32];
                __builtin_memcpy(a_tile, &A[(mt*4)*K_TOTAL + kt*8], 32);
                __builtin_memcpy(b_tile, &B[(kt*8)*N_TOTAL + nt*4], 32);

                aie::vector<int8, 32> va = aie::load_v<32>(a_tile);
                aie::vector<int8, 32> vb = aie::load_v<32>(b_tile);

                if (kt == 0) mm.mul(va, vb);
                else mm.mac(va, vb);
            }
            // Extract results: mmul produces M*N = 16 int32 values
            aie::vector<int32, 16> res = mm.to_vector<int32>();
            alignas(32) int32_t res_buf[16];
            aie::store_v(res_buf, res);
            for (int m = 0; m < 4; ++m)
                for (int n = 0; n < 4; ++n)
                    C[(mt*4+m)*N_TOTAL + nt*4 + n] = res_buf[m * 4 + n];
        }
    }
}

// ---- Requantization Logic ----
template <bool HAS_LEAKY_RELU>
inline void requant_row(const int32_t* acc, const int32_t* bias, const int32_t* mult, const int8_t* shift, int8_t* out) {
    for (int ch = 0; ch < OC_BLOCK; ++ch) {
        int32_t a = acc[ch] + bias[ch];
        int64_t p = (int64_t)a * (int64_t)mult[ch];
        int s = (int)shift[ch];
        int32_t val = (int32_t)((p + ((int64_t)1 << (s - 1))) >> s);
        if constexpr (HAS_LEAKY_RELU) {
            if (val < 0) val = (val * 13) >> 7;
        }
        if (val > 127) val = 127;
        if (val < -128) val = -128;
        out[ch] = (int8_t)val;
    }
}

// ---- Kernels ----

void conv2d_3x3_tile(
    input_buffer<int8, extents<IN_BUF_3x3>>& in_act,
    input_buffer<int8, extents<WT_BUF_3x3>>& in_wt,
    input_buffer<int32, extents<OC_BLOCK>>& in_bias,
    input_buffer<int32, extents<OC_BLOCK>>& in_requant_mult,
    input_buffer<int8, extents<OC_BLOCK>>& in_requant_shift,
    output_buffer<int8, extents<OUT_BUF>>& out_act
) {
    auto p_act = in_act.data(); auto p_wt = in_wt.data(); 
    auto p_bias = in_bias.data(); auto p_mult = in_requant_mult.data();
    auto p_shift = in_requant_shift.data(); auto p_out = out_act.data();

    alignas(32) int8_t col_sub[TILE_W * IC_BLOCK * 9];
    alignas(32) int32_t acc_sub[TILE_W * OC_BLOCK];

    for (int r = 0; r < TILE_H; ++r) {
        // im2col
        for (int w = 0; w < TILE_W; ++w) {
            for (int kh = 0; kh < 3; ++kh) {
                for (int kw = 0; kw < 3; ++kw) {
                    aie::store_v(&col_sub[(w*9 + kh*3 + kw)*32], aie::load_v<32>(&p_act[((r+kh)*IN_W_3x3 + (w+kw))*32]));
                }
            }
        }
        gemm_int8<TILE_W, IC_BLOCK*9, OC_BLOCK>(col_sub, p_wt, acc_sub);
        for (int w = 0; w < TILE_W; ++w) requant_row<true>(&acc_sub[w*32], p_bias, p_mult, p_shift, &p_out[(r*TILE_W+w)*32]);
    }
}

void conv2d_3x3_s2_tile(
    input_buffer<int8, extents<IN_BUF_3x3_S2>>& in_act,
    input_buffer<int8, extents<WT_BUF_3x3>>& in_wt,
    input_buffer<int32, extents<OC_BLOCK>>& in_bias,
    input_buffer<int32, extents<OC_BLOCK>>& in_requant_mult,
    input_buffer<int8, extents<OC_BLOCK>>& in_requant_shift,
    output_buffer<int8, extents<OUT_BUF>>& out_act
) {
    auto p_act = in_act.data(); auto p_wt = in_wt.data();
    auto p_bias = in_bias.data(); auto p_mult = in_requant_mult.data();
    auto p_shift = in_requant_shift.data(); auto p_out = out_act.data();

    alignas(32) int8_t col_sub[TILE_W * IC_BLOCK * 9];
    alignas(32) int32_t acc_sub[TILE_W * OC_BLOCK];

    for (int r = 0; r < TILE_H; ++r) {
        // im2col with stride_w=2: output col w reads input col w*2
        for (int w = 0; w < TILE_W; ++w) {
            for (int kh = 0; kh < 3; ++kh) {
                for (int kw = 0; kw < 3; ++kw) {
                    aie::store_v(&col_sub[(w*9 + kh*3 + kw)*32],
                                 aie::load_v<32>(&p_act[((r+kh)*IN_W_3x3_S2 + (w*2+kw))*32]));
                }
            }
        }
        gemm_int8<TILE_W, IC_BLOCK*9, OC_BLOCK>(col_sub, p_wt, acc_sub);
        for (int w = 0; w < TILE_W; ++w)
            requant_row<true>(&acc_sub[w*32], p_bias, p_mult, p_shift, &p_out[(r*TILE_W+w)*32]);
    }
}

void conv2d_1x1_tile(
    input_buffer<int8, extents<IN_BUF_1x1>>& in_act,
    input_buffer<int8, extents<WT_BUF_1x1>>& in_wt,
    input_buffer<int32, extents<OC_BLOCK>>& in_bias,
    input_buffer<int32, extents<OC_BLOCK>>& in_requant_mult,
    input_buffer<int8, extents<OC_BLOCK>>& in_requant_shift,
    output_buffer<int8, extents<OUT_BUF>>& out_act
) {
    auto p_act = in_act.data(); auto p_wt = in_wt.data();
    auto p_bias = in_bias.data(); auto p_mult = in_requant_mult.data();
    auto p_shift = in_requant_shift.data(); auto p_out = out_act.data();

    alignas(32) int32_t acc_sub[TILE_W * OC_BLOCK];

    for (int r = 0; r < TILE_H; ++r) {
        gemm_int8<TILE_W, IC_BLOCK, OC_BLOCK>(&p_act[r*TILE_W*32], p_wt, acc_sub);
        for (int w = 0; w < TILE_W; ++w) requant_row<true>(&acc_sub[w*32], p_bias, p_mult, p_shift, &p_out[(r*TILE_W+w)*32]);
    }
}

void elem_add_tile(
    input_buffer<int8, extents<OUT_BUF>>& in_a,
    input_buffer<int8, extents<OUT_BUF>>& in_b,
    output_buffer<int8, extents<OUT_BUF>>& out_sum
) {
    auto p_a = in_a.data(); auto p_b = in_b.data(); auto p_c = out_sum.data();
    for (int i = 0; i < OUT_BUF/32; ++i) {
        aie::vector<int16, 32> sum = aie::add(aie::unpack(aie::load_v<32>(&p_a[i*32])), aie::unpack(aie::load_v<32>(&p_b[i*32])));
        aie::store_v(&p_c[i*32], aie::pack(sum));
    }
}