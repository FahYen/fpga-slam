// =============================================================================
// conv_kernel.cpp / conv2d_core.h — AIE-ML INT8 convolution kernels
//
// Target: AMD Versal AI Edge VE2802 (AIE2P / AIE-ML)
// Uses:   aie::mmul<4,8,4, int8, int8> for hardware-optimal INT8 MAC
//         → 4×8 activation tile × 8×4 weight tile per mmul instruction
//         → 128 INT8 MACs per mmul call, maps to the hardware MAC cascade
//
// Convolution is lowered to matrix multiplication (im2col style):
//   - For 3×3 conv: activation is reshaped to [spatial × IC*9]
//     and weights to [IC*9 × OC], then GEMM produces [spatial × OC]
//   - For 1×1 conv: activation is [spatial × IC], weights [IC × OC]
//
// Data layout conventions (must match weight reordering on host):
//   - Activations:  [H][W][C]  (HWC / channel-last)
//   - Weights 3×3:  [IC*kH*kW][OC] (transposed for GEMM B matrix)
//   - Weights 1×1:  [IC][OC]       (transposed for GEMM B matrix)
//   - Output:       [H][W][C]  (HWC / channel-last)
//
// Requantization (per-channel, matches precompute_requant.py):
//   output = saturate( (acc_i32 * int_mult[oc] + round) >> shift[oc] )
//   int_mult and shift are computed on the host from weight_scale × act_scale.
//
// Activation:
//   LeakyReLU with alpha ≈ 0.1, approximated as (x * 13) >> 7 = x * 0.1016
//   Controlled by template parameter — not all layers use it.
//
// =============================================================================

#ifndef CONV2D_CORE_H
#define CONV2D_CORE_H

#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <adf.h>
#include <cstdint>

using namespace adf; // Add namespace here as well to match signatures

// ---- Tile geometry (tuned for AIE-ML 64 KB data memory) -------------------
//
// Memory budget per tile (worst case, 3×3 conv):
//   Input:   (8+2) × (8+2) × 32 =  3,200 bytes   (reduced TILE_W for scratch)
//   Weight:  288 × 32            =  9,216 bytes
//   Bias:    32 × 4              =    128 bytes
//   RQ mult: 32 × 4              =    128 bytes
//   RQ shift:32 × 1              =     32 bytes
//   Output:  8 × 8 × 32         =  2,048 bytes
//   Scratch col_sub: 2×8×288    =  4,608 bytes  (SUB_ROWS=2)
//   Scratch acc_sub: 2×8×32×4   =  2,048 bytes
//   Total:                        ~21 KB → comfortable fit with double-buffering
//
// For wider tiles (TILE_W=32):
//   Input:   10 × 34 × 32        = 10,880 bytes
//   col_sub: 1×32×288             =  9,216 bytes  (SUB_ROWS=1)
//   acc_sub: 1×32×32×4            =  4,096 bytes
//   + weight + output + requant   ~42 KB total → fits in 64 KB

constexpr int TILE_H    = 8;     // output spatial height per tile
constexpr int TILE_W    = 32;    // output spatial width per tile
constexpr int IC_BLOCK  = 32;    // input channels processed per kernel call
constexpr int OC_BLOCK  = 32;    // output channels produced per kernel call

// For 3×3 conv, input tile includes halo of 1 on each side
constexpr int IN_H_3x3  = TILE_H + 2;   // 10
constexpr int IN_W_3x3  = TILE_W + 2;   // 34

// aie::mmul tile size for int8: M=4, K=8, N=4
constexpr int MMUL_M = 4;
constexpr int MMUL_K = 8;
constexpr int MMUL_N = 4;

// ---- Buffer sizes ---------------------------------------------------------

constexpr int IN_BUF_3x3  = IN_H_3x3 * IN_W_3x3 * IC_BLOCK;   // 10,880
constexpr int WT_BUF_3x3  = IC_BLOCK * 9 * OC_BLOCK;            //  9,216
constexpr int OUT_BUF      = TILE_H * TILE_W * OC_BLOCK;         //  8,192

constexpr int IN_BUF_1x1  = TILE_H * TILE_W * IC_BLOCK;         //  8,192
constexpr int WT_BUF_1x1  = IC_BLOCK * OC_BLOCK;                 //  1,024

// Sub-row processing: 1 row at a time keeps scratch small
constexpr int SUB_ROWS    = 1;
constexpr int SUB_SPATIAL = SUB_ROWS * TILE_W;                    // 32


// =============================================================================
// gemm_int8: Blocked INT8 GEMM using aie::mmul<4,8,4>
//
// C[M × N] += A[M × K] × B[K × N]    (all row-major)
//
// The A sub-tile (4×8 = 32 int8) maps naturally to one vector register.
// The B sub-tile (8×4 = 32 int8) likewise fits one register.
// The mmul accumulator holds 4×4 = 16 int32 results.
// =============================================================================

template <int M_TOTAL, int K_TOTAL, int N_TOTAL>
inline void gemm_int8(
    const int8_t* __restrict A,   // [M_TOTAL × K_TOTAL]
    const int8_t* __restrict B,   // [K_TOTAL × N_TOTAL]
    int32_t* __restrict C         // [M_TOTAL × N_TOTAL]
) {
    static_assert(M_TOTAL % MMUL_M == 0, "M must be divisible by 4");
    static_assert(K_TOTAL % MMUL_K == 0, "K must be divisible by 8");
    static_assert(N_TOTAL % MMUL_N == 0, "N must be divisible by 4");

    constexpr int M_TILES = M_TOTAL / MMUL_M;
    constexpr int K_TILES = K_TOTAL / MMUL_K;
    constexpr int N_TILES = N_TOTAL / MMUL_N;

    for (int mt = 0; mt < M_TILES; ++mt) {
        for (int nt = 0; nt < N_TILES; ++nt) {

            // Default-construct zeros the accumulator
            aie::mmul<MMUL_M, MMUL_K, MMUL_N, int8, int8> mm;

            for (int kt = 0; kt < K_TILES; ++kt) {

                // Load A sub-tile [4 rows × 8 cols] — 32 bytes, one register
                aie::vector<int8, MMUL_M * MMUL_K> va;
                for (int m = 0; m < MMUL_M; ++m) {
                    int row = mt * MMUL_M + m;
                    int col = kt * MMUL_K;
                    aie::vector<int8, MMUL_K> rv =
                        aie::load_v<MMUL_K>(&A[row * K_TOTAL + col]);
                    va.insert(m, rv);
                }

                // Load B sub-tile [8 rows × 4 cols] — 32 bytes, one register
                aie::vector<int8, MMUL_K * MMUL_N> vb;
                for (int k = 0; k < MMUL_K; ++k) {
                    int row = kt * MMUL_K + k;
                    int col = nt * MMUL_N;
                    aie::vector<int8, MMUL_N> rv =
                        aie::load_v<MMUL_N>(&B[row * N_TOTAL + col]);
                    vb.insert(k, rv);
                }

                // Accumulate: first iteration mul, rest mac
                if (kt == 0) {
                    mm.mul(va, vb);
                } else {
                    mm.mac(va, vb);
                }
            }

            // Extract 4×4 = 16 int32 results and scatter to C
            aie::vector<int32, MMUL_M * MMUL_N> result = mm.to_vector<int32>();
            for (int m = 0; m < MMUL_M; ++m) {
                int row = mt * MMUL_M + m;
                int col = nt * MMUL_N;
                aie::vector<int32, MMUL_N> slice = result.extract<MMUL_N>(m);
                aie::store_v(&C[row * N_TOTAL + col], slice);
            }
        }
    }
}


// =============================================================================
// requant_bias_lrelu: Per-channel requantize, add bias, optional LeakyReLU,
//                     and saturate to INT8.
//
// For a row of OC_BLOCK (32) accumulator values:
//   1. acc += bias[oc]
//   2. scaled = (acc * mult[oc] + (1 << (shift[oc]-1))) >> shift[oc]
//   3. if (has_leaky_relu && scaled < 0): scaled = (scaled * 13) >> 7
//   4. out = saturate_i8(scaled)
//
// Processes 16 channels at a time (native int32 vector width on AIE-ML =
// 512 bits / 32 bits = 16 lanes).
// =============================================================================

template <bool HAS_LEAKY_RELU>
inline void requant_row(
    const int32_t* __restrict acc,       // [OC_BLOCK] accumulators for one position
    const int32_t* __restrict bias,      // [OC_BLOCK]
    const int32_t* __restrict mult,      // [OC_BLOCK]
    const int8_t* __restrict shift,     // [OC_BLOCK]
    int8_t* __restrict out        // [OC_BLOCK]
) {
    constexpr int HALF = 16;  // int32 vector width on AIE-ML

    // Process lower 16 channels, then upper 16 channels
    for (int half = 0; half < 2; ++half) {
        int off = half * HALF;

        aie::vector<int32, HALF> v_acc  = aie::load_v<HALF>(&acc[off]);
        aie::vector<int32, HALF> v_bias = aie::load_v<HALF>(&bias[off]);
        aie::vector<int32, HALF> v_mult = aie::load_v<HALF>(&mult[off]);

        // 1. Add bias
        v_acc = aie::add(v_acc, v_bias);

        // 2. Per-channel requantize: (acc * mult + round) >> shift
        //    Process element-by-element because shifts differ per channel
        aie::vector<int32, HALF> v_out;
        for (int i = 0; i < HALF; ++i) {
            int64_t prod = (int64_t)v_acc[i] * (int64_t)v_mult[i];
            int s = (int)shift[off + i];
            int64_t rounded = prod + ((int64_t)1 << (s - 1));
            int32_t val = (int32_t)(rounded >> s);
            v_out[i] = val;
        }

        // 3. LeakyReLU (conditional)
        if constexpr (HAS_LEAKY_RELU) {
            aie::vector<int32, HALF> v_zero = aie::zeros<int32, HALF>();
            aie::mask<HALF> neg = aie::lt(v_out, v_zero);

            // Leaky: (x * 13) >> 7 for negatives
            aie::vector<int32, HALF> v_13 = aie::broadcast<int32, HALF>(13);
            aie::vector<int32, HALF> v_leaky;
            for (int i = 0; i < HALF; ++i) {
                v_leaky[i] = (v_out[i] * 13) >> 7;
            }
            v_out = aie::select(v_out, v_leaky, neg);
        }

        // 4. Saturate and store
        for (int i = 0; i < HALF; ++i) {
            int32_t val = v_out[i];
            if (val > 127)  val = 127;
            if (val < -128) val = -128;
            out[off + i] = (int8_t)val;
        }
    }
}


// =============================================================================
// conv2d_3x3_tile: 3×3 convolution on one spatial tile
//
// Buffers:
//   in_act:         [IN_H_3x3 × IN_W_3x3 × IC_BLOCK]  INT8 activations w/ halo
//   in_wt:          [IC_BLOCK*9 × OC_BLOCK]             INT8 weights (GEMM B layout)
//   in_bias:        [OC_BLOCK]                           INT32 bias
//   in_requant_mult:[OC_BLOCK]                           INT32 per-channel multiplier
//   in_requant_shift:[OC_BLOCK]                          INT8 per-channel shift
//   out_act:        [TILE_H × TILE_W × OC_BLOCK]        INT8 output
//
// Flow per sub-row:
//   1. im2col: extract 3×3×IC patches → A matrix [SUB_SPATIAL × IC*9]
//   2. GEMM:   A × B → C in INT32
//   3. Per-channel requant + bias + LeakyReLU + saturate → INT8
// =============================================================================

void conv2d_3x3_tile(
    input_buffer<int8, extents<IN_BUF_3x3>>&    in_act,
    input_buffer<int8, extents<WT_BUF_3x3>>&    in_wt,
    input_buffer<int32, extents<OC_BLOCK>>&     in_bias,
    input_buffer<int32, extents<OC_BLOCK>>&     in_requant_mult,
    input_buffer<int8, extents<OC_BLOCK>>&      in_requant_shift,
    output_buffer<int8, extents<OUT_BUF>>&      out_act
) {
    const int8_t* __restrict act   = (const int8_t*)in_act.data();
    const int8_t* __restrict wt    = (const int8_t*)in_wt.data();
    const int32_t* __restrict bias  = (const int32_t*)in_bias.data();
    const int32_t* __restrict rq_m  = (const int32_t*)in_requant_mult.data();
    const int8_t* __restrict rq_s  = (const int8_t*)in_requant_shift.data();
    int8_t* __restrict out   = (int8_t*)out_act.data();

    constexpr int K_DIM = IC_BLOCK * 9;    // 288
    constexpr int N_DIM = OC_BLOCK;         // 32

    // Scratch: im2col sub-tile + int32 accumulator (per SUB_ROWS)
    // col_sub: SUB_SPATIAL × K_DIM = 32 × 288 = 9,216 bytes
    // acc_sub: SUB_SPATIAL × N_DIM = 32 × 32  = 4,096 bytes (int32)
    alignas(32) int8_t  col_sub[SUB_SPATIAL * K_DIM];
    alignas(32) int32_t acc_sub[SUB_SPATIAL * N_DIM];

    for (int row_start = 0; row_start < TILE_H; row_start += SUB_ROWS) {

        // ---- im2col: extract 3×3×IC patches for this sub-row ----
        int col_idx = 0;
        for (int oh = row_start; oh < row_start + SUB_ROWS; ++oh) {
            for (int ow = 0; ow < TILE_W; ++ow) {
                for (int kh = 0; kh < 3; ++kh) {
                    for (int kw = 0; kw < 3; ++kw) {
                        int ih = oh + kh;
                        int iw = ow + kw;
                        const int8_t* src =
                            &act[(ih * IN_W_3x3 + iw) * IC_BLOCK];
                        aie::vector<int8, 32> v = aie::load_v<32>(src);
                        aie::store_v(&col_sub[col_idx], v);
                        col_idx += IC_BLOCK;
                    }
                }
            }
        }

        // ---- GEMM: col_sub × wt → acc_sub (INT32) ----
        gemm_int8<SUB_SPATIAL, K_DIM, N_DIM>(col_sub, wt, acc_sub);

        // ---- Per-channel requant + bias + LeakyReLU → INT8 ----
        int out_offset = row_start * TILE_W * OC_BLOCK;
        for (int pos = 0; pos < SUB_SPATIAL; ++pos) {
            requant_row<true>(
                &acc_sub[pos * N_DIM],
                bias, rq_m, rq_s,
                &out[out_offset + pos * OC_BLOCK]
            );
        }
    }
}


// =============================================================================
// conv2d_1x1_tile: 1×1 pointwise convolution on one spatial tile
//
// No im2col needed — activations in HWC layout are already the A matrix.
//   A[TILE_H*TILE_W × IC_BLOCK] × B[IC_BLOCK × OC_BLOCK] → C[spatial × OC]
// =============================================================================

void conv2d_1x1_tile(
    input_buffer<int8, extents<IN_BUF_1x1>>&    in_act,
    input_buffer<int8, extents<WT_BUF_1x1>>&    in_wt,
    input_buffer<int32, extents<OC_BLOCK>>&     in_bias,
    input_buffer<int32, extents<OC_BLOCK>>&     in_requant_mult,
    input_buffer<int8, extents<OC_BLOCK>>&      in_requant_shift,
    output_buffer<int8, extents<OUT_BUF>>&      out_act
) {
    const int8_t* __restrict act   = (const int8_t*)in_act.data();
    const int8_t* __restrict wt    = (const int8_t*)in_wt.data();
    const int32_t* __restrict bias  = (const int32_t*)in_bias.data();
    const int32_t* __restrict rq_m  = (const int32_t*)in_requant_mult.data();
    const int8_t* __restrict rq_s  = (const int8_t*)in_requant_shift.data();
    int8_t* __restrict out   = (int8_t*)out_act.data();

    constexpr int K_DIM = IC_BLOCK;   // 32
    constexpr int N_DIM = OC_BLOCK;   // 32

    // acc_sub: SUB_SPATIAL × N_DIM = 32 × 32 = 4,096 bytes
    alignas(32) int32_t acc_sub[SUB_SPATIAL * N_DIM];

    for (int row_start = 0; row_start < TILE_H; row_start += SUB_ROWS) {

        int act_offset = row_start * TILE_W * IC_BLOCK;

        // GEMM: act_sub × wt → acc_sub
        gemm_int8<SUB_SPATIAL, K_DIM, N_DIM>(
            &act[act_offset], wt, acc_sub
        );

        // Per-channel requant + bias + LeakyReLU → INT8
        int out_offset = row_start * TILE_W * OC_BLOCK;
        for (int pos = 0; pos < SUB_SPATIAL; ++pos) {
            requant_row<true>(
                &acc_sub[pos * N_DIM],
                bias, rq_m, rq_s,
                &out[out_offset + pos * OC_BLOCK]
            );
        }
    }
}


// =============================================================================
// elem_add_tile: Saturating element-wise INT8 addition for residual connections
//
// out[i] = saturate_i8( (int16_t)a[i] + (int16_t)b[i] )
// =============================================================================

void elem_add_tile(
    input_buffer<int8, extents<OUT_BUF>>&   in_a,
    input_buffer<int8, extents<OUT_BUF>>&   in_b,
    output_buffer<int8, extents<OUT_BUF>>&  out_sum
) {
    const int8_t* __restrict a = (const int8_t*)in_a.data();
    const int8_t* __restrict b = (const int8_t*)in_b.data();
    int8_t* __restrict c = (int8_t*)out_sum.data();

    constexpr int VEC_LEN  = 32;
    constexpr int NUM_VECS = OUT_BUF / VEC_LEN;  // 256

    for (int i = 0; i < NUM_VECS; ++i) {
        aie::vector<int8, VEC_LEN> va = aie::load_v<VEC_LEN>(&a[i * VEC_LEN]);
        aie::vector<int8, VEC_LEN> vb = aie::load_v<VEC_LEN>(&b[i * VEC_LEN]);

        // Widen to int16, add, saturate back to int8
        aie::vector<int16, VEC_LEN> wa = aie::unpack(va);
        aie::vector<int16, VEC_LEN> wb = aie::unpack(vb);
        aie::vector<int16, VEC_LEN> ws = aie::add(wa, wb);
        aie::vector<int8, VEC_LEN> vc  = aie::pack(ws);

        aie::store_v(&c[i * VEC_LEN], vc);
    }
}


#endif // CONV2D_CORE_H