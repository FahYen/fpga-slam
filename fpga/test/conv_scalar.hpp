// Scalar C++ implementation of the INT8 conv / conv_transpose kernel.
// Mirrors the math in conv_kernel.cpp exactly (plus rounding in the shift)
// but without any AIE APIs so it compiles with plain g++ for unit testing.
//
// Data layout: NCHW (matches the .bin files from generate_golden_int8.py).
// The AIE kernel's H,W,C layout is a re-arrangement done by the data mover
// later — not tested at this level.
#pragma once

#include <cstdint>
#include <cstdlib>

namespace scalar_kernel {

// Per-channel requantize: (acc * mult + round) >> shift
// Uses round-to-nearest-up; matches the Python golden generator.
inline int32_t requantize_one(int32_t acc_with_bias, int32_t mult, int shift) {
    int64_t prod = (int64_t)acc_with_bias * (int64_t)mult;
    if (shift > 0) {
        int64_t rounded = prod + (int64_t(1) << (shift - 1));
        int64_t r = rounded >> shift;
        if (r > INT32_MAX) r = INT32_MAX;
        if (r < INT32_MIN) r = INT32_MIN;
        return (int32_t)r;
    } else if (shift == 0) {
        if (prod > INT32_MAX) prod = INT32_MAX;
        if (prod < INT32_MIN) prod = INT32_MIN;
        return (int32_t)prod;
    } else {
        int64_t r = prod << (-shift);
        if (r > INT32_MAX) r = INT32_MAX;
        if (r < INT32_MIN) r = INT32_MIN;
        return (int32_t)r;
    }
}

// LeakyReLU(alpha=0.1) approximated as (x * 13) >> 7 for negatives.
inline int32_t leaky_relu_i32(int32_t x) {
    return x < 0 ? ((x * 13) >> 7) : x;
}

inline int8_t saturate_i8(int32_t x) {
    if (x > 127) return 127;
    if (x < -128) return -128;
    return (int8_t)x;
}

// --------------------------------------------------------------------------
// Conv2D (Conv op)
//   in_i8:   [IC, H, W]
//   w_i8:    [OC, IC, KH, KW]
//   bias:    [OC]
//   mult:    [OC]
//   shift:   [OC]
//   out_i8:  [OC, OH, OW]
//     OH = (H + 2*pad_h - KH) / stride_h + 1
//     OW = (W + 2*pad_w - KW) / stride_w + 1
// --------------------------------------------------------------------------
inline void conv2d_scalar(
    const int8_t*  in_i8,
    const int8_t*  w_i8,
    const int32_t* bias_i32,
    const int32_t* mult_i32,
    const int8_t*  shift_i8,
    int8_t*        out_i8,
    int IC, int H, int W,
    int OC, int KH, int KW,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    bool has_leaky_relu)
{
    const int OH = (H + 2 * pad_h - KH) / stride_h + 1;
    const int OW = (W + 2 * pad_w - KW) / stride_w + 1;

    for (int oc = 0; oc < OC; ++oc) {
        const int32_t b   = bias_i32[oc];
        const int32_t m   = mult_i32[oc];
        const int     sh  = (int)shift_i8[oc];

        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
                int32_t acc = 0;

                for (int ic = 0; ic < IC; ++ic) {
                    for (int kh = 0; kh < KH; ++kh) {
                        const int ih = oh * stride_h + kh - pad_h;
                        if (ih < 0 || ih >= H) continue;

                        for (int kw = 0; kw < KW; ++kw) {
                            const int iw = ow * stride_w + kw - pad_w;
                            if (iw < 0 || iw >= W) continue;

                            const int8_t a = in_i8[(ic * H + ih) * W + iw];
                            const int8_t w = w_i8[((oc * IC + ic) * KH + kh) * KW + kw];
                            acc += (int32_t)a * (int32_t)w;
                        }
                    }
                }

                acc += b;
                int32_t scaled = requantize_one(acc, m, sh);
                if (has_leaky_relu) scaled = leaky_relu_i32(scaled);
                out_i8[(oc * OH + oh) * OW + ow] = saturate_i8(scaled);
            }
        }
    }
}

// --------------------------------------------------------------------------
// ConvTranspose2D
//   in_i8:   [IC, H, W]
//   w_i8:    [IC, OC, KH, KW]   <-- note the different weight layout vs Conv
//   out_i8:  [OC, OH, OW]
//     OH = stride_h * (H - 1) + KH - 2*pad_h + out_pad_h
//     OW = stride_w * (W - 1) + KW - 2*pad_w + out_pad_w
//
// Math: iterate each output pixel; it receives contributions from every
// input pixel (ih, iw) such that oh + pad_h - kh is divisible by stride_h
// and yields an in-range ih (similarly for w).
// --------------------------------------------------------------------------
inline void conv_transpose2d_scalar(
    const int8_t*  in_i8,
    const int8_t*  w_i8,
    const int32_t* bias_i32,
    const int32_t* mult_i32,
    const int8_t*  shift_i8,
    int8_t*        out_i8,
    int IC, int H, int W,
    int OC, int KH, int KW,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int out_pad_h, int out_pad_w,
    bool has_leaky_relu)
{
    const int OH = stride_h * (H - 1) + KH - 2 * pad_h + out_pad_h;
    const int OW = stride_w * (W - 1) + KW - 2 * pad_w + out_pad_w;

    for (int oc = 0; oc < OC; ++oc) {
        const int32_t b   = bias_i32[oc];
        const int32_t m   = mult_i32[oc];
        const int     sh  = (int)shift_i8[oc];

        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
                int32_t acc = 0;

                for (int kh = 0; kh < KH; ++kh) {
                    const int ih_s = oh + pad_h - kh;
                    if (ih_s < 0 || (ih_s % stride_h) != 0) continue;
                    const int ih = ih_s / stride_h;
                    if (ih >= H) continue;

                    for (int kw = 0; kw < KW; ++kw) {
                        const int iw_s = ow + pad_w - kw;
                        if (iw_s < 0 || (iw_s % stride_w) != 0) continue;
                        const int iw = iw_s / stride_w;
                        if (iw >= W) continue;

                        for (int ic = 0; ic < IC; ++ic) {
                            const int8_t a = in_i8[(ic * H + ih) * W + iw];
                            // ConvTranspose weight layout: [IC, OC, KH, KW]
                            const int8_t w = w_i8[((ic * OC + oc) * KH + kh) * KW + kw];
                            acc += (int32_t)a * (int32_t)w;
                        }
                    }
                }

                acc += b;
                int32_t scaled = requantize_one(acc, m, sh);
                if (has_leaky_relu) scaled = leaky_relu_i32(scaled);
                out_i8[(oc * OH + oh) * OW + ow] = saturate_i8(scaled);
            }
        }
    }
}

} // namespace scalar_kernel
