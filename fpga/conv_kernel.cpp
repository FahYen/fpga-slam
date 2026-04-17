// aie_kernels/conv2d_core.h
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <adf.h>

// Tile dimensions — tuned to fit in AIE-ML local memory (32KB data)
// These control how much of the feature map one tile processes at once
constexpr int TILE_H = 8;
constexpr int TILE_W = 32;
constexpr int IC_BLOCK = 32;   // input channels per block
constexpr int OC_BLOCK = 32;   // output channels per block

void conv2d_3x3(
    input_buffer<int8, extents<TILE_H * TILE_W * IC_BLOCK>>&     in_act,
    input_buffer<int8, extents<OC_BLOCK * IC_BLOCK * 9>>&        in_wt,
    input_buffer<int32, extents<OC_BLOCK>>&                       in_requant_mult,
    input_buffer<int8, extents<OC_BLOCK>>&                        in_requant_shift,
    output_buffer<int8, extents<(TILE_H-2) * (TILE_W-2) * OC_BLOCK>>& out_act
) {
    // Pointers to buffers
    int8_t* act  = (int8_t*)in_act.data();
    int8_t* wt   = (int8_t*)in_wt.data();
    int32_t* rq_m = (int32_t*)in_requant_mult.data();
    int8_t* rq_s  = (int8_t*)in_requant_shift.data();
    int8_t* out  = (int8_t*)out_act.data();
    
    // For each output spatial position
    for (int oh = 0; oh < TILE_H - 2; oh++) {
        for (int ow = 0; ow < TILE_W - 2; ow++) {
            
            // For each output channel block
            for (int oc = 0; oc < OC_BLOCK; oc++) {
                int32_t acc = 0;
                
                // 3x3 convolution across all input channels
                for (int ic = 0; ic < IC_BLOCK; ic++) {
                    for (int kh = 0; kh < 3; kh++) {
                        for (int kw = 0; kw < 3; kw++) {
                            int ih = oh + kh;
                            int iw = ow + kw;
                            
                            int8_t a = act[(ih * TILE_W + iw) * IC_BLOCK + ic];
                            int8_t w = wt[(oc * IC_BLOCK * 9) + (ic * 9) + (kh * 3 + kw)];
                            
                            acc += (int32_t)a * (int32_t)w;
                        }
                    }
                }
                
                // Requantize: multiply and shift
                int32_t scaled = (int64_t)acc * rq_m[oc] >> rq_s[oc];
                
                // LeakyReLU: if negative, multiply by ~0.1
                // 0.1 ≈ 13/128, so (x * 13) >> 7
                if (scaled < 0) {
                    scaled = (scaled * 13) >> 7;
                }
                
                // Saturate to INT8
                if (scaled > 127) scaled = 127;
                if (scaled < -128) scaled = -128;
                
                out[(oh * (TILE_W - 2) + ow) * OC_BLOCK + oc] = (int8_t)scaled;
            }
        }
    }
}

// 1x1 convolution — simpler, no spatial kernel window
void conv2d_1x1(
    input_buffer<int8, extents<TILE_H * TILE_W * IC_BLOCK>>&     in_act,
    input_buffer<int8, extents<OC_BLOCK * IC_BLOCK>>&             in_wt,
    input_buffer<int32, extents<OC_BLOCK>>&                       in_requant_mult,
    input_buffer<int8, extents<OC_BLOCK>>&                        in_requant_shift,
    output_buffer<int8, extents<TILE_H * TILE_W * OC_BLOCK>>&    out_act
) {
    int8_t* act  = (int8_t*)in_act.data();
    int8_t* wt   = (int8_t*)in_wt.data();
    int32_t* rq_m = (int32_t*)in_requant_mult.data();
    int8_t* rq_s  = (int8_t*)in_requant_shift.data();
    int8_t* out  = (int8_t*)out_act.data();
    
    for (int pos = 0; pos < TILE_H * TILE_W; pos++) {
        for (int oc = 0; oc < OC_BLOCK; oc++) {
            int32_t acc = 0;
            
            for (int ic = 0; ic < IC_BLOCK; ic++) {
                acc += (int32_t)act[pos * IC_BLOCK + ic] 
                     * (int32_t)wt[oc * IC_BLOCK + ic];
            }
            
            int32_t scaled = (int64_t)acc * rq_m[oc] >> rq_s[oc];
            if (scaled < 0) scaled = (scaled * 13) >> 7;
            if (scaled > 127) scaled = 127;
            if (scaled < -128) scaled = -128;
            
            out[pos * OC_BLOCK + oc] = (int8_t)scaled;
        }
    }
}