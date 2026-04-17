// =============================================================================
// AIE_graph.cpp — ADF graph for RangeNet DarkNet53 INT8 inference
//
// Defines three kernel types matching conv_kernel.cpp:
//   - conv2d_3x3_tile:  3×3 conv with im2col + GEMM
//   - conv2d_1x1_tile:  1×1 pointwise conv (pure GEMM)
//   - elem_add_tile:    saturating INT8 residual addition
//
// Each conv kernel has 6 ports:
//   in[0]  = activations  (int8)
//   in[1]  = weights      (int8, GEMM-B layout)
//   in[2]  = bias         (int32, per-channel)
//   in[3]  = requant_mult (int32, per-channel)
//   in[4]  = requant_shift(int8, per-channel)
//   out[0] = output       (int8)
//
// The host/PL data mover is responsible for:
//   - Spatial tiling (64×2048 → 8×32 tiles with halo)
//   - Channel blocking (IC/OC > 32 → multiple kernel calls)
//   - Accumulating partial sums across IC blocks
//   - Weight transposition to GEMM-B layout
// =============================================================================

#include "conv_kernel.h"

// Buffer sizes from conv_kernel.cpp
constexpr int BIAS_BUF    = OC_BLOCK * sizeof(int32_t);   // 128 bytes
constexpr int RQ_MULT_BUF = OC_BLOCK * sizeof(int32_t);   // 128 bytes
constexpr int RQ_SHIFT_BUF = OC_BLOCK * sizeof(int8_t);   //  32 bytes


// =============================================================================
// Conv3x3Graph: Subgraph wrapping a single 3×3 conv tile kernel
// =============================================================================

class Conv3x3Graph : public adf::graph {
public:
    adf::port<input>  in_act;
    adf::port<input>  in_wt;
    adf::port<input>  in_bias;
    adf::port<input>  in_rq_mult;
    adf::port<input>  in_rq_shift;
    adf::port<output> out_act;

    adf::kernel k;

    Conv3x3Graph() {
        k = adf::kernel::create(conv2d_3x3_tile);
        adf::source(k) = "conv_kernel.cpp";
        adf::runtime<adf::ratio>(k) = 0.9;

        // Connect ports → kernel buffer inputs
        adf::connect<adf::window<IN_BUF_3x3>>(in_act,      k.in[0]);
        adf::connect<adf::window<WT_BUF_3x3>>(in_wt,       k.in[1]);
        adf::connect<adf::window<OC_BLOCK * 4>>(in_bias,    k.in[2]);   // int32 × OC_BLOCK
        adf::connect<adf::window<OC_BLOCK * 4>>(in_rq_mult, k.in[3]);   // int32 × OC_BLOCK
        adf::connect<adf::window<OC_BLOCK>>(in_rq_shift,    k.in[4]);   // int8  × OC_BLOCK

        // Kernel output
        adf::connect<adf::window<OUT_BUF>>(k.out[0], out_act);
    }
};


// =============================================================================
// Conv1x1Graph: Subgraph wrapping a single 1×1 conv tile kernel
// =============================================================================

class Conv1x1Graph : public adf::graph {
public:
    adf::port<input>  in_act;
    adf::port<input>  in_wt;
    adf::port<input>  in_bias;
    adf::port<input>  in_rq_mult;
    adf::port<input>  in_rq_shift;
    adf::port<output> out_act;

    adf::kernel k;

    Conv1x1Graph() {
        k = adf::kernel::create(conv2d_1x1_tile);
        adf::source(k) = "conv_kernel.cpp";
        adf::runtime<adf::ratio>(k) = 0.9;

        adf::connect<adf::window<IN_BUF_1x1>>(in_act,      k.in[0]);
        adf::connect<adf::window<WT_BUF_1x1>>(in_wt,       k.in[1]);
        adf::connect<adf::window<OC_BLOCK * 4>>(in_bias,    k.in[2]);
        adf::connect<adf::window<OC_BLOCK * 4>>(in_rq_mult, k.in[3]);
        adf::connect<adf::window<OC_BLOCK>>(in_rq_shift,    k.in[4]);

        adf::connect<adf::window<OUT_BUF>>(k.out[0], out_act);
    }
};


// =============================================================================
// ElemAddGraph: Subgraph wrapping the residual add kernel
// =============================================================================

class ElemAddGraph : public adf::graph {
public:
    adf::port<input>  in_a;
    adf::port<input>  in_b;
    adf::port<output> out_sum;

    adf::kernel k;

    ElemAddGraph() {
        k = adf::kernel::create(elem_add_tile);
        adf::source(k) = "conv_kernel.cpp";
        adf::runtime<adf::ratio>(k) = 0.5;

        adf::connect<adf::window<OUT_BUF>>(in_a, k.in[0]);
        adf::connect<adf::window<OUT_BUF>>(in_b, k.in[1]);
        adf::connect<adf::window<OUT_BUF>>(k.out[0], out_sum);
    }
};


// =============================================================================
// RangeNetGraph: Top-level graph with PLIOs for host/PL communication
//
// This instantiates one of each kernel type. The host/PL data mover calls
// them repeatedly for each spatial tile and channel block.
// =============================================================================

class RangeNetGraph : public adf::graph {
public:
    // PLIOs for 3×3 conv
    adf::input_plio  plio_3x3_act;
    adf::input_plio  plio_3x3_wt;
    adf::input_plio  plio_3x3_bias;
    adf::input_plio  plio_3x3_rq_mult;
    adf::input_plio  plio_3x3_rq_shift;
    adf::output_plio plio_3x3_out;

    // PLIOs for 1×1 conv
    adf::input_plio  plio_1x1_act;
    adf::input_plio  plio_1x1_wt;
    adf::input_plio  plio_1x1_bias;
    adf::input_plio  plio_1x1_rq_mult;
    adf::input_plio  plio_1x1_rq_shift;
    adf::output_plio plio_1x1_out;

    // PLIOs for elem add
    adf::input_plio  plio_add_a;
    adf::input_plio  plio_add_b;
    adf::output_plio plio_add_out;

    // Subgraphs
    Conv3x3Graph  g_conv3x3;
    Conv1x1Graph  g_conv1x1;
    ElemAddGraph  g_elem_add;

    RangeNetGraph() {
        // ---- 3×3 conv PLIOs ----
        plio_3x3_act      = adf::input_plio::create("in_3x3_act",
                                adf::plio_64_bits, "data/act_3x3.txt");
        plio_3x3_wt       = adf::input_plio::create("in_3x3_wt",
                                adf::plio_64_bits, "data/wt_3x3.txt");
        plio_3x3_bias     = adf::input_plio::create("in_3x3_bias",
                                adf::plio_32_bits, "data/bias_3x3.txt");
        plio_3x3_rq_mult  = adf::input_plio::create("in_3x3_rq_mult",
                                adf::plio_32_bits, "data/rq_mult_3x3.txt");
        plio_3x3_rq_shift = adf::input_plio::create("in_3x3_rq_shift",
                                adf::plio_32_bits, "data/rq_shift_3x3.txt");
        plio_3x3_out      = adf::output_plio::create("out_3x3",
                                adf::plio_64_bits, "data/out_3x3.txt");

        adf::connect<>(plio_3x3_act.out[0],      g_conv3x3.in_act);
        adf::connect<>(plio_3x3_wt.out[0],       g_conv3x3.in_wt);
        adf::connect<>(plio_3x3_bias.out[0],     g_conv3x3.in_bias);
        adf::connect<>(plio_3x3_rq_mult.out[0],  g_conv3x3.in_rq_mult);
        adf::connect<>(plio_3x3_rq_shift.out[0], g_conv3x3.in_rq_shift);
        adf::connect<>(g_conv3x3.out_act,        plio_3x3_out.in[0]);

        // ---- 1×1 conv PLIOs ----
        plio_1x1_act      = adf::input_plio::create("in_1x1_act",
                                adf::plio_64_bits, "data/act_1x1.txt");
        plio_1x1_wt       = adf::input_plio::create("in_1x1_wt",
                                adf::plio_64_bits, "data/wt_1x1.txt");
        plio_1x1_bias     = adf::input_plio::create("in_1x1_bias",
                                adf::plio_32_bits, "data/bias_1x1.txt");
        plio_1x1_rq_mult  = adf::input_plio::create("in_1x1_rq_mult",
                                adf::plio_32_bits, "data/rq_mult_1x1.txt");
        plio_1x1_rq_shift = adf::input_plio::create("in_1x1_rq_shift",
                                adf::plio_32_bits, "data/rq_shift_1x1.txt");
        plio_1x1_out      = adf::output_plio::create("out_1x1",
                                adf::plio_64_bits, "data/out_1x1.txt");

        adf::connect<>(plio_1x1_act.out[0],      g_conv1x1.in_act);
        adf::connect<>(plio_1x1_wt.out[0],       g_conv1x1.in_wt);
        adf::connect<>(plio_1x1_bias.out[0],     g_conv1x1.in_bias);
        adf::connect<>(plio_1x1_rq_mult.out[0],  g_conv1x1.in_rq_mult);
        adf::connect<>(plio_1x1_rq_shift.out[0], g_conv1x1.in_rq_shift);
        adf::connect<>(g_conv1x1.out_act,        plio_1x1_out.in[0]);

        // ---- Elem add PLIOs ----
        plio_add_a   = adf::input_plio::create("in_add_a",
                           adf::plio_64_bits, "data/add_a.txt");
        plio_add_b   = adf::input_plio::create("in_add_b",
                           adf::plio_64_bits, "data/add_b.txt");
        plio_add_out = adf::output_plio::create("out_add",
                           adf::plio_64_bits, "data/out_add.txt");

        adf::connect<>(plio_add_a.out[0],   g_elem_add.in_a);
        adf::connect<>(plio_add_b.out[0],   g_elem_add.in_b);
        adf::connect<>(g_elem_add.out_sum,  plio_add_out.in[0]);
    }
};


// Top-level graph instantiation
RangeNetGraph my_graph;

int main() {
    my_graph.init();
    my_graph.run(1);   // one tile invocation for simulation
    my_graph.end();
    return 0;
}