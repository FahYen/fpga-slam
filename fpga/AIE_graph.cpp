
#include "conv_kernel.h"

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

        // FIX: Remove <adf::window<...>>. 
        // The compiler automatically matches these to the input_buffer in conv_kernel.h
        adf::connect<>(in_act,      k.in[0]);
        adf::connect<>(in_wt,       k.in[1]);
        adf::connect<>(in_bias,     k.in[2]);
        adf::connect<>(in_rq_mult,  k.in[3]);
        adf::connect<>(in_rq_shift, k.in[4]);
        adf::connect<>(k.out[0],    out_act);
    }
};

// =============================================================================
// Conv3x3S2Graph: Subgraph wrapping a 3×3 stride-(1,2) conv tile kernel
// =============================================================================

class Conv3x3S2Graph : public adf::graph {
public:
    adf::port<input>  in_act;
    adf::port<input>  in_wt;
    adf::port<input>  in_bias;
    adf::port<input>  in_rq_mult;
    adf::port<input>  in_rq_shift;
    adf::port<output> out_act;

    adf::kernel k;

    Conv3x3S2Graph() {
        k = adf::kernel::create(conv2d_3x3_s2_tile);
        adf::source(k) = "conv_kernel.cpp";
        adf::runtime<adf::ratio>(k) = 0.9;

        adf::connect<>(in_act,      k.in[0]);
        adf::connect<>(in_wt,       k.in[1]);
        adf::connect<>(in_bias,     k.in[2]);
        adf::connect<>(in_rq_mult,  k.in[3]);
        adf::connect<>(in_rq_shift, k.in[4]);
        adf::connect<>(k.out[0],    out_act);
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

        // FIX: Removed window templates to match input_buffer
        adf::connect<>(in_act,      k.in[0]);
        adf::connect<>(in_wt,       k.in[1]);
        adf::connect<>(in_bias,     k.in[2]);
        adf::connect<>(in_rq_mult,  k.in[3]);
        adf::connect<>(in_rq_shift, k.in[4]);
        adf::connect<>(k.out[0],    out_act);
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

        // FIX: Removed window templates to match input_buffer
        adf::connect<>(in_a,    k.in[0]);
        adf::connect<>(in_b,    k.in[1]);
        adf::connect<>(k.out[0], out_sum);
    }
};

// =============================================================================
// RangeNetGraph: Top-level graph with PLIOs
// =============================================================================

class RangeNetGraph : public adf::graph {
public:
    // 3×3 conv PLIOs
    // adf::input_plio  plio_3x3_act;
    // adf::input_plio  plio_3x3_wt;
    // adf::input_plio  plio_3x3_bias;
    // adf::input_plio  plio_3x3_rq_mult;
    // adf::input_plio  plio_3x3_rq_shift;
    // adf::output_plio plio_3x3_out;

    // 1×1 conv PLIOs
    adf::input_plio  plio_1x1_act;
    adf::input_plio  plio_1x1_wt;
    adf::input_plio  plio_1x1_bias;
    adf::input_plio  plio_1x1_rq_mult;
    adf::input_plio  plio_1x1_rq_shift;
    adf::output_plio plio_1x1_out;

    // 3×3 stride-2 conv PLIOs
    // adf::input_plio  plio_3x3s2_act;
    // adf::input_plio  plio_3x3s2_wt;
    // adf::input_plio  plio_3x3s2_bias;
    // adf::input_plio  plio_3x3s2_rq_mult;
    // adf::input_plio  plio_3x3s2_rq_shift;
    // adf::output_plio plio_3x3s2_out;

    // Elem add PLIOs
    // adf::input_plio  plio_add_a;
    // adf::input_plio  plio_add_b;
    // adf::output_plio plio_add_out;

    // Conv3x3Graph    g_conv3x3;
    // Conv3x3S2Graph  g_conv3x3_s2;
    Conv1x1Graph    g_conv1x1;
    // ElemAddGraph    g_elem_add;

    RangeNetGraph() {
        // ---- 3×3 conv PLIO Setup ----
        // plio_3x3_act      = adf::input_plio::create("in_3x3_act", adf::plio_64_bits, "data/act_3x3.txt");
        // plio_3x3_wt       = adf::input_plio::create("in_3x3_wt",  adf::plio_64_bits, "data/wt_3x3.txt");
        // plio_3x3_bias     = adf::input_plio::create("in_3x3_bias", adf::plio_32_bits, "data/bias_3x3.txt");
        // plio_3x3_rq_mult  = adf::input_plio::create("in_3x3_rq_mult", adf::plio_32_bits, "data/rq_mult_3x3.txt");
        // plio_3x3_rq_shift = adf::input_plio::create("in_3x3_rq_shift", adf::plio_32_bits, "data/rq_shift_3x3.txt");
        // plio_3x3_out      = adf::output_plio::create("out_3x3", adf::plio_64_bits, "data/out_3x3.txt");

        // adf::connect<>(plio_3x3_act.out[0],      g_conv3x3.in_act);
        // adf::connect<>(plio_3x3_wt.out[0],       g_conv3x3.in_wt);
        // adf::connect<>(plio_3x3_bias.out[0],     g_conv3x3.in_bias);
        // adf::connect<>(plio_3x3_rq_mult.out[0],  g_conv3x3.in_rq_mult);
        // adf::connect<>(plio_3x3_rq_shift.out[0], g_conv3x3.in_rq_shift);
        // adf::connect<>(g_conv3x3.out_act,        plio_3x3_out.in[0]);

        // ---- 1×1 conv PLIO Setup ----
        plio_1x1_act      = adf::input_plio::create("in_1x1_act", adf::plio_64_bits, "data/act_1x1.txt");
        plio_1x1_wt       = adf::input_plio::create("in_1x1_wt",  adf::plio_64_bits, "data/wt_1x1.txt");
        plio_1x1_bias     = adf::input_plio::create("in_1x1_bias", adf::plio_32_bits, "data/bias_1x1.txt");
        plio_1x1_rq_mult  = adf::input_plio::create("in_1x1_rq_mult", adf::plio_32_bits, "data/rq_mult_1x1.txt");
        plio_1x1_rq_shift = adf::input_plio::create("in_1x1_rq_shift", adf::plio_32_bits, "data/rq_shift_1x1.txt");
        plio_1x1_out      = adf::output_plio::create("out_1x1", adf::plio_64_bits, "data/out_1x1.txt");

        adf::connect<>(plio_1x1_act.out[0],      g_conv1x1.in_act);
        adf::connect<>(plio_1x1_wt.out[0],       g_conv1x1.in_wt);
        adf::connect<>(plio_1x1_bias.out[0],     g_conv1x1.in_bias);
        adf::connect<>(plio_1x1_rq_mult.out[0],  g_conv1x1.in_rq_mult);
        adf::connect<>(plio_1x1_rq_shift.out[0], g_conv1x1.in_rq_shift);
        adf::connect<>(g_conv1x1.out_act,        plio_1x1_out.in[0]);

        // ---- 3×3 stride-2 PLIO Setup ----
        // plio_3x3s2_act      = adf::input_plio::create("in_3x3s2_act", adf::plio_64_bits, "data/act_3x3s2.txt");
        // plio_3x3s2_wt       = adf::input_plio::create("in_3x3s2_wt",  adf::plio_64_bits, "data/wt_3x3s2.txt");
        // plio_3x3s2_bias     = adf::input_plio::create("in_3x3s2_bias", adf::plio_32_bits, "data/bias_3x3s2.txt");
        // plio_3x3s2_rq_mult  = adf::input_plio::create("in_3x3s2_rq_mult", adf::plio_32_bits, "data/rq_mult_3x3s2.txt");
        // plio_3x3s2_rq_shift = adf::input_plio::create("in_3x3s2_rq_shift", adf::plio_32_bits, "data/rq_shift_3x3s2.txt");
        // plio_3x3s2_out      = adf::output_plio::create("out_3x3s2", adf::plio_64_bits, "data/out_3x3s2.txt");

        // adf::connect<>(plio_3x3s2_act.out[0],      g_conv3x3_s2.in_act);
        // adf::connect<>(plio_3x3s2_wt.out[0],       g_conv3x3_s2.in_wt);
        // adf::connect<>(plio_3x3s2_bias.out[0],     g_conv3x3_s2.in_bias);
        // adf::connect<>(plio_3x3s2_rq_mult.out[0],  g_conv3x3_s2.in_rq_mult);
        // adf::connect<>(plio_3x3s2_rq_shift.out[0], g_conv3x3_s2.in_rq_shift);
        // adf::connect<>(g_conv3x3_s2.out_act,       plio_3x3s2_out.in[0]);

        // ---- Elem add PLIO Setup ----
        // plio_add_a   = adf::input_plio::create("in_add_a", adf::plio_64_bits, "data/add_a.txt");
        // plio_add_b   = adf::input_plio::create("in_add_b", adf::plio_64_bits, "data/add_b.txt");
        // plio_add_out = adf::output_plio::create("out_add", adf::plio_64_bits, "data/out_add.txt");

        // adf::connect<>(plio_add_a.out[0],   g_elem_add.in_a);
        // adf::connect<>(plio_add_b.out[0],   g_elem_add.in_b);
        // adf::connect<>(g_elem_add.out_sum,  plio_add_out.in[0]);
    }
};

RangeNetGraph my_graph;

// Simulation Wrapper
#if defined(__X86SIM__) || defined(__AIESIM__)
int main() {
    my_graph.init();
    my_graph.run(1);
    my_graph.end();
    return 0;
}
#endif