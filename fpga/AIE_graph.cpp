// aie_graphs/rangenet_graph.h
#include <adf.h>

class RangeNetConvGraph : public adf::graph {
public:
    // PLIOs — interface between PL and AIE
    adf::input_plio  plio_in_act;
    adf::input_plio  plio_in_wt;
    adf::input_plio  plio_in_requant;
    adf::output_plio plio_out_act;
    
    // Kernel instance
    adf::kernel k_conv3x3;
    adf::kernel k_conv1x1;
    
    RangeNetConvGraph() {
        // Create both kernel variants
        k_conv3x3 = adf::kernel::create(conv2d_3x3);
        adf::source(k_conv3x3) = "aie_kernels/conv2d_core.h";
        adf::runtime<ratio>(k_conv3x3) = 0.9;
        
        k_conv1x1 = adf::kernel::create(conv2d_1x1);
        adf::source(k_conv1x1) = "aie_kernels/conv2d_core.h";
        adf::runtime<ratio>(k_conv1x1) = 0.9;
        
        // Connect PLIOs (PL streams) to kernel ports
        adf::connect<>(plio_in_act.out[0], k_conv3x3.in[0]);
        adf::connect<>(plio_in_wt.out[0], k_conv3x3.in[1]);
        adf::connect<>(plio_in_requant.out[0], k_conv3x3.in[2]);
        adf::connect<>(k_conv3x3.out[0], plio_out_act.in[0]);
    }
};

// Top-level graph instantiation
RangeNetConvGraph my_graph;

int main() {
    my_graph.init();
    my_graph.run(1);  // run one invocation
    my_graph.end();
    return 0;
}