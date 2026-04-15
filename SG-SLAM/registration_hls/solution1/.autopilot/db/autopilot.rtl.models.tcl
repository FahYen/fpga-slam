set SynModuleInfo {
  {SRCNAME registration_accumulate_kernel_Pipeline_VITIS_LOOP_70_3 MODELNAME registration_accumulate_kernel_Pipeline_VITIS_LOOP_70_3 RTLNAME registration_accumulate_kernel_registration_accumulate_kernel_Pipeline_VITIS_LOOP_70_3
    SUBMODULES {
      {MODELNAME registration_accumulate_kernel_faddfsub_32ns_32ns_32_7_full_dsp_1 RTLNAME registration_accumulate_kernel_faddfsub_32ns_32ns_32_7_full_dsp_1 BINDTYPE op TYPE fsub IMPL fulldsp LATENCY 6 ALLOW_PRAGMA 1}
      {MODELNAME registration_accumulate_kernel_fdiv_32ns_32ns_32_12_no_dsp_1 RTLNAME registration_accumulate_kernel_fdiv_32ns_32ns_32_12_no_dsp_1 BINDTYPE op TYPE fdiv IMPL fabric LATENCY 11 ALLOW_PRAGMA 1}
      {MODELNAME registration_accumulate_kernel_fpext_32ns_64_2_no_dsp_1 RTLNAME registration_accumulate_kernel_fpext_32ns_64_2_no_dsp_1 BINDTYPE op TYPE fpext IMPL auto LATENCY 1 ALLOW_PRAGMA 1}
      {MODELNAME registration_accumulate_kernel_dadd_64ns_64ns_64_5_full_dsp_1 RTLNAME registration_accumulate_kernel_dadd_64ns_64ns_64_5_full_dsp_1 BINDTYPE op TYPE dadd IMPL fulldsp LATENCY 4 ALLOW_PRAGMA 1}
      {MODELNAME registration_accumulate_kernel_dmul_64ns_64ns_64_6_max_dsp_1 RTLNAME registration_accumulate_kernel_dmul_64ns_64ns_64_6_max_dsp_1 BINDTYPE op TYPE dmul IMPL maxdsp LATENCY 5 ALLOW_PRAGMA 1}
      {MODELNAME registration_accumulate_kernel_flow_control_loop_pipe_sequential_init RTLNAME registration_accumulate_kernel_flow_control_loop_pipe_sequential_init BINDTYPE interface TYPE internal_upc_flow_control INSTNAME registration_accumulate_kernel_flow_control_loop_pipe_sequential_init_U}
    }
  }
  {SRCNAME registration_accumulate_kernel_Pipeline_VITIS_LOOP_141_10 MODELNAME registration_accumulate_kernel_Pipeline_VITIS_LOOP_141_10 RTLNAME registration_accumulate_kernel_registration_accumulate_kernel_Pipeline_VITIS_LOOP_141_10
    SUBMODULES {
      {MODELNAME registration_accumulate_kernel_sparsemux_43_5_64_1_1 RTLNAME registration_accumulate_kernel_sparsemux_43_5_64_1_1 BINDTYPE op TYPE sparsemux IMPL auto}
    }
  }
  {SRCNAME registration_accumulate_kernel_Pipeline_VITIS_LOOP_151_11 MODELNAME registration_accumulate_kernel_Pipeline_VITIS_LOOP_151_11 RTLNAME registration_accumulate_kernel_registration_accumulate_kernel_Pipeline_VITIS_LOOP_151_11
    SUBMODULES {
      {MODELNAME registration_accumulate_kernel_sparsemux_13_3_64_1_1 RTLNAME registration_accumulate_kernel_sparsemux_13_3_64_1_1 BINDTYPE op TYPE sparsemux IMPL auto}
    }
  }
  {SRCNAME registration_accumulate_kernel MODELNAME registration_accumulate_kernel RTLNAME registration_accumulate_kernel IS_TOP 1
    SUBMODULES {
      {MODELNAME registration_accumulate_kernel_fmul_32ns_32ns_32_4_max_dsp_1 RTLNAME registration_accumulate_kernel_fmul_32ns_32ns_32_4_max_dsp_1 BINDTYPE op TYPE fmul IMPL maxdsp LATENCY 3 ALLOW_PRAGMA 1}
      {MODELNAME registration_accumulate_kernel_gmem0_m_axi RTLNAME registration_accumulate_kernel_gmem0_m_axi BINDTYPE interface TYPE adapter IMPL m_axi}
      {MODELNAME registration_accumulate_kernel_gmem1_m_axi RTLNAME registration_accumulate_kernel_gmem1_m_axi BINDTYPE interface TYPE adapter IMPL m_axi}
      {MODELNAME registration_accumulate_kernel_gmem2_m_axi RTLNAME registration_accumulate_kernel_gmem2_m_axi BINDTYPE interface TYPE adapter IMPL m_axi}
      {MODELNAME registration_accumulate_kernel_gmem3_m_axi RTLNAME registration_accumulate_kernel_gmem3_m_axi BINDTYPE interface TYPE adapter IMPL m_axi}
      {MODELNAME registration_accumulate_kernel_gmem4_m_axi RTLNAME registration_accumulate_kernel_gmem4_m_axi BINDTYPE interface TYPE adapter IMPL m_axi}
      {MODELNAME registration_accumulate_kernel_control_s_axi RTLNAME registration_accumulate_kernel_control_s_axi BINDTYPE interface TYPE interface_s_axilite}
      {MODELNAME registration_accumulate_kernel_control_r_s_axi RTLNAME registration_accumulate_kernel_control_r_s_axi BINDTYPE interface TYPE interface_s_axilite}
    }
  }
}
