// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2024.1 (64-bit)
// Tool Version Limit: 2024.05
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
`timescale 1ns/1ps
module registration_accumulate_kernel_control_r_s_axi
#(parameter
    C_S_AXI_ADDR_WIDTH = 7,
    C_S_AXI_DATA_WIDTH = 32
)(
    input  wire                          ACLK,
    input  wire                          ARESET,
    input  wire                          ACLK_EN,
    input  wire [C_S_AXI_ADDR_WIDTH-1:0] AWADDR,
    input  wire                          AWVALID,
    output wire                          AWREADY,
    input  wire [C_S_AXI_DATA_WIDTH-1:0] WDATA,
    input  wire [C_S_AXI_DATA_WIDTH/8-1:0] WSTRB,
    input  wire                          WVALID,
    output wire                          WREADY,
    output wire [1:0]                    BRESP,
    output wire                          BVALID,
    input  wire                          BREADY,
    input  wire [C_S_AXI_ADDR_WIDTH-1:0] ARADDR,
    input  wire                          ARVALID,
    output wire                          ARREADY,
    output wire [C_S_AXI_DATA_WIDTH-1:0] RDATA,
    output wire [1:0]                    RRESP,
    output wire                          RVALID,
    input  wire                          RREADY,
    output wire [63:0]                   src_xyz,
    output wire [63:0]                   tgt_xyz,
    output wire [63:0]                   labels,
    output wire [63:0]                   jtj_out,
    output wire [63:0]                   jtr_out,
    output wire [63:0]                   used_count,
    output wire [63:0]                   dropped_count
);
//------------------------Address Info-------------------
// Protocol Used: ap_ctrl_none
//
// 0x00 : reserved
// 0x04 : reserved
// 0x08 : reserved
// 0x0c : reserved
// 0x10 : Data signal of src_xyz
//        bit 31~0 - src_xyz[31:0] (Read/Write)
// 0x14 : Data signal of src_xyz
//        bit 31~0 - src_xyz[63:32] (Read/Write)
// 0x18 : reserved
// 0x1c : Data signal of tgt_xyz
//        bit 31~0 - tgt_xyz[31:0] (Read/Write)
// 0x20 : Data signal of tgt_xyz
//        bit 31~0 - tgt_xyz[63:32] (Read/Write)
// 0x24 : reserved
// 0x28 : Data signal of labels
//        bit 31~0 - labels[31:0] (Read/Write)
// 0x2c : Data signal of labels
//        bit 31~0 - labels[63:32] (Read/Write)
// 0x30 : reserved
// 0x34 : Data signal of jtj_out
//        bit 31~0 - jtj_out[31:0] (Read/Write)
// 0x38 : Data signal of jtj_out
//        bit 31~0 - jtj_out[63:32] (Read/Write)
// 0x3c : reserved
// 0x40 : Data signal of jtr_out
//        bit 31~0 - jtr_out[31:0] (Read/Write)
// 0x44 : Data signal of jtr_out
//        bit 31~0 - jtr_out[63:32] (Read/Write)
// 0x48 : reserved
// 0x4c : Data signal of used_count
//        bit 31~0 - used_count[31:0] (Read/Write)
// 0x50 : Data signal of used_count
//        bit 31~0 - used_count[63:32] (Read/Write)
// 0x54 : reserved
// 0x58 : Data signal of dropped_count
//        bit 31~0 - dropped_count[31:0] (Read/Write)
// 0x5c : Data signal of dropped_count
//        bit 31~0 - dropped_count[63:32] (Read/Write)
// 0x60 : reserved
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

//------------------------Parameter----------------------
localparam
    ADDR_SRC_XYZ_DATA_0       = 7'h10,
    ADDR_SRC_XYZ_DATA_1       = 7'h14,
    ADDR_SRC_XYZ_CTRL         = 7'h18,
    ADDR_TGT_XYZ_DATA_0       = 7'h1c,
    ADDR_TGT_XYZ_DATA_1       = 7'h20,
    ADDR_TGT_XYZ_CTRL         = 7'h24,
    ADDR_LABELS_DATA_0        = 7'h28,
    ADDR_LABELS_DATA_1        = 7'h2c,
    ADDR_LABELS_CTRL          = 7'h30,
    ADDR_JTJ_OUT_DATA_0       = 7'h34,
    ADDR_JTJ_OUT_DATA_1       = 7'h38,
    ADDR_JTJ_OUT_CTRL         = 7'h3c,
    ADDR_JTR_OUT_DATA_0       = 7'h40,
    ADDR_JTR_OUT_DATA_1       = 7'h44,
    ADDR_JTR_OUT_CTRL         = 7'h48,
    ADDR_USED_COUNT_DATA_0    = 7'h4c,
    ADDR_USED_COUNT_DATA_1    = 7'h50,
    ADDR_USED_COUNT_CTRL      = 7'h54,
    ADDR_DROPPED_COUNT_DATA_0 = 7'h58,
    ADDR_DROPPED_COUNT_DATA_1 = 7'h5c,
    ADDR_DROPPED_COUNT_CTRL   = 7'h60,
    WRIDLE                    = 2'd0,
    WRDATA                    = 2'd1,
    WRRESP                    = 2'd2,
    WRRESET                   = 2'd3,
    RDIDLE                    = 2'd0,
    RDDATA                    = 2'd1,
    RDRESET                   = 2'd2,
    ADDR_BITS                = 7;

//------------------------Local signal-------------------
    reg  [1:0]                    wstate = WRRESET;
    reg  [1:0]                    wnext;
    reg  [ADDR_BITS-1:0]          waddr;
    wire [C_S_AXI_DATA_WIDTH-1:0] wmask;
    wire                          aw_hs;
    wire                          w_hs;
    reg  [1:0]                    rstate = RDRESET;
    reg  [1:0]                    rnext;
    reg  [C_S_AXI_DATA_WIDTH-1:0] rdata;
    wire                          ar_hs;
    wire [ADDR_BITS-1:0]          raddr;
    // internal registers
    reg  [63:0]                   int_src_xyz = 'b0;
    reg  [63:0]                   int_tgt_xyz = 'b0;
    reg  [63:0]                   int_labels = 'b0;
    reg  [63:0]                   int_jtj_out = 'b0;
    reg  [63:0]                   int_jtr_out = 'b0;
    reg  [63:0]                   int_used_count = 'b0;
    reg  [63:0]                   int_dropped_count = 'b0;

//------------------------Instantiation------------------


//------------------------AXI write fsm------------------
assign AWREADY = (wstate == WRIDLE);
assign WREADY  = (wstate == WRDATA);
assign BRESP   = 2'b00;  // OKAY
assign BVALID  = (wstate == WRRESP);
assign wmask   = { {8{WSTRB[3]}}, {8{WSTRB[2]}}, {8{WSTRB[1]}}, {8{WSTRB[0]}} };
assign aw_hs   = AWVALID & AWREADY;
assign w_hs    = WVALID & WREADY;

// wstate
always @(posedge ACLK) begin
    if (ARESET)
        wstate <= WRRESET;
    else if (ACLK_EN)
        wstate <= wnext;
end

// wnext
always @(*) begin
    case (wstate)
        WRIDLE:
            if (AWVALID)
                wnext = WRDATA;
            else
                wnext = WRIDLE;
        WRDATA:
            if (WVALID)
                wnext = WRRESP;
            else
                wnext = WRDATA;
        WRRESP:
            if (BREADY)
                wnext = WRIDLE;
            else
                wnext = WRRESP;
        default:
            wnext = WRIDLE;
    endcase
end

// waddr
always @(posedge ACLK) begin
    if (ACLK_EN) begin
        if (aw_hs)
            waddr <= {AWADDR[ADDR_BITS-1:2], {2{1'b0}}};
    end
end

//------------------------AXI read fsm-------------------
assign ARREADY = (rstate == RDIDLE);
assign RDATA   = rdata;
assign RRESP   = 2'b00;  // OKAY
assign RVALID  = (rstate == RDDATA);
assign ar_hs   = ARVALID & ARREADY;
assign raddr   = ARADDR[ADDR_BITS-1:0];

// rstate
always @(posedge ACLK) begin
    if (ARESET)
        rstate <= RDRESET;
    else if (ACLK_EN)
        rstate <= rnext;
end

// rnext
always @(*) begin
    case (rstate)
        RDIDLE:
            if (ARVALID)
                rnext = RDDATA;
            else
                rnext = RDIDLE;
        RDDATA:
            if (RREADY & RVALID)
                rnext = RDIDLE;
            else
                rnext = RDDATA;
        default:
            rnext = RDIDLE;
    endcase
end

// rdata
always @(posedge ACLK) begin
    if (ACLK_EN) begin
        if (ar_hs) begin
            rdata <= 'b0;
            case (raddr)
                ADDR_SRC_XYZ_DATA_0: begin
                    rdata <= int_src_xyz[31:0];
                end
                ADDR_SRC_XYZ_DATA_1: begin
                    rdata <= int_src_xyz[63:32];
                end
                ADDR_TGT_XYZ_DATA_0: begin
                    rdata <= int_tgt_xyz[31:0];
                end
                ADDR_TGT_XYZ_DATA_1: begin
                    rdata <= int_tgt_xyz[63:32];
                end
                ADDR_LABELS_DATA_0: begin
                    rdata <= int_labels[31:0];
                end
                ADDR_LABELS_DATA_1: begin
                    rdata <= int_labels[63:32];
                end
                ADDR_JTJ_OUT_DATA_0: begin
                    rdata <= int_jtj_out[31:0];
                end
                ADDR_JTJ_OUT_DATA_1: begin
                    rdata <= int_jtj_out[63:32];
                end
                ADDR_JTR_OUT_DATA_0: begin
                    rdata <= int_jtr_out[31:0];
                end
                ADDR_JTR_OUT_DATA_1: begin
                    rdata <= int_jtr_out[63:32];
                end
                ADDR_USED_COUNT_DATA_0: begin
                    rdata <= int_used_count[31:0];
                end
                ADDR_USED_COUNT_DATA_1: begin
                    rdata <= int_used_count[63:32];
                end
                ADDR_DROPPED_COUNT_DATA_0: begin
                    rdata <= int_dropped_count[31:0];
                end
                ADDR_DROPPED_COUNT_DATA_1: begin
                    rdata <= int_dropped_count[63:32];
                end
            endcase
        end
    end
end


//------------------------Register logic-----------------
assign src_xyz       = int_src_xyz;
assign tgt_xyz       = int_tgt_xyz;
assign labels        = int_labels;
assign jtj_out       = int_jtj_out;
assign jtr_out       = int_jtr_out;
assign used_count    = int_used_count;
assign dropped_count = int_dropped_count;
// int_src_xyz[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_src_xyz[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_SRC_XYZ_DATA_0)
            int_src_xyz[31:0] <= (WDATA[31:0] & wmask) | (int_src_xyz[31:0] & ~wmask);
    end
end

// int_src_xyz[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_src_xyz[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_SRC_XYZ_DATA_1)
            int_src_xyz[63:32] <= (WDATA[31:0] & wmask) | (int_src_xyz[63:32] & ~wmask);
    end
end

// int_tgt_xyz[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_tgt_xyz[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_TGT_XYZ_DATA_0)
            int_tgt_xyz[31:0] <= (WDATA[31:0] & wmask) | (int_tgt_xyz[31:0] & ~wmask);
    end
end

// int_tgt_xyz[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_tgt_xyz[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_TGT_XYZ_DATA_1)
            int_tgt_xyz[63:32] <= (WDATA[31:0] & wmask) | (int_tgt_xyz[63:32] & ~wmask);
    end
end

// int_labels[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_labels[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_LABELS_DATA_0)
            int_labels[31:0] <= (WDATA[31:0] & wmask) | (int_labels[31:0] & ~wmask);
    end
end

// int_labels[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_labels[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_LABELS_DATA_1)
            int_labels[63:32] <= (WDATA[31:0] & wmask) | (int_labels[63:32] & ~wmask);
    end
end

// int_jtj_out[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_jtj_out[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_JTJ_OUT_DATA_0)
            int_jtj_out[31:0] <= (WDATA[31:0] & wmask) | (int_jtj_out[31:0] & ~wmask);
    end
end

// int_jtj_out[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_jtj_out[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_JTJ_OUT_DATA_1)
            int_jtj_out[63:32] <= (WDATA[31:0] & wmask) | (int_jtj_out[63:32] & ~wmask);
    end
end

// int_jtr_out[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_jtr_out[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_JTR_OUT_DATA_0)
            int_jtr_out[31:0] <= (WDATA[31:0] & wmask) | (int_jtr_out[31:0] & ~wmask);
    end
end

// int_jtr_out[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_jtr_out[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_JTR_OUT_DATA_1)
            int_jtr_out[63:32] <= (WDATA[31:0] & wmask) | (int_jtr_out[63:32] & ~wmask);
    end
end

// int_used_count[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_used_count[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_USED_COUNT_DATA_0)
            int_used_count[31:0] <= (WDATA[31:0] & wmask) | (int_used_count[31:0] & ~wmask);
    end
end

// int_used_count[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_used_count[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_USED_COUNT_DATA_1)
            int_used_count[63:32] <= (WDATA[31:0] & wmask) | (int_used_count[63:32] & ~wmask);
    end
end

// int_dropped_count[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_dropped_count[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_DROPPED_COUNT_DATA_0)
            int_dropped_count[31:0] <= (WDATA[31:0] & wmask) | (int_dropped_count[31:0] & ~wmask);
    end
end

// int_dropped_count[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_dropped_count[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_DROPPED_COUNT_DATA_1)
            int_dropped_count[63:32] <= (WDATA[31:0] & wmask) | (int_dropped_count[63:32] & ~wmask);
    end
end


//------------------------Memory logic-------------------

endmodule
