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

#define CONTROL_R_ADDR_SRC_XYZ_DATA       0x10
#define CONTROL_R_BITS_SRC_XYZ_DATA       64
#define CONTROL_R_ADDR_TGT_XYZ_DATA       0x1c
#define CONTROL_R_BITS_TGT_XYZ_DATA       64
#define CONTROL_R_ADDR_LABELS_DATA        0x28
#define CONTROL_R_BITS_LABELS_DATA        64
#define CONTROL_R_ADDR_JTJ_OUT_DATA       0x34
#define CONTROL_R_BITS_JTJ_OUT_DATA       64
#define CONTROL_R_ADDR_JTR_OUT_DATA       0x40
#define CONTROL_R_BITS_JTR_OUT_DATA       64
#define CONTROL_R_ADDR_USED_COUNT_DATA    0x4c
#define CONTROL_R_BITS_USED_COUNT_DATA    64
#define CONTROL_R_ADDR_DROPPED_COUNT_DATA 0x58
#define CONTROL_R_BITS_DROPPED_COUNT_DATA 64
