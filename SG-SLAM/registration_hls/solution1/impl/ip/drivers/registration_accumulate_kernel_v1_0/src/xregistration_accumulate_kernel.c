// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2024.1 (64-bit)
// Tool Version Limit: 2024.05
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
/***************************** Include Files *********************************/
#include "xregistration_accumulate_kernel.h"

/************************** Function Implementation *************************/
#ifndef __linux__
int XRegistration_accumulate_kernel_CfgInitialize(XRegistration_accumulate_kernel *InstancePtr, XRegistration_accumulate_kernel_Config *ConfigPtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(ConfigPtr != NULL);

    InstancePtr->Control_r_BaseAddress = ConfigPtr->Control_r_BaseAddress;
    InstancePtr->Control_BaseAddress = ConfigPtr->Control_BaseAddress;
    InstancePtr->IsReady = XIL_COMPONENT_IS_READY;

    return XST_SUCCESS;
}
#endif

void XRegistration_accumulate_kernel_Start(XRegistration_accumulate_kernel *InstancePtr) {
    u32 Data;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XRegistration_accumulate_kernel_ReadReg(InstancePtr->Control_BaseAddress, XREGISTRATION_ACCUMULATE_KERNEL_CONTROL_ADDR_AP_CTRL) & 0x80;
    XRegistration_accumulate_kernel_WriteReg(InstancePtr->Control_BaseAddress, XREGISTRATION_ACCUMULATE_KERNEL_CONTROL_ADDR_AP_CTRL, Data | 0x01);
}

u32 XRegistration_accumulate_kernel_IsDone(XRegistration_accumulate_kernel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XRegistration_accumulate_kernel_ReadReg(InstancePtr->Control_BaseAddress, XREGISTRATION_ACCUMULATE_KERNEL_CONTROL_ADDR_AP_CTRL);
    return (Data >> 1) & 0x1;
}

u32 XRegistration_accumulate_kernel_IsIdle(XRegistration_accumulate_kernel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XRegistration_accumulate_kernel_ReadReg(InstancePtr->Control_BaseAddress, XREGISTRATION_ACCUMULATE_KERNEL_CONTROL_ADDR_AP_CTRL);
    return (Data >> 2) & 0x1;
}

u32 XRegistration_accumulate_kernel_IsReady(XRegistration_accumulate_kernel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XRegistration_accumulate_kernel_ReadReg(InstancePtr->Control_BaseAddress, XREGISTRATION_ACCUMULATE_KERNEL_CONTROL_ADDR_AP_CTRL);
    // check ap_start to see if the pcore is ready for next input
    return !(Data & 0x1);
}

void XRegistration_accumulate_kernel_EnableAutoRestart(XRegistration_accumulate_kernel *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRegistration_accumulate_kernel_WriteReg(InstancePtr->Control_BaseAddress, XREGISTRATION_ACCUMULATE_KERNEL_CONTROL_ADDR_AP_CTRL, 0x80);
}

void XRegistration_accumulate_kernel_DisableAutoRestart(XRegistration_accumulate_kernel *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRegistration_accumulate_kernel_WriteReg(InstancePtr->Control_BaseAddress, XREGISTRATION_ACCUMULATE_KERNEL_CONTROL_ADDR_AP_CTRL, 0);
}

void XRegistration_accumulate_kernel_Set_src_xyz(XRegistration_accumulate_kernel *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRegistration_accumulate_kernel_WriteReg(InstancePtr->Control_r_BaseAddress, XREGISTRATION_ACCUMULATE_KERNEL_CONTROL_R_ADDR_SRC_XYZ_DATA, (u32)(Data));
    XRegistration_accumulate_kernel_WriteReg(InstancePtr->Control_r_BaseAddress, XREGISTRATION_ACCUMULATE_KERNEL_CONTROL_R_ADDR_SRC_XYZ_DATA + 4, (u32)(Data >> 32));
}

u64 XRegistration_accumulate_kernel_Get_src_xyz(XRegistration_accumulate_kernel *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XRegistration_accumulate_kernel_ReadReg(InstancePtr->Control_r_BaseAddress, XREGISTRATION_ACCUMULATE_KERNEL_CONTROL_R_ADDR_SRC_XYZ_DATA);
    Data += (u64)XRegistration_accumulate_kernel_ReadReg(InstancePtr->Control_r_BaseAddress, XREGISTRATION_ACCUMULATE_KERNEL_CONTROL_R_ADDR_SRC_XYZ_DATA + 4) << 32;
    return Data;
}

void XRegistration_accumulate_kernel_Set_tgt_xyz(XRegistration_accumulate_kernel *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRegistration_accumulate_kernel_WriteReg(InstancePtr->Control_r_BaseAddress, XREGISTRATION_ACCUMULATE_KERNEL_CONTROL_R_ADDR_TGT_XYZ_DATA, (u32)(Data));
    XRegistration_accumulate_kernel_WriteReg(InstancePtr->Control_r_BaseAddress, XREGISTRATION_ACCUMULATE_KERNEL_CONTROL_R_ADDR_TGT_XYZ_DATA + 4, (u32)(Data >> 32));
}

u64 XRegistration_accumulate_kernel_Get_tgt_xyz(XRegistration_accumulate_kernel *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XRegistration_accumulate_kernel_ReadReg(InstancePtr->Control_r_BaseAddress, XREGISTRATION_ACCUMULATE_KERNEL_CONTROL_R_ADDR_TGT_XYZ_DATA);
    Data += (u64)XRegistration_accumulate_kernel_ReadReg(InstancePtr->Control_r_BaseAddress, XREGISTRATION_ACCUMULATE_KERNEL_CONTROL_R_ADDR_TGT_XYZ_DATA + 4) << 32;
    return Data;
}

void XRegistration_accumulate_kernel_Set_labels(XRegistration_accumulate_kernel *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRegistration_accumulate_kernel_WriteReg(InstancePtr->Control_r_BaseAddress, XREGISTRATION_ACCUMULATE_KERNEL_CONTROL_R_ADDR_LABELS_DATA, (u32)(Data));
    XRegistration_accumulate_kernel_WriteReg(InstancePtr->Control_r_BaseAddress, XREGISTRATION_ACCUMULATE_KERNEL_CONTROL_R_ADDR_LABELS_DATA + 4, (u32)(Data >> 32));
}

u64 XRegistration_accumulate_kernel_Get_labels(XRegistration_accumulate_kernel *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XRegistration_accumulate_kernel_ReadReg(InstancePtr->Control_r_BaseAddress, XREGISTRATION_ACCUMULATE_KERNEL_CONTROL_R_ADDR_LABELS_DATA);
    Data += (u64)XRegistration_accumulate_kernel_ReadReg(InstancePtr->Control_r_BaseAddress, XREGISTRATION_ACCUMULATE_KERNEL_CONTROL_R_ADDR_LABELS_DATA + 4) << 32;
    return Data;
}

void XRegistration_accumulate_kernel_Set_jtj_out(XRegistration_accumulate_kernel *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRegistration_accumulate_kernel_WriteReg(InstancePtr->Control_r_BaseAddress, XREGISTRATION_ACCUMULATE_KERNEL_CONTROL_R_ADDR_JTJ_OUT_DATA, (u32)(Data));
    XRegistration_accumulate_kernel_WriteReg(InstancePtr->Control_r_BaseAddress, XREGISTRATION_ACCUMULATE_KERNEL_CONTROL_R_ADDR_JTJ_OUT_DATA + 4, (u32)(Data >> 32));
}

u64 XRegistration_accumulate_kernel_Get_jtj_out(XRegistration_accumulate_kernel *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XRegistration_accumulate_kernel_ReadReg(InstancePtr->Control_r_BaseAddress, XREGISTRATION_ACCUMULATE_KERNEL_CONTROL_R_ADDR_JTJ_OUT_DATA);
    Data += (u64)XRegistration_accumulate_kernel_ReadReg(InstancePtr->Control_r_BaseAddress, XREGISTRATION_ACCUMULATE_KERNEL_CONTROL_R_ADDR_JTJ_OUT_DATA + 4) << 32;
    return Data;
}

void XRegistration_accumulate_kernel_Set_jtr_out(XRegistration_accumulate_kernel *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRegistration_accumulate_kernel_WriteReg(InstancePtr->Control_r_BaseAddress, XREGISTRATION_ACCUMULATE_KERNEL_CONTROL_R_ADDR_JTR_OUT_DATA, (u32)(Data));
    XRegistration_accumulate_kernel_WriteReg(InstancePtr->Control_r_BaseAddress, XREGISTRATION_ACCUMULATE_KERNEL_CONTROL_R_ADDR_JTR_OUT_DATA + 4, (u32)(Data >> 32));
}

u64 XRegistration_accumulate_kernel_Get_jtr_out(XRegistration_accumulate_kernel *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XRegistration_accumulate_kernel_ReadReg(InstancePtr->Control_r_BaseAddress, XREGISTRATION_ACCUMULATE_KERNEL_CONTROL_R_ADDR_JTR_OUT_DATA);
    Data += (u64)XRegistration_accumulate_kernel_ReadReg(InstancePtr->Control_r_BaseAddress, XREGISTRATION_ACCUMULATE_KERNEL_CONTROL_R_ADDR_JTR_OUT_DATA + 4) << 32;
    return Data;
}

void XRegistration_accumulate_kernel_Set_used_count(XRegistration_accumulate_kernel *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRegistration_accumulate_kernel_WriteReg(InstancePtr->Control_r_BaseAddress, XREGISTRATION_ACCUMULATE_KERNEL_CONTROL_R_ADDR_USED_COUNT_DATA, (u32)(Data));
    XRegistration_accumulate_kernel_WriteReg(InstancePtr->Control_r_BaseAddress, XREGISTRATION_ACCUMULATE_KERNEL_CONTROL_R_ADDR_USED_COUNT_DATA + 4, (u32)(Data >> 32));
}

u64 XRegistration_accumulate_kernel_Get_used_count(XRegistration_accumulate_kernel *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XRegistration_accumulate_kernel_ReadReg(InstancePtr->Control_r_BaseAddress, XREGISTRATION_ACCUMULATE_KERNEL_CONTROL_R_ADDR_USED_COUNT_DATA);
    Data += (u64)XRegistration_accumulate_kernel_ReadReg(InstancePtr->Control_r_BaseAddress, XREGISTRATION_ACCUMULATE_KERNEL_CONTROL_R_ADDR_USED_COUNT_DATA + 4) << 32;
    return Data;
}

void XRegistration_accumulate_kernel_Set_dropped_count(XRegistration_accumulate_kernel *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRegistration_accumulate_kernel_WriteReg(InstancePtr->Control_r_BaseAddress, XREGISTRATION_ACCUMULATE_KERNEL_CONTROL_R_ADDR_DROPPED_COUNT_DATA, (u32)(Data));
    XRegistration_accumulate_kernel_WriteReg(InstancePtr->Control_r_BaseAddress, XREGISTRATION_ACCUMULATE_KERNEL_CONTROL_R_ADDR_DROPPED_COUNT_DATA + 4, (u32)(Data >> 32));
}

u64 XRegistration_accumulate_kernel_Get_dropped_count(XRegistration_accumulate_kernel *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XRegistration_accumulate_kernel_ReadReg(InstancePtr->Control_r_BaseAddress, XREGISTRATION_ACCUMULATE_KERNEL_CONTROL_R_ADDR_DROPPED_COUNT_DATA);
    Data += (u64)XRegistration_accumulate_kernel_ReadReg(InstancePtr->Control_r_BaseAddress, XREGISTRATION_ACCUMULATE_KERNEL_CONTROL_R_ADDR_DROPPED_COUNT_DATA + 4) << 32;
    return Data;
}

void XRegistration_accumulate_kernel_Set_correspondence_count(XRegistration_accumulate_kernel *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRegistration_accumulate_kernel_WriteReg(InstancePtr->Control_BaseAddress, XREGISTRATION_ACCUMULATE_KERNEL_CONTROL_ADDR_CORRESPONDENCE_COUNT_DATA, Data);
}

u32 XRegistration_accumulate_kernel_Get_correspondence_count(XRegistration_accumulate_kernel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XRegistration_accumulate_kernel_ReadReg(InstancePtr->Control_BaseAddress, XREGISTRATION_ACCUMULATE_KERNEL_CONTROL_ADDR_CORRESPONDENCE_COUNT_DATA);
    return Data;
}

void XRegistration_accumulate_kernel_Set_kernel(XRegistration_accumulate_kernel *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRegistration_accumulate_kernel_WriteReg(InstancePtr->Control_BaseAddress, XREGISTRATION_ACCUMULATE_KERNEL_CONTROL_ADDR_KERNEL_DATA, Data);
}

u32 XRegistration_accumulate_kernel_Get_kernel(XRegistration_accumulate_kernel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XRegistration_accumulate_kernel_ReadReg(InstancePtr->Control_BaseAddress, XREGISTRATION_ACCUMULATE_KERNEL_CONTROL_ADDR_KERNEL_DATA);
    return Data;
}

void XRegistration_accumulate_kernel_InterruptGlobalEnable(XRegistration_accumulate_kernel *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRegistration_accumulate_kernel_WriteReg(InstancePtr->Control_BaseAddress, XREGISTRATION_ACCUMULATE_KERNEL_CONTROL_ADDR_GIE, 1);
}

void XRegistration_accumulate_kernel_InterruptGlobalDisable(XRegistration_accumulate_kernel *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRegistration_accumulate_kernel_WriteReg(InstancePtr->Control_BaseAddress, XREGISTRATION_ACCUMULATE_KERNEL_CONTROL_ADDR_GIE, 0);
}

void XRegistration_accumulate_kernel_InterruptEnable(XRegistration_accumulate_kernel *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XRegistration_accumulate_kernel_ReadReg(InstancePtr->Control_BaseAddress, XREGISTRATION_ACCUMULATE_KERNEL_CONTROL_ADDR_IER);
    XRegistration_accumulate_kernel_WriteReg(InstancePtr->Control_BaseAddress, XREGISTRATION_ACCUMULATE_KERNEL_CONTROL_ADDR_IER, Register | Mask);
}

void XRegistration_accumulate_kernel_InterruptDisable(XRegistration_accumulate_kernel *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XRegistration_accumulate_kernel_ReadReg(InstancePtr->Control_BaseAddress, XREGISTRATION_ACCUMULATE_KERNEL_CONTROL_ADDR_IER);
    XRegistration_accumulate_kernel_WriteReg(InstancePtr->Control_BaseAddress, XREGISTRATION_ACCUMULATE_KERNEL_CONTROL_ADDR_IER, Register & (~Mask));
}

void XRegistration_accumulate_kernel_InterruptClear(XRegistration_accumulate_kernel *InstancePtr, u32 Mask) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRegistration_accumulate_kernel_WriteReg(InstancePtr->Control_BaseAddress, XREGISTRATION_ACCUMULATE_KERNEL_CONTROL_ADDR_ISR, Mask);
}

u32 XRegistration_accumulate_kernel_InterruptGetEnabled(XRegistration_accumulate_kernel *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XRegistration_accumulate_kernel_ReadReg(InstancePtr->Control_BaseAddress, XREGISTRATION_ACCUMULATE_KERNEL_CONTROL_ADDR_IER);
}

u32 XRegistration_accumulate_kernel_InterruptGetStatus(XRegistration_accumulate_kernel *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XRegistration_accumulate_kernel_ReadReg(InstancePtr->Control_BaseAddress, XREGISTRATION_ACCUMULATE_KERNEL_CONTROL_ADDR_ISR);
}

