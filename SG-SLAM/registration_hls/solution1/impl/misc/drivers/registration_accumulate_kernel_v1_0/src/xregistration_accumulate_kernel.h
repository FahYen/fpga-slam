// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2024.1 (64-bit)
// Tool Version Limit: 2024.05
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
#ifndef XREGISTRATION_ACCUMULATE_KERNEL_H
#define XREGISTRATION_ACCUMULATE_KERNEL_H

#ifdef __cplusplus
extern "C" {
#endif

/***************************** Include Files *********************************/
#ifndef __linux__
#include "xil_types.h"
#include "xil_assert.h"
#include "xstatus.h"
#include "xil_io.h"
#else
#include <stdint.h>
#include <assert.h>
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stddef.h>
#endif
#include "xregistration_accumulate_kernel_hw.h"

/**************************** Type Definitions ******************************/
#ifdef __linux__
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
#else
typedef struct {
#ifdef SDT
    char *Name;
#else
    u16 DeviceId;
#endif
    u64 Control_r_BaseAddress;
    u64 Control_BaseAddress;
} XRegistration_accumulate_kernel_Config;
#endif

typedef struct {
    u64 Control_r_BaseAddress;
    u64 Control_BaseAddress;
    u32 IsReady;
} XRegistration_accumulate_kernel;

typedef u32 word_type;

/***************** Macros (Inline Functions) Definitions *********************/
#ifndef __linux__
#define XRegistration_accumulate_kernel_WriteReg(BaseAddress, RegOffset, Data) \
    Xil_Out32((BaseAddress) + (RegOffset), (u32)(Data))
#define XRegistration_accumulate_kernel_ReadReg(BaseAddress, RegOffset) \
    Xil_In32((BaseAddress) + (RegOffset))
#else
#define XRegistration_accumulate_kernel_WriteReg(BaseAddress, RegOffset, Data) \
    *(volatile u32*)((BaseAddress) + (RegOffset)) = (u32)(Data)
#define XRegistration_accumulate_kernel_ReadReg(BaseAddress, RegOffset) \
    *(volatile u32*)((BaseAddress) + (RegOffset))

#define Xil_AssertVoid(expr)    assert(expr)
#define Xil_AssertNonvoid(expr) assert(expr)

#define XST_SUCCESS             0
#define XST_DEVICE_NOT_FOUND    2
#define XST_OPEN_DEVICE_FAILED  3
#define XIL_COMPONENT_IS_READY  1
#endif

/************************** Function Prototypes *****************************/
#ifndef __linux__
#ifdef SDT
int XRegistration_accumulate_kernel_Initialize(XRegistration_accumulate_kernel *InstancePtr, UINTPTR BaseAddress);
XRegistration_accumulate_kernel_Config* XRegistration_accumulate_kernel_LookupConfig(UINTPTR BaseAddress);
#else
int XRegistration_accumulate_kernel_Initialize(XRegistration_accumulate_kernel *InstancePtr, u16 DeviceId);
XRegistration_accumulate_kernel_Config* XRegistration_accumulate_kernel_LookupConfig(u16 DeviceId);
#endif
int XRegistration_accumulate_kernel_CfgInitialize(XRegistration_accumulate_kernel *InstancePtr, XRegistration_accumulate_kernel_Config *ConfigPtr);
#else
int XRegistration_accumulate_kernel_Initialize(XRegistration_accumulate_kernel *InstancePtr, const char* InstanceName);
int XRegistration_accumulate_kernel_Release(XRegistration_accumulate_kernel *InstancePtr);
#endif

void XRegistration_accumulate_kernel_Start(XRegistration_accumulate_kernel *InstancePtr);
u32 XRegistration_accumulate_kernel_IsDone(XRegistration_accumulate_kernel *InstancePtr);
u32 XRegistration_accumulate_kernel_IsIdle(XRegistration_accumulate_kernel *InstancePtr);
u32 XRegistration_accumulate_kernel_IsReady(XRegistration_accumulate_kernel *InstancePtr);
void XRegistration_accumulate_kernel_EnableAutoRestart(XRegistration_accumulate_kernel *InstancePtr);
void XRegistration_accumulate_kernel_DisableAutoRestart(XRegistration_accumulate_kernel *InstancePtr);

void XRegistration_accumulate_kernel_Set_src_xyz(XRegistration_accumulate_kernel *InstancePtr, u64 Data);
u64 XRegistration_accumulate_kernel_Get_src_xyz(XRegistration_accumulate_kernel *InstancePtr);
void XRegistration_accumulate_kernel_Set_tgt_xyz(XRegistration_accumulate_kernel *InstancePtr, u64 Data);
u64 XRegistration_accumulate_kernel_Get_tgt_xyz(XRegistration_accumulate_kernel *InstancePtr);
void XRegistration_accumulate_kernel_Set_labels(XRegistration_accumulate_kernel *InstancePtr, u64 Data);
u64 XRegistration_accumulate_kernel_Get_labels(XRegistration_accumulate_kernel *InstancePtr);
void XRegistration_accumulate_kernel_Set_jtj_out(XRegistration_accumulate_kernel *InstancePtr, u64 Data);
u64 XRegistration_accumulate_kernel_Get_jtj_out(XRegistration_accumulate_kernel *InstancePtr);
void XRegistration_accumulate_kernel_Set_jtr_out(XRegistration_accumulate_kernel *InstancePtr, u64 Data);
u64 XRegistration_accumulate_kernel_Get_jtr_out(XRegistration_accumulate_kernel *InstancePtr);
void XRegistration_accumulate_kernel_Set_used_count(XRegistration_accumulate_kernel *InstancePtr, u64 Data);
u64 XRegistration_accumulate_kernel_Get_used_count(XRegistration_accumulate_kernel *InstancePtr);
void XRegistration_accumulate_kernel_Set_dropped_count(XRegistration_accumulate_kernel *InstancePtr, u64 Data);
u64 XRegistration_accumulate_kernel_Get_dropped_count(XRegistration_accumulate_kernel *InstancePtr);
void XRegistration_accumulate_kernel_Set_correspondence_count(XRegistration_accumulate_kernel *InstancePtr, u32 Data);
u32 XRegistration_accumulate_kernel_Get_correspondence_count(XRegistration_accumulate_kernel *InstancePtr);
void XRegistration_accumulate_kernel_Set_kernel(XRegistration_accumulate_kernel *InstancePtr, u32 Data);
u32 XRegistration_accumulate_kernel_Get_kernel(XRegistration_accumulate_kernel *InstancePtr);

void XRegistration_accumulate_kernel_InterruptGlobalEnable(XRegistration_accumulate_kernel *InstancePtr);
void XRegistration_accumulate_kernel_InterruptGlobalDisable(XRegistration_accumulate_kernel *InstancePtr);
void XRegistration_accumulate_kernel_InterruptEnable(XRegistration_accumulate_kernel *InstancePtr, u32 Mask);
void XRegistration_accumulate_kernel_InterruptDisable(XRegistration_accumulate_kernel *InstancePtr, u32 Mask);
void XRegistration_accumulate_kernel_InterruptClear(XRegistration_accumulate_kernel *InstancePtr, u32 Mask);
u32 XRegistration_accumulate_kernel_InterruptGetEnabled(XRegistration_accumulate_kernel *InstancePtr);
u32 XRegistration_accumulate_kernel_InterruptGetStatus(XRegistration_accumulate_kernel *InstancePtr);

#ifdef __cplusplus
}
#endif

#endif
