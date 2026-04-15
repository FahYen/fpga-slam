// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2024.1 (64-bit)
// Tool Version Limit: 2024.05
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
#ifndef __linux__

#include "xstatus.h"
#ifdef SDT
#include "xparameters.h"
#endif
#include "xregistration_accumulate_kernel.h"

extern XRegistration_accumulate_kernel_Config XRegistration_accumulate_kernel_ConfigTable[];

#ifdef SDT
XRegistration_accumulate_kernel_Config *XRegistration_accumulate_kernel_LookupConfig(UINTPTR BaseAddress) {
	XRegistration_accumulate_kernel_Config *ConfigPtr = NULL;

	int Index;

	for (Index = (u32)0x0; XRegistration_accumulate_kernel_ConfigTable[Index].Name != NULL; Index++) {
		if (!BaseAddress || XRegistration_accumulate_kernel_ConfigTable[Index].Control_r_BaseAddress == BaseAddress) {
			ConfigPtr = &XRegistration_accumulate_kernel_ConfigTable[Index];
			break;
		}
	}

	return ConfigPtr;
}

int XRegistration_accumulate_kernel_Initialize(XRegistration_accumulate_kernel *InstancePtr, UINTPTR BaseAddress) {
	XRegistration_accumulate_kernel_Config *ConfigPtr;

	Xil_AssertNonvoid(InstancePtr != NULL);

	ConfigPtr = XRegistration_accumulate_kernel_LookupConfig(BaseAddress);
	if (ConfigPtr == NULL) {
		InstancePtr->IsReady = 0;
		return (XST_DEVICE_NOT_FOUND);
	}

	return XRegistration_accumulate_kernel_CfgInitialize(InstancePtr, ConfigPtr);
}
#else
XRegistration_accumulate_kernel_Config *XRegistration_accumulate_kernel_LookupConfig(u16 DeviceId) {
	XRegistration_accumulate_kernel_Config *ConfigPtr = NULL;

	int Index;

	for (Index = 0; Index < XPAR_XREGISTRATION_ACCUMULATE_KERNEL_NUM_INSTANCES; Index++) {
		if (XRegistration_accumulate_kernel_ConfigTable[Index].DeviceId == DeviceId) {
			ConfigPtr = &XRegistration_accumulate_kernel_ConfigTable[Index];
			break;
		}
	}

	return ConfigPtr;
}

int XRegistration_accumulate_kernel_Initialize(XRegistration_accumulate_kernel *InstancePtr, u16 DeviceId) {
	XRegistration_accumulate_kernel_Config *ConfigPtr;

	Xil_AssertNonvoid(InstancePtr != NULL);

	ConfigPtr = XRegistration_accumulate_kernel_LookupConfig(DeviceId);
	if (ConfigPtr == NULL) {
		InstancePtr->IsReady = 0;
		return (XST_DEVICE_NOT_FOUND);
	}

	return XRegistration_accumulate_kernel_CfgInitialize(InstancePtr, ConfigPtr);
}
#endif

#endif

