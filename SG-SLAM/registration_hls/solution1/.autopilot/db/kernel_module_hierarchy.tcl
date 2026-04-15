set ModuleHierarchy {[{
"Name" : "registration_accumulate_kernel","ID" : "0","Type" : "sequential",
"SubInsts" : [
	{"Name" : "grp_registration_accumulate_kernel_Pipeline_VITIS_LOOP_70_3_fu_406","ID" : "1","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "VITIS_LOOP_70_3","ID" : "2","Type" : "pipeline"},]},
	{"Name" : "grp_registration_accumulate_kernel_Pipeline_VITIS_LOOP_151_11_fu_480","ID" : "3","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "VITIS_LOOP_151_11","ID" : "4","Type" : "pipeline"},]},],
"SubLoops" : [
	{"Name" : "VITIS_LOOP_140_9","ID" : "5","Type" : "no",
	"SubInsts" : [
	{"Name" : "grp_registration_accumulate_kernel_Pipeline_VITIS_LOOP_141_10_fu_448","ID" : "6","Type" : "sequential",
			"SubLoops" : [
			{"Name" : "VITIS_LOOP_141_10","ID" : "7","Type" : "pipeline"},]},]},]
}]}