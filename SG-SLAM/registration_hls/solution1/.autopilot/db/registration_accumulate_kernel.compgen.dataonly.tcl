# This script segment is generated automatically by AutoPilot

set axilite_register_dict [dict create]
set port_control {
correspondence_count { 
	dir I
	width 32
	depth 1
	mode ap_none
	offset 16
	offset_end 23
}
kernel { 
	dir I
	width 32
	depth 1
	mode ap_none
	offset 24
	offset_end 31
}
ap_start { }
ap_done { }
ap_ready { }
ap_idle { }
interrupt {
}
}
dict set axilite_register_dict control $port_control


set port_control_r {
src_xyz { 
	dir I
	width 64
	depth 1
	mode ap_none
	offset 16
	offset_end 27
}
tgt_xyz { 
	dir I
	width 64
	depth 1
	mode ap_none
	offset 28
	offset_end 39
}
labels { 
	dir I
	width 64
	depth 1
	mode ap_none
	offset 40
	offset_end 51
}
jtj_out { 
	dir I
	width 64
	depth 1
	mode ap_none
	offset 52
	offset_end 63
}
jtr_out { 
	dir I
	width 64
	depth 1
	mode ap_none
	offset 64
	offset_end 75
}
used_count { 
	dir I
	width 64
	depth 1
	mode ap_none
	offset 76
	offset_end 87
}
dropped_count { 
	dir I
	width 64
	depth 1
	mode ap_none
	offset 88
	offset_end 99
}
}
dict set axilite_register_dict control_r $port_control_r


