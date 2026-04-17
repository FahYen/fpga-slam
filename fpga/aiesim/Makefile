# =============================================================================
# Makefile for AIE x86 functional simulation
#
# Prerequisites:
#   - Vitis 2023.2+ installed and sourced (source /tools/Xilinx/Vitis/2023.2/settings64.sh)
#   - PLIO data files generated in data/ (run gen_plio_data.py first)
#
# Usage:
#   make x86compile    # Compile graph for x86 simulation
#   make x86sim        # Run x86 simulation
#   make verify        # Compare output against golden
#   make clean         # Remove build artifacts
#   make all           # compile + simulate + verify
# =============================================================================

# Paths
AIE_SRC_DIR    = ..
GRAPH_FILE     = $(AIE_SRC_DIR)/AIE_graph.cpp
KERNEL_FILE    = $(AIE_SRC_DIR)/conv_kernel.cpp

# Platform (not needed for x86sim but aiecompiler may want it)
# Set to a Versal platform if available, otherwise x86sim doesn't strictly need it
PLATFORM       ?= 

# Build directory
BUILD_DIR      = build_x86
WORK_DIR       = $(BUILD_DIR)/Work

# Compiler flags
AIE_FLAGS      = --target=x86sim
AIE_FLAGS     += --include=$(AIE_SRC_DIR)
AIE_FLAGS     += --workdir=$(WORK_DIR)
AIE_FLAGS     += --verbose
ifdef PLATFORM
AIE_FLAGS     += --platform=$(PLATFORM)
endif

# x86sim flags
SIM_FLAGS      = --pkg-dir=$(WORK_DIR)

.PHONY: all x86compile x86sim verify clean help

all: x86compile x86sim verify

help:
	@echo "AIE x86 Functional Simulation"
	@echo ""
	@echo "  1. Generate PLIO data:  python3 gen_plio_data.py --golden-dir <path> --gemm-dir <path>"
	@echo "  2. Compile:             make x86compile"
	@echo "  3. Simulate:            make x86sim"
	@echo "  4. Verify:              make verify"
	@echo ""
	@echo "Targets:"
	@echo "  x86compile  - Compile AIE graph for x86 simulation"
	@echo "  x86sim      - Run x86 functional simulation"
	@echo "  verify      - Compare simulation output against golden"
	@echo "  clean       - Remove build artifacts"

# Step 1: Compile the AIE graph for x86 simulation
x86compile: $(GRAPH_FILE) $(KERNEL_FILE)
	@echo "==== Compiling AIE graph for x86 simulation ===="
	@mkdir -p $(BUILD_DIR)
	aiecompiler $(AIE_FLAGS) $(GRAPH_FILE)
	@echo "==== Compilation complete ===="

# Step 2: Run x86 simulation
x86sim: $(WORK_DIR)
	@echo "==== Running x86 simulation ===="
	cd $(BUILD_DIR) && x86simulator $(SIM_FLAGS)
	@echo "==== Simulation complete ===="
	@echo "Output files should be in $(BUILD_DIR)/x86simulator_output/"

# Step 3: Verify output against golden
verify:
	@echo "==== Verifying simulation output ===="
	python3 verify_x86sim.py \
		--sim-output $(BUILD_DIR)/x86simulator_output/data/out_3x3.txt \
		--expected data/expected_out.txt \
		--tile-meta data/tile_meta.json
	@echo "==== Verification complete ===="

clean:
	rm -rf $(BUILD_DIR)
	@echo "Cleaned build artifacts"
