SELF_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
MLIR_TOOLS_DIR := $(abspath $(SELF_DIR)/../..)
REPO_ROOT := $(abspath $(MLIR_TOOLS_DIR)/../..)

.DEFAULT_GOAL := saturn-vfrope-test2-compile

NPU_BIN ?= $(MLIR_TOOLS_DIR)/build/bin
NPU_OPT := $(NPU_BIN)/npu-opt
NPU_TRANSLATE := $(NPU_BIN)/npu-translate
NPU_LLC := $(NPU_BIN)/npu-llc

TORCH_FRONTEND_DIR := $(REPO_ROOT)/toolchains/torch-mlir-vfrope
FRONTEND_BUILD := $(TORCH_FRONTEND_DIR)/build
FRONTEND_PYTHON := $(TORCH_FRONTEND_DIR)/.deps/venv/bin/python
TORCH_MLIR_PYTHONPATH := $(FRONTEND_BUILD)/tools/torch-mlir/python_packages/torch_mlir
FRONTEND_READY := $(FRONTEND_BUILD)/.tvela-vfrope-linalg-ready

RISCV_BIN ?= $(REPO_ROOT)/.conda-env/riscv-tools/bin
RISCV_PREFIX ?= riscv64-unknown-elf-
CC := $(RISCV_BIN)/$(RISCV_PREFIX)gcc
OBJDUMP := $(RISCV_BIN)/$(RISCV_PREFIX)objdump
READELF := $(RISCV_BIN)/$(RISCV_PREFIX)readelf
NM := $(RISCV_BIN)/$(RISCV_PREFIX)nm

BUILD_DIR := $(SELF_DIR)/build
LINALG_MLIR := $(BUILD_DIR)/vfrope_test2_linalg.mlir
LOWERED_MLIR := $(BUILD_DIR)/vfrope_test2_llvm.mlir
LLVM_IR := $(BUILD_DIR)/vfrope_test2.ll
MLIR_OBJECT := $(BUILD_DIR)/vfrope_test2_mlir.o
RUNTIME_OBJECT := $(BUILD_DIR)/vfrope_runtime.o
ELF := $(BUILD_DIR)/vfrope_test2.elf

ARCH_FLAGS := -march=rv64gcv -mabi=lp64d -mcmodel=medany
ASFLAGS := $(ARCH_FLAGS)
LDFLAGS := $(ARCH_FLAGS) -static -specs=htif_nano.specs
LLVM_FEATURES := +v,+m,+a,+f,+d,+c,+zicsr,+zifencei

.PHONY: saturn-vfrope-test2-frontend-setup \
	saturn-vfrope-test2-export \
	saturn-vfrope-test2-lower \
	saturn-vfrope-test2-translate \
	saturn-vfrope-test2-compile \
	saturn-vfrope-test2-check \
	saturn-vfrope-test2 \
	frontend-setup \
	all \
	check \
	clean

all: saturn-vfrope-test2-compile

saturn-vfrope-test2: saturn-vfrope-test2-compile

frontend-setup: saturn-vfrope-test2-frontend-setup

check: saturn-vfrope-test2-check

saturn-vfrope-test2-frontend-setup: $(FRONTEND_READY)

saturn-vfrope-test2-export: $(LINALG_MLIR)

saturn-vfrope-test2-lower: $(LOWERED_MLIR)

saturn-vfrope-test2-translate: $(LLVM_IR)

saturn-vfrope-test2-compile: $(ELF)

$(FRONTEND_READY): $(TORCH_FRONTEND_DIR)/setup.sh \
		$(TORCH_FRONTEND_DIR)/versions.lock \
		$(TORCH_FRONTEND_DIR)/patches/0001-preserve-tvela-vfrope-in-linalg.patch
	$(TORCH_FRONTEND_DIR)/setup.sh

$(LINALG_MLIR): vfrope_test2.py $(FRONTEND_READY) | $(BUILD_DIR)
	PYTHONPATH=$(TORCH_MLIR_PYTHONPATH) \
		$(FRONTEND_PYTHON) $< --output $@

$(LOWERED_MLIR): $(LINALG_MLIR) | $(BUILD_DIR)
	$(NPU_OPT) $< \
		--one-shot-bufferize='bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map' \
		--buffer-results-to-out-params \
		--buffer-deallocation-pipeline \
		--lower-tvela-to-fvela-runtime \
		--convert-linalg-to-loops \
		--lower-affine \
		--convert-scf-to-cf \
		--expand-strided-metadata \
		--finalize-memref-to-llvm \
		--convert-arith-to-llvm \
		--convert-index-to-llvm \
		--convert-cf-to-llvm \
		--convert-func-to-llvm \
		--reconcile-unrealized-casts \
		-o $@

$(LLVM_IR): $(LOWERED_MLIR)
	$(NPU_TRANSLATE) $< --npu-to-llvmir -o $@

$(MLIR_OBJECT): $(LLVM_IR)
	$(NPU_LLC) $< \
		-filetype=obj \
		-mtriple=riscv64 \
		-code-model=medium \
		-mattr=$(LLVM_FEATURES) \
		-float-abi=hard \
		-o $@

$(RUNTIME_OBJECT): vfrope_runtime.S | $(BUILD_DIR)
	$(CC) $(ASFLAGS) -c $< -o $@

$(ELF): $(MLIR_OBJECT) $(RUNTIME_OBJECT)
	$(CC) $(LDFLAGS) $^ -o $@

$(BUILD_DIR):
	mkdir -p $@

saturn-vfrope-test2-check: saturn-vfrope-test2-compile
	PYTHONPATH=$(TORCH_MLIR_PYTHONPATH) \
		$(FRONTEND_PYTHON) $(TORCH_FRONTEND_DIR)/test/test_vfrope_linalg.py
	test "$$(grep -c 'tvela.vfrope_q15_chunk' $(LINALG_MLIR))" -eq 1
	test "$$(grep -c 'arith.constant dense<' $(LINALG_MLIR))" -eq 2
	grep -q 'func.func @main() -> i32' $(LINALG_MLIR)
	grep -q 'func.func @vfrope_self_test' $(LINALG_MLIR)
	! grep -q 'torch\.' $(LINALG_MLIR)
	! grep -q 'dense_resource' $(LINALG_MLIR)
	! grep -q 'llvm.emit_c_interface' $(LINALG_MLIR)
	! grep -q 'tvela\.' $(LOWERED_MLIR)
	! grep -q 'linalg\.' $(LOWERED_MLIR)
	$(READELF) -h $(ELF) | grep -q 'Machine:.*RISC-V'
	$(NM) $(ELF) | grep -Eq '[[:space:]]T main$$'
	$(NM) $(ELF) | grep -Eq '[[:space:]]T vfrope_self_test$$'
	$(NM) $(ELF) | grep -Eq '[[:space:]]T fvela_rope_runtime$$'
	! $(NM) $(ELF) | grep -q '_mlir_ciface'
	$(OBJDUMP) -s -j .rodata $(ELF) | grep -qi '00200040 0060ff7f 00e000c0 00a00080'
	$(OBJDUMP) -s -j .rodata $(ELF) | grep -qi '05b94af6 077e7e63 94be1b1d 5196cc77'
	$(OBJDUMP) -d $(ELF) | grep -Eq 'vsetvli|vsetivli'
	$(OBJDUMP) -d $(ELF) | grep -q 'vle16.v'
	$(OBJDUMP) -d $(ELF) | grep -qi '4a21c1d7'
	$(OBJDUMP) -d $(ELF) | grep -q 'vse16.v'
	@echo "Verified $(ELF)"

clean:
	rm -rf $(BUILD_DIR)
