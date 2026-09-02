SELF_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
MLIR_TOOLS_DIR := $(abspath $(SELF_DIR)/../..)
REPO_ROOT := $(abspath $(MLIR_TOOLS_DIR)/../..)

.DEFAULT_GOAL := fvela-npu-ternary-compile

NPU_BIN ?= $(MLIR_TOOLS_DIR)/build/bin
NPU_OPT := $(NPU_BIN)/npu-opt
NPU_TRANSLATE := $(NPU_BIN)/npu-translate
NPU_LLC := $(NPU_BIN)/npu-llc

TORCH_FRONTEND_DIR := $(REPO_ROOT)/toolchains/torch-mlir-fvela
FRONTEND_BUILD := $(TORCH_FRONTEND_DIR)/build
FRONTEND_PYTHON := $(TORCH_FRONTEND_DIR)/.deps/venv/bin/python
TORCH_MLIR_PYTHONPATH := $(FRONTEND_BUILD)/tools/torch-mlir/python_packages/torch_mlir
FRONTEND_READY := $(FRONTEND_BUILD)/.tvela-fvela-linalg-ready

RISCV_BIN ?= $(REPO_ROOT)/.conda-env/riscv-tools/bin
RISCV_PREFIX ?= riscv64-unknown-elf-
CC := $(RISCV_BIN)/$(RISCV_PREFIX)gcc
OBJDUMP := $(RISCV_BIN)/$(RISCV_PREFIX)objdump
READELF := $(RISCV_BIN)/$(RISCV_PREFIX)readelf
NM := $(RISCV_BIN)/$(RISCV_PREFIX)nm

BUILD_DIR := $(SELF_DIR)/build
LINALG_MLIR := $(BUILD_DIR)/ternary_matmul_linalg.mlir
LOWERED_MLIR := $(BUILD_DIR)/ternary_matmul_llvm.mlir
LLVM_IR := $(BUILD_DIR)/ternary_matmul.ll
OBJECT := $(BUILD_DIR)/ternary_matmul.o
ELF := $(BUILD_DIR)/ternary_matmul.elf

ARCH_FLAGS := -march=rv64gc -mabi=lp64d -mcmodel=medany
LDFLAGS := $(ARCH_FLAGS) -static -specs=htif_nano.specs
LLVM_FEATURES := +npuext,+m,+a,+f,+d,+c,+zicsr,+zifencei

.PHONY: fvela-npu-ternary-frontend-setup \
	fvela-npu-ternary-export \
	fvela-npu-ternary-lower \
	fvela-npu-ternary-translate \
	fvela-npu-ternary-compile \
	fvela-npu-ternary-check \
	frontend-setup \
	all \
	check \
	clean

all: fvela-npu-ternary-compile

frontend-setup: fvela-npu-ternary-frontend-setup

check: fvela-npu-ternary-check

fvela-npu-ternary-frontend-setup: $(FRONTEND_READY)

fvela-npu-ternary-export: $(LINALG_MLIR)

fvela-npu-ternary-lower: $(LOWERED_MLIR)

fvela-npu-ternary-translate: $(LLVM_IR)

fvela-npu-ternary-compile: $(ELF)

$(FRONTEND_READY): $(TORCH_FRONTEND_DIR)/setup.sh \
		$(TORCH_FRONTEND_DIR)/versions.lock \
		$(TORCH_FRONTEND_DIR)/patches/0001-preserve-tvela-ops-in-linalg.patch
	$(TORCH_FRONTEND_DIR)/setup.sh

$(LINALG_MLIR): ternary_matmul.py $(FRONTEND_READY) | $(BUILD_DIR)
	PYTHONPATH=$(TORCH_MLIR_PYTHONPATH) \
		$(FRONTEND_PYTHON) $< --output $@

$(LOWERED_MLIR): $(LINALG_MLIR) | $(BUILD_DIR)
	$(NPU_OPT) $< \
		--one-shot-bufferize='bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map' \
		--buffer-results-to-out-params \
		--buffer-deallocation-pipeline \
		--convert-linalg-to-gemmini \
		--convert-linalg-to-loops \
		--lower-gemmini='dim=16 addr_len=32 acc_rows=1024 bank_rows=4096 elem_t=i8 acc_t=i32' \
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

$(OBJECT): $(LLVM_IR)
	$(NPU_LLC) $< \
		-filetype=obj \
		-mtriple=riscv64 \
		-code-model=medium \
		-mattr=$(LLVM_FEATURES) \
		-float-abi=hard \
		-o $@

$(ELF): $(OBJECT)
	$(CC) $(LDFLAGS) $^ -o $@

$(BUILD_DIR):
	mkdir -p $@

fvela-npu-ternary-check: fvela-npu-ternary-compile
	PYTHONPATH=$(TORCH_MLIR_PYTHONPATH) \
		$(FRONTEND_PYTHON) $(TORCH_FRONTEND_DIR)/test/test_ternary_matmul_linalg.py
	test "$$(grep -c 'tvela.ternary_matmul' $(LINALG_MLIR))" -eq 1
	test "$$(grep -c 'arith.constant dense<' $(LINALG_MLIR))" -eq 3
	grep -q 'func.func @main() -> i32' $(LINALG_MLIR)
	grep -q 'func.func @ternary_matmul_self_test' $(LINALG_MLIR)
	! grep -q 'torch\.' $(LINALG_MLIR)
	! grep -q 'tvela\.' $(LOWERED_MLIR)
	! grep -q 'gemmini\.tile_matmul' $(LOWERED_MLIR)
	! grep -q 'linalg\.' $(LOWERED_MLIR)
	grep -q '@llvm.riscv.loop.ws(i64 1048576, i64 0)' $(LLVM_IR)
	$(READELF) -h $(ELF) | grep -q 'Machine:.*RISC-V'
	$(NM) $(ELF) | grep -Eq '[[:space:]]T main$$'
	$(NM) $(ELF) | grep -Eq '[[:space:]]T ternary_matmul_self_test$$'
	$(OBJDUMP) -d $(ELF) | grep -qi '1005b07b'
	@echo "Verified $(ELF)"

clean:
	rm -rf $(BUILD_DIR)
