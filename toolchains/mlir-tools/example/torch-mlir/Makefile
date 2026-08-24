NPU_BUILD ?= ../../build
NPU_OPT := $(NPU_BUILD)/bin/npu-opt
NPU_TRANSLATE := $(NPU_BUILD)/bin/npu-translate
NPU_LLC := $(NPU_BUILD)/bin/npu-llc
INTERFACES := ../../midend/Interfaces

PYTHON ?= python
# Matches the other examples in ../makefile. The newlib toolchain
# (CROSS_PREFIX=riscv64-unknown-elf-) works too.
CROSS_PREFIX ?= riscv64-unknown-linux-gnu-
CROSS_CXX ?= $(CROSS_PREFIX)g++
CROSS_OBJCOPY ?= $(CROSS_PREFIX)objcopy
SPIKE ?= spike
PK ?= pk
SPIKE_OPTIONS ?= --extension=gemmini

# Element type of the matmul result. Gemmini has one C move-out format per
# type: i8 narrows the accumulator (fullC = false), i32 moves it out unchanged
# (fullC = true). Inputs are int8 either way.
DTYPE ?= i8
ifeq ($(filter $(DTYPE),i8 i32),)
  $(error DTYPE must be i8 or i32, got '$(DTYPE)')
endif
ifeq ($(DTYPE),i32)
  RUNNER_FLAGS := -DRESULT_I32
else
  RUNNER_FLAGS :=
endif
# Fold the i8 truncation into Gemmini's C move-out. That move-out clamps to
# [-128, 127] while arith.trunci keeps the low 8 bits, so the two agree only
# while the accumulator stays in the int8 range. Off by default; no effect on
# DTYPE=i32, which has no truncation to fold.
FUSE_TRUNCATION ?= 0
ifeq ($(FUSE_TRUNCATION),1)
  GEMMINI_CONVERT := -convert-linalg-to-gemmini="fuse-truncation=true"
  VARIANT := $(DTYPE)-fused
  NARROWING_FLAGS := -DSATURATING_RESULT
else
  GEMMINI_CONVERT := -convert-linalg-to-gemmini
  VARIANT := $(DTYPE)
endif

# Appended to the runner compile, e.g. EXTRA_CXXFLAGS="-DLHS_VALUE=100 -DRHS_VALUE=100"
EXTRA_CXXFLAGS ?=

OUT := build
STEM := $(OUT)/matmul-$(VARIANT)
LINALG_MLIR := $(OUT)/matmul-$(DTYPE)-linalg.mlir
BUFFERIZED_MLIR := $(OUT)/matmul-$(DTYPE)-bufferized.mlir
GEMMINI_MLIR := $(STEM)-gemmini.mlir
LLVM_IR := $(STEM).ll
ASM := $(STEM).s
OBJECT := $(STEM).o
RUNNER_OBJECT := $(STEM)-runner.o
EXECUTABLE := $(STEM)-gemmini.elf

GEMMINI_OPTIONS := dim=16 addr_len=32 acc_rows=1024 bank_rows=4096 elem_t=i8 acc_t=i32
NPU_PIPELINE := -llvm-request-c-wrappers $(GEMMINI_CONVERT) \
	-convert-linalg-to-loops -lower-gemmini="$(GEMMINI_OPTIONS)"
LLC_OPTIONS := -mtriple=riscv64 -mattr=+npuext,+D -float-abi=hard

.PHONY: all import lower llvm asm compile run clean check-tools FORCE \
	i8 i32 run-i8 run-i32 run-all compile-i8 compile-i32 compile-all

all: compile

import: $(BUFFERIZED_MLIR)

$(BUFFERIZED_MLIR): export_matmul.py
	@mkdir -p $(OUT)
	$(PYTHON) $< --output-dir $(OUT) --result-dtype $(DTYPE)

lower: $(GEMMINI_MLIR)

$(GEMMINI_MLIR): $(BUFFERIZED_MLIR) | check-tools
	$(NPU_OPT) $< $(NPU_PIPELINE) -o $@

llvm: $(LLVM_IR)

$(LLVM_IR): $(GEMMINI_MLIR) | check-tools
	$(NPU_TRANSLATE) $< -npu-to-llvmir -o $@

asm: $(ASM)

$(ASM): $(LLVM_IR) | check-tools
	$(NPU_LLC) $< $(LLC_OPTIONS) -filetype=asm -o $@

compile: $(EXECUTABLE)

$(OBJECT): $(LLVM_IR) | check-tools
	$(NPU_LLC) $< $(LLC_OPTIONS) -filetype=obj -o $@
	# npu-llc records an ISA string ("zicsr", "i2p1") that binutils older than
	# 2.36 cannot parse -- the newlib toolchain shipped with conda esp-tools is
	# one -- and the linker then refuses to merge the object. The attributes
	# only drive that compatibility check, so dropping them is safe.
	$(CROSS_OBJCOPY) --remove-section=.riscv.attributes $@

# FORCE so a changed EXTRA_CXXFLAGS takes effect without a clean.
$(RUNNER_OBJECT): runner.cpp FORCE
	@mkdir -p $(OUT)
	$(CROSS_CXX) -c $< -O2 -static -I$(INTERFACES) $(RUNNER_FLAGS) $(NARROWING_FLAGS) $(EXTRA_CXXFLAGS) -o $@

$(EXECUTABLE): $(OBJECT) $(RUNNER_OBJECT)
	$(CROSS_CXX) $(OBJECT) $(RUNNER_OBJECT) -O2 -static -o $@

run: $(EXECUTABLE)
	$(SPIKE) $(SPIKE_OPTIONS) $(PK) $(EXECUTABLE)

# Convenience wrappers so both result types can be built from one invocation.
compile-i8:  ; @$(MAKE) --no-print-directory DTYPE=i8 compile
compile-i32: ; @$(MAKE) --no-print-directory DTYPE=i32 compile
compile-all: compile-i8 compile-i32
run-i8:  ; @$(MAKE) --no-print-directory DTYPE=i8 run
run-i32: ; @$(MAKE) --no-print-directory DTYPE=i32 run
run-all: run-i8 run-i32

check-tools:
	@test -x $(NPU_OPT) || { echo "missing $(NPU_OPT); build mlir-tools first"; exit 1; }
	@test -x $(NPU_TRANSLATE) || { echo "missing $(NPU_TRANSLATE); build mlir-tools first"; exit 1; }
	@test -x $(NPU_LLC) || { echo "missing $(NPU_LLC); build mlir-tools first"; exit 1; }

FORCE:

clean:
	rm -rf $(OUT)
