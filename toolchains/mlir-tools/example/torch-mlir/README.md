# torch-mlir int8 matmul example

This example compiles a 16x16 `torch.matmul` with `int8` inputs through
torch-mlir and the T-Vela Gemmini compiler, and runs it on Spike:

```text
PyTorch -> Linalg on tensors -> Linalg on memrefs -> Gemmini -> LLVM IR
        -> RISC-V object -> executable -> Spike/Gemmini
```

Inputs are `int8` and Gemmini accumulates in `i32`. Two knobs select how that
accumulator reaches memory.

| knob | value | effect |
| ---- | ----- | ------ |
| `DTYPE` | `i8` (default) | result is `memref<16x16xi8>`; the accumulator is narrowed |
| | `i32` | result is `memref<16x16xi32>`; `fullC = true`, nothing is lost |
| `FUSE_TRUNCATION` | `0` (default) | the narrowing stays a separate `arith.trunci` loop — **wraps** |
| | `1` | the narrowing is folded into Gemmini's C move-out — **saturates** |

`FUSE_TRUNCATION` has no effect on `DTYPE=i32`, which has no truncation to fold.

## Which narrowing do I want?

The two disagree as soon as the accumulator leaves `[-128, 127]`, which a
16x16 int8 matmul reaches easily. Measured on Spike with `--extension=gemmini`:

| inputs | i32 accumulator | `FUSE_TRUNCATION=0` (i8) | `FUSE_TRUNCATION=1` (i8) |
| ------ | --------------- | ------------------------ | ------------------------ |
| 1 x 2 | 32 | 32 | 32 |
| 100 x 100 | 160000 | **0** | **127** |
| -100 x 100 | -160000 | **0** | **-128** |

The default matches PyTorch's and MLIR's `arith.trunci` semantics, so it is the
one to use when reproducing eager-mode results. Turn `FUSE_TRUNCATION=1` on for
a quantized flow, where clamping to the int8 range is the intended behaviour
and folding the narrowing away saves a full pass over the result. Use
`DTYPE=i32` when the exact product matters.

## Install torch-mlir

Create the `tmlir` Conda environment and install the snapshot packages using
the command from the [torch-mlir README](https://github.com/llvm/torch-mlir):

```bash
conda create -n tmlir python=3.11
conda activate tmlir
python -m pip install --upgrade pip
pip install --pre torch-mlir torchvision \
  --extra-index-url https://download.pytorch.org/whl/nightly/cpu \
  -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels
```

Build `toolchains/mlir-tools` as described in the repository root README. The
build must provide `build/bin/npu-opt`, `npu-translate`, and `npu-llc`.

## Install the RISC-V toolchain, Spike, and pk

`scripts/build-setup.sh` from the repository root does all of this. To set up
just what this example needs, from the repository root:

```bash
# Cross compilers (prebuilt, ~154 MB). flex supplies libfl.so.2, which the
# bundled binutils links against; dtc is a Spike build dependency.
conda create -n esp-tools -c ucb-bar -c conda-forge esp-tools=1.0.1
conda install -n esp-tools -c conda-forge flex dtc

export RISCV="$(conda info --base)/envs/esp-tools/esp-tools"
export PATH="$RISCV/bin:$(conda info --base)/envs/esp-tools/bin:$PATH"
export LD_LIBRARY_PATH="$(conda info --base)/envs/esp-tools/lib:$RISCV/lib:$LD_LIBRARY_PATH"

git submodule update --init --depth 1 \
  toolchains/riscv-tools/riscv-isa-sim toolchains/riscv-tools/riscv-pk generators/gemmini
git -C generators/gemmini submodule update --init --depth 1 software/libgemmini

# Spike
mkdir -p toolchains/riscv-tools/riscv-isa-sim/build
(cd toolchains/riscv-tools/riscv-isa-sim/build && \
 ../configure --prefix="$RISCV" --with-boost=no --with-boost-asio=no --with-boost-regex=no && \
 make -j"$(nproc)" && make install)

# Proxy kernel
mkdir -p toolchains/riscv-tools/riscv-pk/build
(cd toolchains/riscv-tools/riscv-pk/build && \
 CC= CXX= ../configure --prefix="$RISCV" --host=riscv64-unknown-elf && \
 make -j"$(nproc)" && make install)

# Gemmini functional model, loaded by spike --extension=gemmini
make -C generators/gemmini/software/libgemmini install
```

`pk` lands in `$RISCV/riscv64-unknown-elf/bin/pk`, which is not on `PATH`, so
pass it explicitly:

```bash
export PK="$RISCV/riscv64-unknown-elf/bin/pk"
```

Check that `generators/gemmini/software/libgemmini/gemmini_params.h` matches
the pipeline's `GEMMINI_OPTIONS` — the defaults (`DIM 16`, `ADDR_LEN 32`,
`ACC_ROWS 1024`, `BANK_ROWS 4096`, `elem_t = int8_t`, `acc_t = int32_t`) do.

## Compile and run

Run the example from this directory with the `tmlir` environment active and
`$RISCV` on `PATH`:

```bash
cd toolchains/mlir-tools/example/torch-mlir

make run-all PK="$PK"                        # both result types
make run DTYPE=i32 PK="$PK"                  # one of them
make run DTYPE=i8 FUSE_TRUNCATION=1 PK="$PK" # opt into the fused move-out
```

Each run prints the result type, the narrowing it expects, and the i32
accumulator it was checked against:

```text
PASS: torch.matmul int8 16x16 -> int8 (wrapping) produced 32 in every cell, i32 accumulator = 32
PASS: torch.matmul int8 16x16 -> int32 (none) produced 32 in every cell, i32 accumulator = 32
```

To reproduce the table above, override the input values. The runner adjusts the
value it expects to the selected narrowing, so every configuration should pass:

```bash
make run DTYPE=i8  FUSE_TRUNCATION=0 EXTRA_CXXFLAGS="-DLHS_VALUE=100 -DRHS_VALUE=100" PK="$PK"
make run DTYPE=i8  FUSE_TRUNCATION=1 EXTRA_CXXFLAGS="-DLHS_VALUE=100 -DRHS_VALUE=100" PK="$PK"
make run DTYPE=i32                   EXTRA_CXXFLAGS="-DLHS_VALUE=100 -DRHS_VALUE=100" PK="$PK"
```

Each compiler boundary can be inspected on its own. Artifacts are named after
the configuration, so the variants coexist under `build/`:

```bash
make import DTYPE=i32   # build/matmul-i32-{linalg,bufferized}.mlir
make lower  DTYPE=i32   # build/matmul-i32-gemmini.mlir
make llvm   DTYPE=i32   # build/matmul-i32.ll
make asm    DTYPE=i32   # build/matmul-i32.s
```

The configurations diverge at the Gemmini boundary:

```mlir
// DTYPE=i8, FUSE_TRUNCATION=0 -- the accumulator is C, truncation stays behind
gemmini.tile_matmul %arg0 %arg1 %alloc %bias {fullC = true}
  : memref<16x16xi8> memref<16x16xi8> memref<16x16xi32> memref<16x16xi32>
linalg.generic ins(%alloc) outs(%arg2) { arith.trunci }

// DTYPE=i8, FUSE_TRUNCATION=1 -- the i8 result is C, truncation is gone
gemmini.tile_matmul %arg0 %arg1 %arg2 %bias
  : memref<16x16xi8> memref<16x16xi8> memref<16x16xi8> memref<16x16xi32>

// DTYPE=i32 -- the accumulator is moved out as is
gemmini.tile_matmul %arg0 %arg1 %arg2 %bias {fullC = true}
  : memref<16x16xi8> memref<16x16xi8> memref<16x16xi32> memref<16x16xi32>
```

`fullC` reaches the hardware as a bit in the `loop_ws` flags (1 versus 3) and
as the C row stride in bytes (16 versus 64).

## Why the exporter rewrites the Torch IR

`export_matmul.py` retypes `torch.aten.matmul`'s result to `si32` in raw Torch
IR before running torch-mlir's standard Linalg lowering. Two torch-mlir
behaviours make this necessary:

* Integer matmul is lowered through an **i64** accumulator, which does not
  match Gemmini's i32 `acc_t`. Retyping the result makes the accumulator i32.
* Truncating to `i8` inside the matmul conversion aborts with
  `unimplemented: for conversion to byte or char type dstOriginalDtype has to
  be passed to convertScalarToDtype`, because a signless `i8` cannot say
  whether the Torch dtype was `Byte` or `Char`. Snapshot wheels then crash on
  the null result. For `DTYPE=i8` the exporter emits the cast as a separate
  `torch.prims.convert_element_type`, which lowers through its own pattern.

There is no PyTorch operation for "int8 operands, int32 result" that torch-mlir
can consume: `torch.matmul` requires matching dtypes, and `torch.aten._int_mm`
has no Torch-to-Linalg conversion pattern. Upcasting the operands with
`.to(torch.int32)` instead produces an i32 `linalg.matmul`, which the Gemmini
pass does not recognise as an int8 matmul.

The Python stage also performs bufferization with identity memref layouts. The
`-convert-linalg-to-gemmini` pass accepts statically shaped, memref-based
`linalg.matmul`, rather than tensor-based Linalg emitted directly by
torch-mlir.

## Overrides

```bash
make compile NPU_BUILD=/path/to/mlir-tools-build
make run SPIKE=/path/to/spike PK=/path/to/pk
make compile CROSS_PREFIX=riscv64-unknown-elf-
```

The object rule strips `.riscv.attributes` from `npu-llc`'s output. That
section records an ISA string (`zicsr`, `i2p1`) which binutils older than 2.36
cannot parse -- the newlib toolchain shipped with conda `esp-tools` is one --
and the linker then refuses to merge the object. The attributes only drive that
compatibility check, so dropping them is safe.
