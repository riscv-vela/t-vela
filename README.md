
# T-Vela

![T-Vela compiler flow](figures/Toolchain.png)

## What is T-Vela?

T-Vela is an MLIR-level compiler toolchain that lowers PyTorch workloads to RISC-V systems with AI acceleration.
It exports standard operations as Linalg, preserves accelerator-specific operations in the T-Vela dialect until target selection, and applies hardware-aware transformations to generate RISC-V code for Original Gemmini, VelaNPU, and VelaVPU.

~~~text
PyTorch model
  -> torch-mlir
  -> mixed Linalg/T-Vela MLIR
  -> target-specific MLIR lowering
  -> LLVM dialect / LLVM IR
  -> RISC-V object or executable
~~~

The current targets are:

| Target | Main input operations | Target lowering | Current endpoint |
| --- | --- | --- | --- |
| Original Gemmini | Linalg matmul, batch matmul, and convolution | Gemmini dialect and RISC-V intrinsics | RISC-V ELF and Spike with the Gemmini extension |
| VelaNPU | `tvela.ternary_matmul` | F-Vela ternary matmul mode | Bare-metal RISC-V ELF and compiler validation |
| VelaVPU | `tvela.vfrope_q15_chunk` | F-Vela RoPE assembly runtime | Bare-metal RISC-V ELF and compiler validation |

`F-Vela` and the `fvela-*` names remain in paths, symbols, and Make targets because they are concrete implementation interfaces. This document uses VelaNPU and VelaVPU as the user-facing hardware names.

## Install

This section builds only the compiler toolchain. Chipyard, Spike and a RISC-V cross toolchain are not required to build `npu-opt`, `npu-translate`, or `npu-llc`.

The commands assume a Linux host, a checkout of this repository, and the following tools:

- Git
- CMake
- Ninja
- A C and C++ compiler with C++17 support
- Python 3 for LLVM/MLIR tests

Run all commands from the repository root.

### Build LLVM, MLIR, and Clang

~~~bash
git submodule update --init toolchains/mlir-tools/llvm
cmake -G Ninja \
  -S toolchains/mlir-tools/llvm/llvm \
  -B toolchains/mlir-tools/llvm/build \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build toolchains/mlir-tools/llvm/build \
  --target check-mlir \
  -j "$(nproc)"
~~~

`check-mlir` builds the required LLVM/MLIR components and runs the upstream MLIR test suite.

### Build and test T-Vela

~~~bash
cmake -G Ninja \
  -S toolchains/mlir-tools \
  -B toolchains/mlir-tools/build \
  -DMLIR_DIR="$PWD/toolchains/mlir-tools/llvm/build/lib/cmake/mlir" \
  -DLLVM_DIR="$PWD/toolchains/mlir-tools/llvm/build/lib/cmake/llvm" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build toolchains/mlir-tools/build -j "$(nproc)"
cmake --build toolchains/mlir-tools/build --target check-npu
~~~

The binaries are generated in `toolchains/mlir-tools/build/bin`:

| Tool | Role |
| --- | --- |
| `npu-opt` | Runs T-Vela, Linalg, Gemmini, bufferization, and standard MLIR passes |
| `npu-translate` | Translates supported MLIR LLVM/intrinsic operations to LLVM IR |
| `npu-llc` | Generates RISC-V assembly or object code with the T-Vela LLVM backend |

### Optional RISC-V environment

Producing final ELFs and running the Original Gemmini example require the repository's RISC-V environment. The full Chipyard setup extends the compiler-only installation with the required cross toolchain and simulation environment.

~~~bash
conda activate base
./build-setup.sh riscv-tools
source env.sh
~~~

The full setup also initializes Chipyard and simulation collateral, so expect a much larger download and build.

## Compile with T-Vela

The compiler drivers form the following common pipeline:

| Stage | Tool | Main output |
| --- | --- | --- |
| MLIR optimization and lowering | `npu-opt` | Lower-level MLIR |
| MLIR-to-LLVM translation | `npu-translate --npu-to-llvmir` | Textual LLVM IR |
| RISC-V code generation | `npu-llc` | Assembly or object file |
| Final link | RISC-V GCC toolchain | RISC-V ELF |

The example Makefiles connect these stages. Run the following commands from the repository root.

### Original Gemmini

The representative Original Gemmini example lowers an INT8 NHWC/HWCF convolution from Linalg to Gemmini and generates a statically linked RISC-V Linux executable.

Load the RISC-V environment:

~~~bash
source env.sh
~~~

Compile the example:

~~~bash
make -C toolchains/mlir-tools/example \
  gemmini-linalg-conv2d-nhwc-hwcf-i8-compile
~~~

The target applies the following flow:

~~~text
linalg.conv_2d_nhwc_hwcf
  -> --convert-linalg-to-gemmini
  -> gemmini.tile_conv
  -> --convert-linalg-to-loops
  -> --lower-gemmini="dim=4 addr_len=32 acc_rows=1024
                      bank_rows=4096 elem_t=i8 acc_t=i32"
  -> npu-translate --npu-to-llvmir
  -> npu-llc -mtriple=riscv64 -mattr=+npuext,+D
  -> RISC-V GCC
~~~

The output object and executable are `toolchains/mlir-tools/example/log.o` and `toolchains/mlir-tools/example/a.out`.

The lowering parameters must match the Gemmini configuration used to build the Spike extension.
In particular, verify `DIM`, accumulator rows, scratchpad rows, and element types before treating a run as a target-compatible result.

Run the executable on Spike with the Gemmini functional extension:

~~~bash
make -C toolchains/mlir-tools/example \
  gemmini-linalg-conv2d-nhwc-hwcf-i8-run
~~~

The `run` target depends on the compile target. The separate `*-lower` target is useful for inspecting `log.mlir`, but the compile target lowers the original input itself rather than consuming that file.

For a PyTorch INT8 matmul flow, see [the torch-mlir example](toolchains/mlir-tools/example/torch-mlir/README.md).

### VelaNPU ternary matmul

The VelaNPU example exports a packed ternary matrix multiplication from PyTorch, preserves it as a T-Vela custom operation, and reuses Gemmini lowering with ternary mode enabled.

Both Vela examples assume that the optional RISC-V environment from the installation section has already been created.

~~~text
torch.tvela.ternary_matmul
  -> tvela.ternary_matmul
  -> one-shot bufferization
  -> tvela.ternary_matmul_buffer
  -> gemmini.tile_matmul {noBias = true, ternary = true}
  -> Gemmini LOOP_WS intrinsics with is_mpgemm
  -> bare-metal RISC-V ELF
~~~

The example uses these shapes:

| Value | Shape and type |
| --- | --- |
| Activation | `tensor<1x64xi8>` |
| Logical weights | `tensor<64x64xi8>` with values in `{-1, 0, 1}` |
| Packed weights | `tensor<64x16xi8>` with four 2-bit weights per byte |
| Result | `tensor<1x64xi8>` with saturating move-out |

Packing uses `00 = 0`, `01 = 1`, and `11 = -1`, starting at the low bits of each byte.

Python 3.12 and network access are required on the first build. The Makefile invokes `toolchains/torch-mlir-fvela/setup.sh`, which builds the pinned PyTorch/torch-mlir frontend recorded in `versions.lock`.

~~~bash
make -C toolchains/mlir-tools/example/fvela-npu-ternary \
  fvela-npu-ternary-compile
make -C toolchains/mlir-tools/example/fvela-npu-ternary check
~~~

The output is `toolchains/mlir-tools/example/fvela-npu-ternary/build/ternary_matmul.elf`.
The check target validates the Python reference, T-Vela carrier, ternary mode bit, RISC-V ELF structure, symbols, and custom instruction encoding.

This README covers T-Vela compilation and compiler validation for VelaNPU. The generated ELF can be integrated with the corresponding VelaNPU hardware environment for system-level evaluation.

### VelaVPU RoPE

The VelaVPU example preserves one Q1.15 RoPE chunk as a T-Vela operation and lowers it to an assembly runtime call.

~~~text
torch.tvela.vfrope_q15_chunk
  -> tvela.vfrope_q15_chunk
  -> one-shot bufferization
  -> tvela.vfrope_q15_chunk_buffer
  -> call @fvela_rope_runtime(input, output, config)
  -> RISC-V Vector load, custom RoPE instruction, and vector store
  -> bare-metal RISC-V ELF
~~~

The current example operates on eight signed 16-bit Q1.15 values with `m = 65` and `idx = 0`. The two attributes are packed into a 64-bit runtime configuration.

~~~bash
make -C toolchains/mlir-tools/example/fvela-vpu-rope \
  saturn-vfrope-test2-compile
make -C toolchains/mlir-tools/example/fvela-vpu-rope check
~~~

The output is `toolchains/mlir-tools/example/fvela-vpu-rope/build/vfrope_test2.elf`.
The check target validates frontend conversion, the runtime symbol, embedded reference data, RISC-V ELF structure, RVV load/store operations, and the custom RoPE instruction encoding.

This README covers T-Vela compilation and compiler validation for VelaVPU. The generated ELF can be integrated with the corresponding VelaVPU hardware environment for system-level evaluation.

## Supported Targets and Current Status

![Step-by-step MLIR lowering](figures/Lowering_results.png)

The image shows Linalg input, a high-level Gemmini tiled operation, and low-level Gemmini intrinsics in the Original Gemmini path.

| Target | Demonstrated workload | Validation in this repository |
| --- | --- | --- |
| Original Gemmini | INT8/floating-point matmul and convolution, plus batch matmul lowering | MLIR lowering, RISC-V code generation, and selected Spike execution |
| VelaNPU | Fixed-shape packed ternary matmul | Frontend, MLIR, ELF/symbol, and instruction encoding validation |
| VelaVPU | One eight-element Q1.15 RoPE chunk | Frontend, MLIR, ELF/symbol/data, and RVV/custom encoding validation |

`check-npu` covers dialect verification, bufferization, T-Vela-to-Gemmini conversion, Gemmini ternary lowering, and VelaVPU runtime lowering.
Each Vela example also provides a `check` target for its frontend-to-ELF contract.

This section summarizes T-Vela's compiler support for VelaNPU and VelaVPU, including target-specific lowering and RISC-V binary generation.
These compiler outputs can be integrated with the corresponding Vela hardware environments for system-level validation and performance evaluation.

## Details

### MLIR Toolchain Directory Structure

~~~text
toolchains/
├── torch-mlir-fvela/              # Pinned PyTorch/torch-mlir frontend
└── mlir-tools/
    ├── llvm/                       # Upstream llvm-project submodule
    ├── midend/                    # Dialects, conversions, and MLIR lowering
    │   ├── include/
    │   └── lib/
    ├── backend/                   # LLVM intrinsics and RISC-V backend support
    ├── tools/                     # npu-opt, npu-translate, and npu-llc
    ├── example/                   # Original Gemmini, VelaNPU, and VelaVPU flows
    └── test/                      # Dialect and conversion regression tests
~~~

Generated `build` directories are omitted. The frontend, compiler passes, backend, command-line tools, examples, and regression tests remain separated by their main responsibilities.

### Compiler Pass and Lowering Details

#### `--convert-linalg-to-gemmini`

| Input operation | Conversion |
| --- | --- |
| `linalg.matmul` | `gemmini.tile_matmul` |
| `linalg.batch_matmul` | Per-batch subviews followed by matmul conversion |
| `linalg.conv_2d_nchw_fchw` | Layout conversion followed by `gemmini.tile_conv` |
| `linalg.conv_2d_nhwc_hwcf` | Flattening followed by `gemmini.tile_conv` |
| `tvela.ternary_matmul_buffer` | `gemmini.tile_matmul` with `noBias` and `ternary` |

| Option | Default | Description |
| --- | --- | --- |
| `acc_t` | `i32` | Accumulator and generated bias type; `f32` is used by floating-point paths |
| `fuse-truncation` | `false` | Folds a matching INT32-to-INT8 result conversion into Gemmini move-out |

`fuse-truncation` is off by default because `arith.trunci` keeps the low eight bits while Gemmini INT8 move-out saturates to `[-128, 127]`. INT8 input with INT32 output instead sets `fullC = true`.

#### `--lower-gemmini`

This pass lowers Gemmini operations to intrinsics and converts remaining Affine, SCF, Arith, MemRef, ControlFlow, and Func operations toward the LLVM dialect.

| Option | Default | Hardware meaning |
| --- | ---: | --- |
| `dim` | `16` | Systolic array dimension |
| `addr_len` | `32` | Gemmini local address width |
| `acc_rows` | `1024` | Accumulator rows |
| `bank_rows` | `4096` | Scratchpad rows per bank |
| `elem_t` | `i8` | Input and element type |
| `acc_t` | `i32` | Accumulator type |

The legalization pads matmul dimensions to `dim` and selects tile sizes within scratchpad and accumulator capacity.
It reserves space for double buffering and supports output-stationary and weight-stationary dataflows where permitted.

#### VelaNPU ternary extension

VelaNPU reuses the shared Gemmini lowering pipeline. `--convert-linalg-to-gemmini` creates `gemmini.tile_matmul {ternary = true}`, and `--lower-gemmini` validates and lowers it.

The path requires weight-stationary dataflow, INT8 A/B/C buffers, an INT32 bias buffer, and a packed B width that expands by four to the logical C width.
The final `LOOP_WS` payload carries `is_mpgemm` in `rs1[20]`. Code generation uses the custom `+npuext` RISC-V feature.

#### `--lower-tvela-to-fvela-runtime`

The VelaVPU pass:

1. Extracts aligned input and output pointers from the memrefs.
2. Packs `config = (m << 16) | idx` into an `i64`.
3. Calls `fvela_rope_runtime(ptr, ptr, i64)`.
4. Adds a private declaration when required.

An incompatible existing runtime symbol is rejected, while modules without RoPE remain unchanged.
The linked assembly loads eight i16 values with RVV, executes the raw `vfrope.fvx` encoding, and stores the results. This path targets the RISC-V Vector `+v` feature independently of the Gemmini `+npuext` path.

#### LLVM translation and code generation

`npu-translate --npu-to-llvmir` converts Gemmini intrinsics to `llvm.riscv.*` intrinsics.
`npu-llc` selects the custom Gemmini instruction patterns when `+npuext` is enabled. VelaVPU instead uses standard LLVM/RVV lowering plus the linked assembly runtime.

## Maintenance and Acknowledgments

T-Vela is developed as part of the broader Vela project.
ASO at Yonsei University has contributed to T-Vela and maintains this repository.

If you publish research using T-Vela, please acknowledge the T-Vela project in your publication.

## Reference

- [PyTorch](https://pytorch.org/)
- [MLIR](https://mlir.llvm.org/)
- [LLVM Project](https://llvm.org/)
- [torch-mlir](https://github.com/llvm/torch-mlir)
- [RISC-V International](https://riscv.org/)
- [Chipyard](https://github.com/ucb-bar/chipyard)
- [Gemmini](https://github.com/ucb-bar/gemmini)
- [Spike](https://github.com/riscv-software-src/riscv-isa-sim)
- [F-Vela](https://github.com/riscv-vela/f-vela)

## License

The repository-level [LICENSE](LICENSE) contains the BSD 3-Clause License.
Some files, third-party components, and Git submodules use their own terms, including [LICENSE.SiFive](LICENSE.SiFive) and the LLVM Project's Apache License 2.0 with LLVM exceptions. Review each component's notices before redistribution.
