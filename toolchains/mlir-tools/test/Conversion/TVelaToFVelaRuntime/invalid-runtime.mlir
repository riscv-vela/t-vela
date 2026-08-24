// RUN: not npu-opt %s --lower-tvela-to-fvela-runtime 2>&1 | FileCheck %s

module {
  func.func private @fvela_rope_runtime(i64)

  func.func @vfrope(%input: memref<8xi16>, %output: memref<8xi16>) {
    tvela.vfrope_q15_chunk_buffer %input, %output {m = 65 : i64, idx = 0 : i64} : memref<8xi16>, memref<8xi16>
    return
  }
}

// CHECK: error: has an incompatible fvela_rope_runtime signature
