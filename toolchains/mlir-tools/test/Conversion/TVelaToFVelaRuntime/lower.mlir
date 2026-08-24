// RUN: npu-opt %s --lower-tvela-to-fvela-runtime | FileCheck %s

module {
  func.func private @fvela_rope_runtime(!llvm.ptr, !llvm.ptr, i64)

  func.func @two_calls(%input: memref<8xi16>, %output0: memref<8xi16>, %output1: memref<8xi16>) {
    tvela.vfrope_q15_chunk_buffer %input, %output0 {m = 65 : i64, idx = 0 : i64} : memref<8xi16>, memref<8xi16>
    tvela.vfrope_q15_chunk_buffer %input, %output1 {m = 64 : i64, idx = 8 : i64} : memref<8xi16>, memref<8xi16>
    return
  }
}

// CHECK-COUNT-1: func.func private @fvela_rope_runtime(!llvm.ptr, !llvm.ptr, i64)
// CHECK-LABEL: func.func @two_calls
// CHECK: arith.constant 4259840 : i64
// CHECK: call @fvela_rope_runtime
// CHECK: arith.constant 4194312 : i64
// CHECK: call @fvela_rope_runtime
// CHECK-NOT: tvela.vfrope_q15_chunk_buffer
