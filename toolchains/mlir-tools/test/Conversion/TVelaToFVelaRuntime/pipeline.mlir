// RUN: npu-opt %s --one-shot-bufferize='bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map' --lower-tvela-to-fvela-runtime | FileCheck %s

module {
  func.func @vfrope(%input: tensor<8xi16>) -> tensor<8xi16> {
    %output = tvela.vfrope_q15_chunk %input {m = 64 : i64, idx = 8 : i64} : tensor<8xi16> -> tensor<8xi16>
    return %output : tensor<8xi16>
  }
}

// CHECK: func.func private @fvela_rope_runtime(!llvm.ptr, !llvm.ptr, i64)
// CHECK-LABEL: func.func @vfrope
// CHECK: memref.alloc() {{.*}} : memref<8xi16>
// CHECK: arith.constant 4194312 : i64
// CHECK: call @fvela_rope_runtime
// CHECK-NOT: tvela.
