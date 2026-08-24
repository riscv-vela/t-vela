// RUN: npu-opt %s --one-shot-bufferize='bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map' | FileCheck %s

module {
  func.func @vfrope_test2(%input: tensor<8xi16>) -> tensor<8xi16> {
    %output = tvela.vfrope_q15_chunk %input {m = 64 : i64, idx = 8 : i64} : tensor<8xi16> -> tensor<8xi16>
    return %output : tensor<8xi16>
  }
}

// CHECK-LABEL: func.func @vfrope_test2
// CHECK: %[[OUTPUT:.*]] = memref.alloc() {{.*}} : memref<8xi16>
// CHECK: tvela.vfrope_q15_chunk_buffer %{{.*}}, %[[OUTPUT]]
// CHECK-SAME: idx = 8 : i64
// CHECK-SAME: m = 64 : i64
// CHECK-NOT: tvela.vfrope_q15_chunk {{.*}}tensor
