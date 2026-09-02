// RUN: npu-opt %s --one-shot-bufferize='bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map' | FileCheck %s

module {
  func.func @vfrope_test2(%input: tensor<8xi16>) -> tensor<8xi16> {
    %output = tvela.vfrope_q15_chunk %input {m = 64 : i64, idx = 8 : i64} : tensor<8xi16> -> tensor<8xi16>
    return %output : tensor<8xi16>
  }

  func.func @ternary_matmul(%lhs: tensor<1x64xi8>, %packed_rhs: tensor<64x16xi8>) -> tensor<1x64xi8> {
    %output = tvela.ternary_matmul %lhs, %packed_rhs : tensor<1x64xi8>, tensor<64x16xi8> -> tensor<1x64xi8>
    return %output : tensor<1x64xi8>
  }
}

// CHECK-LABEL: func.func @vfrope_test2
// CHECK: %[[OUTPUT:.*]] = memref.alloc() {{.*}} : memref<8xi16>
// CHECK: tvela.vfrope_q15_chunk_buffer %{{.*}}, %[[OUTPUT]]
// CHECK-SAME: idx = 8 : i64
// CHECK-SAME: m = 64 : i64
// CHECK-NOT: tvela.vfrope_q15_chunk {{.*}}tensor
// CHECK-LABEL: func.func @ternary_matmul
// CHECK: %[[TERNARY_OUTPUT:.*]] = memref.alloc() {{.*}} : memref<1x64xi8>
// CHECK: tvela.ternary_matmul_buffer %{{.*}}, %{{.*}}, %[[TERNARY_OUTPUT]]
// CHECK-SAME: memref<1x64xi8>, memref<64x16xi8>, memref<1x64xi8>
// CHECK-NOT: tvela.ternary_matmul {{.*}}tensor
