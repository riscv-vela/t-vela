// RUN: npu-opt %s --one-shot-bufferize='bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map' --convert-linalg-to-gemmini | FileCheck %s

module {
  func.func @ternary_matmul(%lhs: tensor<1x64xi8>, %packed_rhs: tensor<64x16xi8>) -> tensor<1x64xi8> {
    %output = tvela.ternary_matmul %lhs, %packed_rhs : tensor<1x64xi8>, tensor<64x16xi8> -> tensor<1x64xi8>
    return %output : tensor<1x64xi8>
  }
}

// CHECK-LABEL: func.func @ternary_matmul
// CHECK: %[[OUTPUT:.*]] = memref.alloc() {{.*}} : memref<1x64xi8>
// CHECK: %[[BIAS:.*]] = memref.alloc() : memref<1x64xi32>
// CHECK-NOT: linalg.fill {{.*}} outs(%[[BIAS]] : memref<1x64xi32>)
// CHECK: gemmini.tile_matmul %{{.*}} %{{.*}} %[[OUTPUT]] %[[BIAS]] {noBias = true, ternary = true}
// CHECK-SAME: memref<1x64xi8> memref<64x16xi8> memref<1x64xi8> memref<1x64xi32>
// CHECK-NOT: tvela.ternary_matmul
