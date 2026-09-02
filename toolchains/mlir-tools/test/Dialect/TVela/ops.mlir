// RUN: npu-opt %s | FileCheck %s

module {
  func.func @vfrope_test2(%input: tensor<8xi16>) -> tensor<8xi16> {
    %output = tvela.vfrope_q15_chunk %input {m = 64 : i64, idx = 8 : i64} : tensor<8xi16> -> tensor<8xi16>
    return %output : tensor<8xi16>
  }

  func.func @config_boundaries(%input: tensor<8xi16>) -> tensor<8xi16> {
    %output = tvela.vfrope_q15_chunk %input {m = 0 : i64, idx = 65535 : i64} : tensor<8xi16> -> tensor<8xi16>
    return %output : tensor<8xi16>
  }

  func.func @ternary_matmul(%lhs: tensor<1x64xi8>, %packed_rhs: tensor<64x16xi8>) -> tensor<1x64xi8> {
    %output = tvela.ternary_matmul %lhs, %packed_rhs : tensor<1x64xi8>, tensor<64x16xi8> -> tensor<1x64xi8>
    return %output : tensor<1x64xi8>
  }
}

// CHECK-LABEL: func.func @vfrope_test2
// CHECK: tvela.vfrope_q15_chunk
// CHECK-SAME: idx = 8 : i64
// CHECK-SAME: m = 64 : i64
// CHECK-SAME: tensor<8xi16> -> tensor<8xi16>
// CHECK-LABEL: func.func @config_boundaries
// CHECK: tvela.vfrope_q15_chunk
// CHECK-SAME: idx = 65535 : i64
// CHECK-SAME: m = 0 : i64
// CHECK-LABEL: func.func @ternary_matmul
// CHECK: tvela.ternary_matmul
// CHECK-SAME: tensor<1x64xi8>, tensor<64x16xi8> -> tensor<1x64xi8>
