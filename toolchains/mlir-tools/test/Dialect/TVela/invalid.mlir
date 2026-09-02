// RUN: npu-opt %s -split-input-file -verify-diagnostics

module {
  func.func @wrong_shape(%input: tensor<4xi16>) -> tensor<4xi16> {
    // expected-error@+1 {{requires input to be tensor<8xi16>}}
    %output = tvela.vfrope_q15_chunk %input {m = 65 : i64, idx = 0 : i64} : tensor<4xi16> -> tensor<4xi16>
    return %output : tensor<4xi16>
  }
}

// -----

module {
  func.func @wrong_dtype(%input: tensor<8xf32>) -> tensor<8xf32> {
    // expected-error@+1 {{requires input to be tensor<8xi16>}}
    %output = tvela.vfrope_q15_chunk %input {m = 65 : i64, idx = 0 : i64} : tensor<8xf32> -> tensor<8xf32>
    return %output : tensor<8xf32>
  }
}

// -----

module {
  func.func @m_out_of_range(%input: tensor<8xi16>) -> tensor<8xi16> {
    // expected-error@+1 {{requires m and idx in the range 0..65535}}
    %output = tvela.vfrope_q15_chunk %input {m = 65536 : i64, idx = 0 : i64} : tensor<8xi16> -> tensor<8xi16>
    return %output : tensor<8xi16>
  }
}

// -----

module {
  func.func @idx_out_of_range(%input: tensor<8xi16>) -> tensor<8xi16> {
    // expected-error@+1 {{requires m and idx in the range 0..65535}}
    %output = tvela.vfrope_q15_chunk %input {m = 65 : i64, idx = 65536 : i64} : tensor<8xi16> -> tensor<8xi16>
    return %output : tensor<8xi16>
  }
}

// -----

module {
  func.func @negative_config(%input: tensor<8xi16>) -> tensor<8xi16> {
    // expected-error@+1 {{requires m and idx in the range 0..65535}}
    %output = tvela.vfrope_q15_chunk %input {m = -1 : i64, idx = 0 : i64} : tensor<8xi16> -> tensor<8xi16>
    return %output : tensor<8xi16>
  }
}

// -----

module {
  func.func @wrong_memory_space(%input: memref<8xi16, 1>, %output: memref<8xi16, 1>) {
    // expected-error@+1 {{requires input to use the default memory space}}
    tvela.vfrope_q15_chunk_buffer %input, %output {m = 65 : i64, idx = 0 : i64} : memref<8xi16, 1>, memref<8xi16, 1>
    return
  }
}

// -----

module {
  func.func @wrong_ternary_packed_shape(%lhs: tensor<1x64xi8>, %packed_rhs: tensor<64x64xi8>) -> tensor<1x64xi8> {
    // expected-error@+1 {{requires packed_rhs to be 'tensor<64x16xi8>'}}
    %output = tvela.ternary_matmul %lhs, %packed_rhs : tensor<1x64xi8>, tensor<64x64xi8> -> tensor<1x64xi8>
    return %output : tensor<1x64xi8>
  }
}

// -----

module {
  func.func @wrong_ternary_output_dtype(%lhs: tensor<1x64xi8>, %packed_rhs: tensor<64x16xi8>) -> tensor<1x64xi32> {
    // expected-error@+1 {{requires result to be 'tensor<1x64xi8>'}}
    %output = tvela.ternary_matmul %lhs, %packed_rhs : tensor<1x64xi8>, tensor<64x16xi8> -> tensor<1x64xi32>
    return %output : tensor<1x64xi32>
  }
}
