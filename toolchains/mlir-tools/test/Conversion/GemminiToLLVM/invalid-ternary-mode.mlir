// RUN: not npu-opt %s --lower-gemmini 2>&1 | FileCheck %s

func.func @output_stationary_ternary() {
  %a = memref.alloc() : memref<1x64xi8>
  %b = memref.alloc() : memref<64x16xi8>
  %c = memref.alloc() : memref<1x64xi8>
  %d = memref.alloc() : memref<1x64xi32>
  // CHECK: error: 'gemmini.tile_matmul' op requires weight-stationary dataflow when ternary=true
  gemmini.tile_matmul %a %b %c %d {dataflow = 0, ternary = true} :
    memref<1x64xi8> memref<64x16xi8> memref<1x64xi8> memref<1x64xi32>
  return
}
