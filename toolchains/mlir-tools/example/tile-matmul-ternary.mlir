// RUN: npu-opt %s \
// RUN:     --lower-gemmini | \
// RUN: FileCheck %s

func.func @main() -> i8 {
  %i0 = arith.constant 0 : i8
  %aArray = memref.alloc() {alignment = 16} : memref<1x64xi8>
  %bArray = memref.alloc() {alignment = 16} : memref<64x16xi8>
  %cArray = memref.alloc() {alignment = 16} : memref<1x64xi8>
  %dArray = memref.alloc() {alignment = 64} : memref<1x64xi32>

  // CHECK: 1048577
  // CHECK: "gemmini.intr.loop_ws"
  gemmini.tile_matmul %aArray %bArray %cArray %dArray {dataflow=1, ternary=true} :
    memref<1x64xi8> memref<64x16xi8> memref<1x64xi8> memref<1x64xi32>

  return %i0 : i8
}
