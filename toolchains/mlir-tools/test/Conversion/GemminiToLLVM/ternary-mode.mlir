// RUN: npu-opt %s --lower-gemmini | FileCheck %s

func.func @ternary_no_bias() {
  %a = memref.alloc() : memref<1x64xi8>
  %b = memref.alloc() : memref<64x16xi8>
  %c = memref.alloc() : memref<1x64xi8>
  %d = memref.alloc() : memref<1x64xi32>
  gemmini.tile_matmul %a %b %c %d {noBias = true, ternary = true} :
    memref<1x64xi8> memref<64x16xi8> memref<1x64xi8> memref<1x64xi32>
  return
}

func.func @ternary_with_bias() {
  %a = memref.alloc() : memref<1x64xi8>
  %b = memref.alloc() : memref<64x16xi8>
  %c = memref.alloc() : memref<1x64xi8>
  %d = memref.alloc() : memref<1x64xi32>
  gemmini.tile_matmul %a %b %c %d {ternary = true} :
    memref<1x64xi8> memref<64x16xi8> memref<1x64xi8> memref<1x64xi32>
  return
}

// CHECK-LABEL: llvm.func @ternary_no_bias
// CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: "gemmini.intr.loop_ws_config_addrs_dc"(%[[ZERO]],
// CHECK: %[[MODE_NO_BIAS:.*]] = llvm.mlir.constant(1048576 : i64) : i64
// CHECK: "gemmini.intr.loop_ws"(%[[MODE_NO_BIAS]],

// CHECK-LABEL: llvm.func @ternary_with_bias
// CHECK: %[[MODE:.*]] = llvm.mlir.constant(1048577 : i64) : i64
// CHECK: "gemmini.intr.loop_ws"(%[[MODE]],
