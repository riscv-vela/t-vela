// RUN: npu-opt %s --convert-linalg-to-gemmini | FileCheck %s
// RUN: npu-opt %s --convert-linalg-to-gemmini="fuse-truncation=true" \
// RUN:   | FileCheck %s --check-prefix=FUSE
// RUN: npu-opt %s --convert-linalg-to-gemmini="fuse-truncation=true" \
// RUN:   --convert-linalg-to-loops \
// RUN:   --lower-gemmini="dim=16 addr_len=32 acc_rows=1024 bank_rows=4096 elem_t=i8 acc_t=i32" \
// RUN:   | FileCheck %s --check-prefix=LLVM

// This is the bufferized shape emitted for torch.matmul with i8 inputs and an
// i8 result.
//
// By default the accumulator stays the Gemmini C operand and the truncation is
// left as a separate loop, which keeps arith.trunci's wrapping semantics.
// CHECK-LABEL: func.func @matmul_i8(
// CHECK: gemmini.tile_matmul %arg0 %arg1 %alloc
// CHECK-SAME: fullC = true
// CHECK-SAME: memref<16x16xi8> memref<16x16xi8> memref<16x16xi32> memref<16x16xi32>
// CHECK: linalg.generic
// CHECK: arith.trunci
//
// fuse-truncation folds the truncation into the move-out. Gemmini clamps there
// instead of wrapping, so this is opt-in.
// FUSE-LABEL: func.func @matmul_i8(
// FUSE: gemmini.tile_matmul %arg0 %arg1 %arg2
// FUSE-SAME: : memref<16x16xi8> memref<16x16xi8> memref<16x16xi8> memref<16x16xi32>
// FUSE-NOT: fullC
// FUSE-NOT: linalg.generic
// LLVM-LABEL: llvm.func @matmul_i8(
// LLVM: llvm.mlir.constant(16 : i64)
// LLVM: %[[I8_FLAGS:.+]] = llvm.mlir.constant(1 : i64)
// LLVM: "gemmini.intr.loop_ws"(%[[I8_FLAGS]],
func.func @matmul_i8(%lhs: memref<16x16xi8>, %rhs: memref<16x16xi8>,
                     %result: memref<16x16xi8>) {
  %acc = memref.alloc() : memref<16x16xi32>
  %zero = arith.constant 0 : i32
  linalg.fill ins(%zero : i32) outs(%acc : memref<16x16xi32>)
  linalg.matmul
      ins(%lhs, %rhs : memref<16x16xi8>, memref<16x16xi8>)
      outs(%acc : memref<16x16xi32>)
  linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%acc : memref<16x16xi32>)
      outs(%result : memref<16x16xi8>) {
    ^bb0(%value: i32, %unused: i8):
      %truncated = arith.trunci %value : i32 to i8
      linalg.yield %truncated : i8
  }
  memref.dealloc %acc : memref<16x16xi32>
  return
}

// An i32 result is moved out without truncation, in either mode.
// CHECK-LABEL: func.func @matmul_i32(
// CHECK: gemmini.tile_matmul %arg0 %arg1 %arg2
// CHECK-SAME: fullC = true
// CHECK-SAME: : memref<16x16xi8> memref<16x16xi8> memref<16x16xi32> memref<16x16xi32>
// FUSE-LABEL: func.func @matmul_i32(
// FUSE: gemmini.tile_matmul %arg0 %arg1 %arg2
// FUSE-SAME: fullC = true
// LLVM-LABEL: llvm.func @matmul_i32(
// LLVM: llvm.mlir.constant(64 : i64)
// LLVM: %[[I32_FLAGS:.+]] = llvm.mlir.constant(3 : i64)
// LLVM: "gemmini.intr.loop_ws"(%[[I32_FLAGS]],
func.func @matmul_i32(%lhs: memref<16x16xi8>, %rhs: memref<16x16xi8>,
                      %result: memref<16x16xi32>) {
  linalg.matmul
      ins(%lhs, %rhs : memref<16x16xi8>, memref<16x16xi8>)
      outs(%result : memref<16x16xi32>)
  return
}

// Even with fuse-truncation, the fusion must not fire when it would move the
// truncation across an operation that observes either buffer. Each case below
// falls back to the plain lowering: the accumulator stays the Gemmini C
// operand and the linalg.generic survives.

// A fill placed after the matmul clears the product, so folding the truncation
// into the move-out would turn an all-zero result into the matmul result.
// FUSE-LABEL: func.func @matmul_i8_clobbered_accumulator(
// FUSE: gemmini.tile_matmul
// FUSE-SAME: fullC = true
// FUSE-SAME: memref<16x16xi8> memref<16x16xi8> memref<16x16xi32> memref<16x16xi32>
// FUSE: linalg.fill
// FUSE: linalg.generic
// FUSE: arith.trunci
func.func @matmul_i8_clobbered_accumulator(%lhs: memref<16x16xi8>,
                                           %rhs: memref<16x16xi8>,
                                           %result: memref<16x16xi8>) {
  %zero = arith.constant 0 : i32
  %acc = memref.alloc() : memref<16x16xi32>
  linalg.matmul
      ins(%lhs, %rhs : memref<16x16xi8>, memref<16x16xi8>)
      outs(%acc : memref<16x16xi32>)
  linalg.fill ins(%zero : i32) outs(%acc : memref<16x16xi32>)
  linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%acc : memref<16x16xi32>)
      outs(%result : memref<16x16xi8>) {
    ^bb0(%value: i32, %unused: i8):
      %truncated = arith.trunci %value : i32 to i8
      linalg.yield %truncated : i8
  }
  memref.dealloc %acc : memref<16x16xi32>
  return
}

// The destination is allocated after the matmul, so it cannot become the C
// operand of an op built at the matmul.
// FUSE-LABEL: func.func @matmul_i8_late_destination(
// FUSE: gemmini.tile_matmul
// FUSE-SAME: fullC = true
// FUSE-SAME: memref<16x16xi32> memref<16x16xi32>
// FUSE: memref.alloc() : memref<16x16xi8>
// FUSE: linalg.generic
func.func @matmul_i8_late_destination(%lhs: memref<16x16xi8>,
                                      %rhs: memref<16x16xi8>) {
  %acc = memref.alloc() : memref<16x16xi32>
  linalg.matmul
      ins(%lhs, %rhs : memref<16x16xi8>, memref<16x16xi8>)
      outs(%acc : memref<16x16xi32>)
  %result = memref.alloc() : memref<16x16xi8>
  linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%acc : memref<16x16xi32>)
      outs(%result : memref<16x16xi8>) {
    ^bb0(%value: i32, %unused: i8):
      %truncated = arith.trunci %value : i32 to i8
      linalg.yield %truncated : i8
  }
  gemmini.print %result : memref<16x16xi8>
  memref.dealloc %result : memref<16x16xi8>
  memref.dealloc %acc : memref<16x16xi32>
  return
}

// Hoisting the truncation out of the loop would write the destination even
// when the loop does not run.
// FUSE-LABEL: func.func @matmul_i8_truncation_in_loop(
// FUSE: gemmini.tile_matmul
// FUSE-SAME: fullC = true
// FUSE-SAME: memref<16x16xi32> memref<16x16xi32>
// FUSE: scf.for
// FUSE: linalg.generic
func.func @matmul_i8_truncation_in_loop(%lhs: memref<16x16xi8>,
                                        %rhs: memref<16x16xi8>,
                                        %result: memref<16x16xi8>,
                                        %bound: index) {
  %zero = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %acc = memref.alloc() : memref<16x16xi32>
  linalg.fill ins(%zero : i32) outs(%acc : memref<16x16xi32>)
  linalg.matmul
      ins(%lhs, %rhs : memref<16x16xi8>, memref<16x16xi8>)
      outs(%acc : memref<16x16xi32>)
  scf.for %i = %c0 to %bound step %c1 {
    linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%acc : memref<16x16xi32>)
        outs(%result : memref<16x16xi8>) {
      ^bb0(%value: i32, %unused: i8):
        %truncated = arith.trunci %value : i32 to i8
        linalg.yield %truncated : i8
    }
  }
  memref.dealloc %acc : memref<16x16xi32>
  return
}
