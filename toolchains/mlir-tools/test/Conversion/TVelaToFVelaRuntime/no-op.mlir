// RUN: npu-opt %s --lower-tvela-to-fvela-runtime | FileCheck %s --implicit-check-not=fvela_rope_runtime --implicit-check-not=tvela.

module {
  func.func @unrelated(%value: i32) -> i32 {
    return %value : i32
  }
}

// CHECK-LABEL: func.func @unrelated
