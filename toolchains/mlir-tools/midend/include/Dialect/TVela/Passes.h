//===- Passes.h - T-Vela passes --------------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TVELA_PASSES_H
#define TVELA_PASSES_H

namespace mlir::npu {

void registerLowerTVelaToFVelaRuntimePass();

} // namespace mlir::npu

#endif // TVELA_PASSES_H
