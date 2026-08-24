//===- BufferizableOpInterfaceImpl.h - T-Vela bufferization ----*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TVELA_BUFFERIZABLEOPINTERFACEIMPL_H
#define TVELA_BUFFERIZABLEOPINTERFACEIMPL_H

namespace mlir {
class DialectRegistry;
}

namespace npu::tvela {

void registerBufferizableOpInterfaceExternalModels(
    mlir::DialectRegistry &registry);

} // namespace npu::tvela

#endif // TVELA_BUFFERIZABLEOPINTERFACEIMPL_H
