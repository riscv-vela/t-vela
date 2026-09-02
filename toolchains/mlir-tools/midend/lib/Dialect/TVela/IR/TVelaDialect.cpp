//===- TVelaDialect.cpp - T-Vela dialect implementation -----------------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TVela/TVelaDialect.h"
#include "TVela/TVelaOps.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;
using namespace npu::tvela;

#include "TVela/TVelaDialect.cpp.inc"

#define GET_OP_CLASSES
#include "TVela/TVela.cpp.inc"

namespace {

LogicalResult verifyConfig(Operation *op, int64_t m, int64_t idx) {
  constexpr int64_t maxConfigField = 0xffff;
  if (m < 0 || m > maxConfigField || idx < 0 || idx > maxConfigField)
    return op->emitOpError("requires m and idx in the range 0..65535");
  return success();
}

LogicalResult verifyTensorType(Operation *op, Type type, StringRef name) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  if (!tensorType || tensorType.getShape() != ArrayRef<int64_t>{8} ||
      !tensorType.getElementType().isSignlessInteger(16))
    return op->emitOpError() << "requires " << name << " to be tensor<8xi16>";
  return success();
}

LogicalResult verifyMemRefType(Operation *op, Type type, StringRef name) {
  auto memRefType = dyn_cast<MemRefType>(type);
  if (!memRefType || memRefType.getShape() != ArrayRef<int64_t>{8} ||
      !memRefType.getElementType().isSignlessInteger(16))
    return op->emitOpError() << "requires " << name << " to be memref<8xi16>";
  if (!memRefType.getLayout().isIdentity())
    return op->emitOpError()
           << "requires " << name << " to have an identity layout";
  if (memRefType.getMemorySpace())
    return op->emitOpError()
           << "requires " << name << " to use the default memory space";
  return success();
}

LogicalResult verifyTernaryTensorType(Operation *op, Type type,
                                      ArrayRef<int64_t> shape, StringRef name) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  if (!tensorType || tensorType.getShape() != shape ||
      !tensorType.getElementType().isSignlessInteger(8)) {
    auto expected =
        RankedTensorType::get(shape, IntegerType::get(op->getContext(), 8));
    return op->emitOpError() << "requires " << name << " to be " << expected;
  }
  return success();
}

LogicalResult verifyTernaryMemRefType(Operation *op, Type type,
                                      ArrayRef<int64_t> shape, StringRef name) {
  auto memRefType = dyn_cast<MemRefType>(type);
  if (!memRefType || memRefType.getShape() != shape ||
      !memRefType.getElementType().isSignlessInteger(8)) {
    auto expected =
        MemRefType::get(shape, IntegerType::get(op->getContext(), 8));
    return op->emitOpError() << "requires " << name << " to be " << expected;
  }
  if (!memRefType.getLayout().isIdentity())
    return op->emitOpError()
           << "requires " << name << " to have an identity layout";
  if (memRefType.getMemorySpace())
    return op->emitOpError()
           << "requires " << name << " to use the default memory space";
  return success();
}

} // namespace

void TVelaDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TVela/TVela.cpp.inc"
      >();
}

LogicalResult VFRopeQ15ChunkOp::verify() {
  if (failed(verifyTensorType(getOperation(), getInput().getType(), "input")) ||
      failed(verifyTensorType(getOperation(), getResult().getType(), "result")))
    return failure();
  return verifyConfig(getOperation(), getM(), getIdx());
}

LogicalResult VFRopeQ15ChunkBufferOp::verify() {
  if (failed(verifyMemRefType(getOperation(), getInput().getType(), "input")) ||
      failed(verifyMemRefType(getOperation(), getOutput().getType(), "output")))
    return failure();
  return verifyConfig(getOperation(), getM(), getIdx());
}

LogicalResult TernaryMatmulOp::verify() {
  if (failed(verifyTernaryTensorType(getOperation(), getLhs().getType(),
                                     {1, 64}, "lhs")) ||
      failed(verifyTernaryTensorType(getOperation(), getPackedRhs().getType(),
                                     {64, 16}, "packed_rhs")) ||
      failed(verifyTernaryTensorType(getOperation(), getResult().getType(),
                                     {1, 64}, "result")))
    return failure();
  return success();
}

LogicalResult TernaryMatmulBufferOp::verify() {
  if (failed(verifyTernaryMemRefType(getOperation(), getLhs().getType(),
                                     {1, 64}, "lhs")) ||
      failed(verifyTernaryMemRefType(getOperation(), getPackedRhs().getType(),
                                     {64, 16}, "packed_rhs")) ||
      failed(verifyTernaryMemRefType(getOperation(), getOutput().getType(),
                                     {1, 64}, "output")))
    return failure();
  return success();
}
