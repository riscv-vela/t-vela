//===- BufferizableOpInterfaceImpl.cpp - T-Vela bufferization -----------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TVela/BufferizableOpInterfaceImpl.h"

#include "TVela/TVelaDialect.h"
#include "TVela/TVelaOps.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/DialectRegistry.h"

using namespace mlir;
using namespace mlir::bufferization;
using namespace npu::tvela;

namespace {

struct VFRopeQ15ChunkOpInterface
    : public BufferizableOpInterface::ExternalModel<VFRopeQ15ChunkOpInterface,
                                                    VFRopeQ15ChunkOp> {
  bool bufferizesToAllocation(Operation *op, Value value) const { return true; }

  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {};
  }

  AliasingOpOperandList
  getAliasingOpOperands(Operation *op, Value value,
                        const AnalysisState &state) const {
    return {};
  }

  FailureOr<BaseMemRefType>
  getBufferType(Operation *op, Value value, const BufferizationOptions &options,
                SmallVector<Value> &invocationStack) const {
    auto ropeOp = cast<VFRopeQ15ChunkOp>(op);
    auto inputType = bufferization::getBufferType(ropeOp.getInput(), options,
                                                  invocationStack);
    if (failed(inputType))
      return failure();
    auto resultType = cast<RankedTensorType>(ropeOp.getResult().getType());
    return MemRefType::get(resultType.getShape(), resultType.getElementType(),
                           MemRefLayoutAttrInterface(),
                           inputType->getMemorySpace());
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto ropeOp = cast<VFRopeQ15ChunkOp>(op);
    FailureOr<Value> inputBuffer =
        getBuffer(rewriter, ropeOp.getInput(), options);
    if (failed(inputBuffer))
      return failure();

    auto inputType = cast<MemRefType>(inputBuffer->getType());
    auto outputType = MemRefType::get(
        inputType.getShape(), inputType.getElementType(),
        MemRefLayoutAttrInterface(), inputType.getMemorySpace());
    FailureOr<Value> outputBuffer = options.createAlloc(
        rewriter, ropeOp.getLoc(), outputType, ValueRange());
    if (failed(outputBuffer))
      return failure();

    rewriter.create<VFRopeQ15ChunkBufferOp>(ropeOp.getLoc(), *inputBuffer,
                                            *outputBuffer, ropeOp.getMAttr(),
                                            ropeOp.getIdxAttr());
    replaceOpWithBufferizedValues(rewriter, op, *outputBuffer);
    return success();
  }
};

} // namespace

void npu::tvela::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, TVelaDialect *dialect) {
    VFRopeQ15ChunkOp::attachInterface<VFRopeQ15ChunkOpInterface>(*ctx);
  });
}
