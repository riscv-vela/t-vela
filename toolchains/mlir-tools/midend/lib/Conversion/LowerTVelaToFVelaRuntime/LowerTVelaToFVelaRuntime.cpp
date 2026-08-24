//===- LowerTVelaToFVelaRuntime.cpp - Lower T-Vela RoPE calls -----------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TVela/Passes.h"
#include "TVela/TVelaDialect.h"
#include "TVela/TVelaOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

constexpr StringLiteral kRuntimeName = "fvela_rope_runtime";

class VFRopeBufferOpLowering
    : public OpRewritePattern<::npu::tvela::VFRopeQ15ChunkBufferOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(::npu::tvela::VFRopeQ15ChunkBufferOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type i64Type = rewriter.getI64Type();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    Value inputIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
        loc, op.getInput());
    Value outputIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
        loc, op.getOutput());
    Value inputInt =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, inputIndex);
    Value outputInt =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, outputIndex);
    Value inputPtr = rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, inputInt);
    Value outputPtr =
        rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, outputInt);
    uint64_t config = (static_cast<uint64_t>(op.getM()) << 16) |
                      static_cast<uint64_t>(op.getIdx());
    Value configValue = rewriter.create<arith::ConstantIntOp>(
        loc, static_cast<int64_t>(config), 64);

    rewriter.create<func::CallOp>(loc, kRuntimeName, TypeRange(),
                                  ValueRange{inputPtr, outputPtr, configValue});
    rewriter.eraseOp(op);
    return success();
  }
};

class LowerTVelaToFVelaRuntimePass
    : public PassWrapper<LowerTVelaToFVelaRuntimePass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTVelaToFVelaRuntimePass)

  StringRef getArgument() const final { return "lower-tvela-to-fvela-runtime"; }
  StringRef getDescription() const final {
    return "Lower T-Vela operations to the F-Vela assembly runtime";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, LLVM::LLVMDialect,
                    memref::MemRefDialect, ::npu::tvela::TVelaDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool hasVFRopeOp = false;
    module.walk([&](Operation *op) {
      hasVFRopeOp |= isa<::npu::tvela::VFRopeQ15ChunkOp,
                         ::npu::tvela::VFRopeQ15ChunkBufferOp>(op);
    });
    if (!hasVFRopeOp)
      return;

    MLIRContext *context = &getContext();
    auto ptrType = LLVM::LLVMPointerType::get(context);
    auto expectedType = FunctionType::get(
        context, TypeRange{ptrType, ptrType, IntegerType::get(context, 64)},
        TypeRange());

    if (Operation *symbol = SymbolTable::lookupSymbolIn(module, kRuntimeName)) {
      auto runtime = dyn_cast<func::FuncOp>(symbol);
      if (!runtime || runtime.getFunctionType() != expectedType) {
        symbol->emitError("has an incompatible fvela_rope_runtime signature");
        signalPassFailure();
        return;
      }
    } else {
      OpBuilder builder(context);
      builder.setInsertionPointToStart(module.getBody());
      auto runtime = builder.create<func::FuncOp>(module.getLoc(), kRuntimeName,
                                                  expectedType);
      runtime.setPrivate();
    }

    ConversionTarget target(*context);
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    target.addIllegalOp<::npu::tvela::VFRopeQ15ChunkOp,
                        ::npu::tvela::VFRopeQ15ChunkBufferOp>();

    RewritePatternSet patterns(context);
    patterns.add<VFRopeBufferOpLowering>(context);
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void mlir::npu::registerLowerTVelaToFVelaRuntimePass() {
  PassRegistration<LowerTVelaToFVelaRuntimePass>();
}
