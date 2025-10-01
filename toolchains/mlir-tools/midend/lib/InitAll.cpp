//===----------- InitAll.cpp - Register all dialects and passes -----------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "npu-mlir-c/InitAll.h"

#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dialect.h"

#include "Dialect/Gemmini/GemminiDialect.h"

namespace mlir {
namespace npu {
void registerLowerGemminiPass();
void registerLowerLinalgToGemminiPass();
} // namespace npu
} // namespace mlir

void mlir::npu::registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<::npu::gemmini::GemminiDialect>();
}

void mlir::npu::registerAllPasses() {
  mlir::npu::registerLowerGemminiPass();
  mlir::npu::registerLowerLinalgToGemminiPass();
}
