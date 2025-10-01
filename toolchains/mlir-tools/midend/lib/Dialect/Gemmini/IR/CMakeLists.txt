add_mlir_dialect_library(npuGemmini
  GemminiDialect.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
)
