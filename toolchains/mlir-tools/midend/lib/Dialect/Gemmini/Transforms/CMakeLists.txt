add_mlir_dialect_library(npuGemminiTransforms
  LegalizeForLLVMExport.cpp

  LINK_LIBS PUBLIC
  npuGemmini
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  )
