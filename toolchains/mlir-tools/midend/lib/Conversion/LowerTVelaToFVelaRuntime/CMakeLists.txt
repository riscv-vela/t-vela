add_mlir_library(LowerTVelaToFVelaRuntimePass
  LowerTVelaToFVelaRuntime.cpp

  LINK_LIBS PUBLIC
  npuTVela
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRTransformUtils
  )
