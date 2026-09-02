add_mlir_library(LowerLinalgToGemminiPass
  LowerLinalgToGemmini.cpp

  LINK_LIBS PUBLIC
  npuTVela
  MLIRArithDialect
  npuGemmini
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTransformUtils
)
