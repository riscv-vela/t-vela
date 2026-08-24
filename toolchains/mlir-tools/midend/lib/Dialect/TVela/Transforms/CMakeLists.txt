add_mlir_dialect_library(npuTVelaTransforms
  BufferizableOpInterfaceImpl.cpp

  LINK_LIBS PUBLIC
  npuTVela
  MLIRBufferizationDialect
  MLIRIR
  MLIRMemRefDialect
  )
