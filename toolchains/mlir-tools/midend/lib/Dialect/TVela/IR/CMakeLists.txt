add_mlir_dialect_library(npuTVela
  TVelaDialect.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
  )
