add_mlir_translation_library(npuGemminiToLLVMIRTranslation
  GemminiToLLVMIRTranslation.cpp

  DEPENDS
  npuGemminiConversionsIncGen
  npu_intrinsics_gen
  
  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  npuGemmini
  )
