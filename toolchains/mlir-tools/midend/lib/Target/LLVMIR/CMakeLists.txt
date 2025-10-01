add_subdirectory(Dialect)

set(LLVM_OPTIONAL_SOURCES
  ConvertnpuToLLVMIR.cpp
  )

add_mlir_translation_library(npuToLLVMIRTranslationRegistration
  ConvertnpuToLLVMIR.cpp

  LINK_LIBS PUBLIC
  npuGemminiToLLVMIRTranslation
  )
