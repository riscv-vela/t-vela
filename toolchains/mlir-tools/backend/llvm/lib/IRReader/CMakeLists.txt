set(LLVM_IRReader_DIR ${LLVM_MAIN_SRC_DIR}/lib/IRReader)

add_llvm_component_library(LLVMnpuIRReader
  ${LLVM_IRReader_DIR}/IRReader.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/IRReader

  DEPENDS
  npu_intrinsics_gen

  LINK_COMPONENTS
  npuAsmParser
  npuBitReader
  npuCore
  Support
  )
