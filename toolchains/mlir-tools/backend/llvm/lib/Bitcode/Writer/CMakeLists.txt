set(LLVM_Writer_DIR ${LLVM_MAIN_SRC_DIR}/lib/Bitcode/Writer)


add_llvm_component_library(LLVMnpuBitWriter
  ${LLVM_Writer_DIR}/BitWriter.cpp
  ${LLVM_Writer_DIR}/BitcodeWriter.cpp
  ${LLVM_Writer_DIR}/BitcodeWriterPass.cpp
  ${LLVM_Writer_DIR}/ValueEnumerator.cpp

  DEPENDS
  npu_intrinsics_gen

  LINK_COMPONENTS
  Analysis
  npuCore
  MC
  npuObject
  Object
  Support
  TargetParser
  )
