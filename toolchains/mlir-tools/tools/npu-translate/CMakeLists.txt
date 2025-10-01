set(LLVM_LINK_COMPONENTS
  Support
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_llvm_tool(npu-translate
  npu-translate.cpp
  )
llvm_update_compile_flags(npu-translate)
target_link_libraries(npu-translate
  PRIVATE
  ${dialect_libs}
  ${translation_libs}
  LLVMnpuCore
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSPIRVDialect
  MLIRTranslateLib
  MLIRSupport
  )

mlir_check_link_libraries(npu-translate)
