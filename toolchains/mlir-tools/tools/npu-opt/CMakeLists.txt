get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_llvm_tool(npu-opt
  npu-opt.cpp
)

target_link_libraries(npu-opt
  PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  npuGemmini
  LowerGemminiPass
  LowerLinalgToGemminiPass

  MLIRTestTransforms
  MLIRTestTransformDialect
  MLIRTransforms
  MLIRTransformUtils
  MLIROptLib
  MLIRTestTransforms
  MLIRTestTransformDialect
  MLIRTransforms
  MLIRTransformUtils
  )
