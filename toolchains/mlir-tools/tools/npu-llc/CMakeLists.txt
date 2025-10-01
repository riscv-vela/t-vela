set(LLVM_LINK_COMPONENTS
  npuRISCVCodeGen
  npuRISCVAsmParser
  npuRISCVDesc
  npuRISCVInfo

  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos
  
  npuAnalysis
  npuAsmParser
  npuAsmPrinter
  npuCodeGen
  npuCore
  npuIRReader
  MC
  npuMIRParser
  npuRemarks
  npuScalarOpts
  npuSelectionDAG
  Support
  npuTarget
  npuIPO
  npuVectorize
  TargetParser
  npuTransformUtils
)

add_llvm_tool(npu-llc
  ${LLVM_MAIN_SRC_DIR}/tools/llc/llc.cpp

  DEPENDS
  npu_intrinsics_gen
  SUPPORT_PLUGINS
  )

export_executable_symbols_for_plugins(npu-llc)
