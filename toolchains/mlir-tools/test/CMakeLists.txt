configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(NPU_TEST_DEPENDS
  FileCheck
  not
  npu-opt
)

add_lit_testsuite(check-tests "Running the npu-mlir regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${NPU_TEST_DEPENDS}
)
set_target_properties(check-tests PROPERTIES FOLDER "Tests")

add_lit_testsuites(NPU ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${NPU_TEST_DEPENDS})
