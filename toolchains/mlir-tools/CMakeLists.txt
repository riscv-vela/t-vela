cmake_minimum_required(VERSION 3.10)

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

#-------------------------------------------------------------------------------
# Project setup and globals
#-------------------------------------------------------------------------------

project(npu-mlir LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
include(ExternalProject)

#-------------------------------------------------------------------------------
# Options and settings
#-------------------------------------------------------------------------------

option(LLVM_INCLUDE_TOOLS "Generate build targets for the LLVM tools." ON)
option(LLVM_BUILD_TOOLS "Build the LLVM tools. If OFF, just generate build targets." ON)

#-------------------------------------------------------------------------------
# MLIR/LLVM Configuration
#-------------------------------------------------------------------------------
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

if(NOT DEFINED LLVM_MAIN_SRC_DIR)
  get_filename_component(LLVM_MAIN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/llvm/llvm ABSOLUTE)
endif()
set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

find_program(LLVM_TABLEGEN_EXE "llvm-tblgen" ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

set(npu_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(npu_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(npu_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(npu_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(npu_MIDEND_INCLUDE_DIR ${npu_SOURCE_DIR}/midend/include/)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${npu_BINARY_DIR})


# Generate libraries into `lib` of build directory.
set(LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)

# Add npu files to the include path
include_directories(${npu_MAIN_INCLUDE_DIR})
include_directories(${npu_MIDEND_INCLUDE_DIR})
include_directories(${npu_MIDEND_INCLUDE_DIR}/Interface)
include_directories(${npu_MIDEND_INCLUDE_DIR}/Dialect)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/midend/include/Dialect)
include_directories(BEFORE ${CMAKE_CURRENT_BINARY_DIR}/backend/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${npu_SOURCE_DIR}/lib)
include_directories(${npu_SOURCE_DIR}/midend/Interfaces)
# include_directories(${npu_SOURCE_DIR}/frontend/Interfaces)

# Add MLIR and LLVM headers to the include path
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

# Configure CMake.
list(APPEND CMAKE_MODULE_PATH ${MLIR_MAIN_SRC_DIR}/cmake/modules)
list(APPEND CMAKE_MODULE_PATH ${LLVM_MAIN_SRC_DIR}/cmake)

#-------------------------------------------------------------------------------
# Directory setup
#-------------------------------------------------------------------------------

add_subdirectory(cmake)
# add_subdirectory(frontend)
add_subdirectory(midend)
add_subdirectory(backend)
add_subdirectory(tools)

#-------------------------------------------------------------------------------
# Target check-npu
#-------------------------------------------------------------------------------

add_custom_target(check-npu
  DEPENDS check-tests
)
