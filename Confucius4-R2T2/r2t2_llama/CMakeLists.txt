cmake_minimum_required(VERSION 3.18)
project(confucius_r2t2_native LANGUAGES CXX)

# llama.cpp source is NOT shipped in this repository. Point LLAMA_CPP_DIR at a
# local checkout (see README for the pinned commit) when rebuilding the native
# extension. The release ships prebuilt .so artifacts under r2t2_llama/bin/ so
# a source checkout is only needed for developers rebuilding the extension.
set(LLAMA_CPP_DIR "${CMAKE_CURRENT_LIST_DIR}/../third_party/llama.cpp" CACHE PATH "llama.cpp source directory")

set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)

add_subdirectory("${LLAMA_CPP_DIR}" llama_cpp EXCLUDE_FROM_ALL)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(qwen3asr_native native_ext.cpp)
target_compile_features(qwen3asr_native PRIVATE cxx_std_17)
target_include_directories(qwen3asr_native PRIVATE
    "${LLAMA_CPP_DIR}/include"
    "${LLAMA_CPP_DIR}/tools/mtmd"
)
target_link_libraries(qwen3asr_native PRIVATE mtmd llama)

# Emit the pybind extension into r2t2_llama/native/ next to native/__init__.py,
# so `from r2t2_llama.native.qwen3asr_native import Qwen3ASRNative` resolves at
# runtime.
# The rpath points at ../bin where the prebuilt libllama/libggml/libmtmd .so
# artifacts are shipped alongside the package.
set_target_properties(qwen3asr_native PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/native"
    BUILD_RPATH
        "${CMAKE_BINARY_DIR}/bin"
    INSTALL_RPATH
        "$ORIGIN/../bin"
)
