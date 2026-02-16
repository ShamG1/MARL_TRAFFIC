cmake_minimum_required(VERSION 3.18)

if(POLICY CMP0148)
  cmake_policy(SET CMP0148 OLD)
endif()

project(SIM_MARL_ENV_ENV LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS ON)

# ============================================================================
# 编译优化选项
# ============================================================================

# Release模式下的优化标志
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# GCC/Clang优化选项
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # 基础优化
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
    
    # 针对本机CPU优化（如果不需要可移植性）
    option(ENABLE_NATIVE_OPTIMIZATION "Enable -march=native for maximum performance on this CPU" ON)
    if(ENABLE_NATIVE_OPTIMIZATION)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
    endif()
    
    # 快速数学运算（可能影响精度，但对于MCTS通常可接受）
    option(ENABLE_FAST_MATH "Enable -ffast-math for faster floating point operations" ON)
    if(ENABLE_FAST_MATH)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffast-math")
    endif()
    
    # 循环展开
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -funroll-loops")
    
    # 链接时优化（LTO）- 可显著提升性能但增加编译时间
    option(ENABLE_LTO "Enable Link Time Optimization" ON)
    if(ENABLE_LTO)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -flto")
        set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -flto")
    endif()
    
    # 向量化提示
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ftree-vectorize")
    
    # 启用OpenMP（可选，用于并行化）
    option(ENABLE_OPENMP "Enable OpenMP for parallel operations" OFF)
    if(ENABLE_OPENMP)
        find_package(OpenMP)
        if(OpenMP_CXX_FOUND)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        endif()
    endif()
endif()

# MSVC优化选项
if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Ob2 /Oi /Ot /GL")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG")
    
    # 启用AVX2（如果CPU支持）
    option(ENABLE_AVX2 "Enable AVX2 instructions" ON)
    if(ENABLE_AVX2)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /arch:AVX2")
    endif()
endif()

# ============================================================================
# Python和pybind11
# ============================================================================

if(NOT Python3_EXECUTABLE)
    set(Python3_EXECUTABLE /home/g/anaconda3/envs/g308/bin/python)
endif()

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('include'))"
    OUTPUT_VARIABLE Python3_INCLUDE_DIRS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; import os; print(os.path.dirname(sysconfig.get_config_var('LIBDIR') or '/'))"
    OUTPUT_VARIABLE Python3_LIBRARY_DIRS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(pybind11 CONFIG QUIET)

if (NOT pybind11_FOUND)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _PY_OK)
    if (_PY_OK EQUAL 0 AND EXISTS "${PYBIND11_CMAKE_DIR}/pybind11Config.cmake")
        list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
        find_package(pybind11 CONFIG REQUIRED)
    endif()
endif()

if (NOT pybind11_FOUND)
    include(FetchContent)
    set(FETCHCONTENT_QUIET FALSE)
    message(STATUS "Fetching pybind11 via FetchContent...")
    FetchContent_Declare(
        pybind11
        URL https://github.com/pybind/pybind11/archive/refs/tags/v2.11.1.zip
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# ============================================================================
# LibTorch
# ============================================================================

if(NOT DEFINED Torch_DIR)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import torch, os; base=os.path.dirname(torch.__file__); print(os.path.join(base,'share','cmake','Torch'))"
        OUTPUT_VARIABLE Torch_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()

find_package(Torch REQUIRED)


# FreeType for text rendering

# ============================================================================
# 目标定义
# ============================================================================

set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui)

set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl2.cpp
)

set(MCTS_SOURCES
    bindings.cpp
    Car.cpp
    BitmapMask.cpp
    ScenarioEnv.cpp
    ScenarioEnv_render.cpp
    Lidar.cpp
    RouteGen.cpp
    RoadMask.cpp
    LineMask.cpp
    TrafficFlow.cpp
    mcts_search.cpp
    Renderer.cpp
    ImGuiOverlay.cpp
    stb_image_impl.cpp
    ${IMGUI_SOURCES}
)

# Assets directory (for icon loading)
add_definitions(-DCPP_ASSETS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/assets")

if(WIN32)
    pybind11_add_module(SIM_MARL_ENV ${MCTS_SOURCES})
else()
    add_compile_definitions(SIM_MARL_ENABLE_RENDER)
    pybind11_add_module(SIM_MARL_ENV ${MCTS_SOURCES})
endif()

# ============================================================================
# 编译选项应用到目标
# ============================================================================

# 针对目标的额外优化
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(SIM_MARL_ENV PRIVATE
        $<$<CONFIG:Release>:-fvisibility=hidden>  # 隐藏不必要的符号
        $<$<CONFIG:Release>:-fomit-frame-pointer>  # 省略帧指针
    )
endif()

# ============================================================================
# Torch链接
# ============================================================================

target_link_libraries(SIM_MARL_ENV PRIVATE ${TORCH_LIBRARIES})

# ============================================================================
# OpenGL + GLFW
# ============================================================================

find_package(OpenGL REQUIRED)

if(WIN32)
    if(NOT DEFINED GLFW_ROOT)
        set(GLFW_ROOT "${CMAKE_SOURCE_DIR}/../../TOOLS/glfw-3.4.bin.WIN64")
    endif()

    set(GLFW_INCLUDE_DIR "${GLFW_ROOT}/include")
    set(GLFW_LIB_DIR "${GLFW_ROOT}/lib-vc2022")

    if(NOT EXISTS "${GLFW_INCLUDE_DIR}/GLFW/glfw3.h")
        message(FATAL_ERROR "Cannot find GLFW headers in ${GLFW_INCLUDE_DIR}. Check GLFW_ROOT")
    endif()

    if(EXISTS "${GLFW_LIB_DIR}/glfw3.lib")
        set(GLFW_LIB_NAME glfw3)
    elseif(EXISTS "${GLFW_LIB_DIR}/glfw3dll.lib")
        set(GLFW_LIB_NAME glfw3dll)
    else()
        message(FATAL_ERROR "Cannot find glfw3.lib or glfw3dll.lib in ${GLFW_LIB_DIR}. Check GLFW_ROOT")
    endif()

    target_include_directories(SIM_MARL_ENV PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${GLFW_INCLUDE_DIR})
    target_link_directories(SIM_MARL_ENV PRIVATE ${GLFW_LIB_DIR})

    target_link_libraries(SIM_MARL_ENV PRIVATE ${GLFW_LIB_NAME} opengl32)
    target_link_libraries(SIM_MARL_ENV PRIVATE gdi32)

    target_compile_options(SIM_MARL_ENV PRIVATE /utf-8)
else()
    find_package(glfw3 3.3 REQUIRED)
    target_include_directories(SIM_MARL_ENV PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR} 
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
)
    target_link_libraries(SIM_MARL_ENV PRIVATE glfw OpenGL::GL dl pthread)
    
    # OpenMP链接（如果启用）
    if(ENABLE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(SIM_MARL_ENV PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

# ============================================================================
# 输出编译配置信息
# ============================================================================

message(STATUS "=== MCTS Build Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Native optimization: ${ENABLE_NATIVE_OPTIMIZATION}")
message(STATUS "Fast math: ${ENABLE_FAST_MATH}")
message(STATUS "LTO: ${ENABLE_LTO}")
message(STATUS "OpenMP: ${ENABLE_OPENMP}")
if(MSVC)
    message(STATUS "AVX2: ${ENABLE_AVX2}")
endif()
message(STATUS "Release flags: ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "================================")