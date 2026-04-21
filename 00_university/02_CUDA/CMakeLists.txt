# Minimum CMake version recommended for good C++17 support and modern targets.
cmake_minimum_required(VERSION 3.18)
project(HeatDiffusionSim VERSION 1.0 LANGUAGES CXX CUDA)

# Strictly require the C++17 standard.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compilation command database for IDEs to locate headers.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ==============================================================================
# Phase 1: Sequential Baseline Setup
# ==============================================================================
# Note: You will need to create empty dummy files for src/main.cc, src/grid.cc, 
# and src/solver_seq.cc so CMake can configure successfully initially.

add_executable(heat_sim_seq 
    src/main.cc 
    src/grid.cc 
    src/solver_seq.cc
)

# Specify include directory for header files (.h).
target_include_directories(heat_sim_seq PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Compilation flags oriented towards HPC and extreme performance.
target_compile_options(heat_sim_seq PRIVATE -O3 -march=native -Wall -Wextra -Werror -pedantic)

# ==============================================================================
# Future Phases Placeholder (Uncomment as you progress)
# ==============================================================================

# find_package(OpenMP)
# if(OpenMP_CXX_FOUND)
#     add_executable(heat_sim_omp src/main.cc src/grid.cc src/solver_omp.cc)
#     target_include_directories(heat_sim_omp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
#     target_link_libraries(heat_sim_omp PUBLIC OpenMP::OpenMP_CXX)
# endif()

# find_package(MPI)
# if(MPI_CXX_FOUND)
#     add_executable(heat_sim_mpi src/main.cc src/grid.cc src/solver_mpi.cc)
#     target_include_directories(heat_sim_mpi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
#     target_link_libraries(heat_sim_mpi PUBLIC MPI::MPI_CXX)
# endif()

# if(PROJECT_IS_TOP_LEVEL OR DEFINED CMAKE_CUDA_COMPILER)
#     add_executable(heat_sim_cuda src/main.cc src/grid.cc src/solver_cuda.cu)
#     target_include_directories(heat_sim_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
#     target_compile_options(heat_sim_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -arch=native>)
# endif()

# -----------------------------------------------------------------------------
# OPTIONAL TARGET: Code formatter (Google C++ Style Guide).
# Allows running 'make format' to automatically standardize the code.
# -----------------------------------------------------------------------------
find_program(CLANG_FORMAT_EXE clang-format)
if(CLANG_FORMAT_EXE)
    file(GLOB_RECURSE ALL_SOURCE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
    )
    add_custom_target(format
        COMMAND ${CLANG_FORMAT_EXE} -i -style=file ${ALL_SOURCE_FILES}
        COMMENT "Applying Google C++ Style Guide via .clang-format..."
    )
endif()