cmake_minimum_required(VERSION 3.15)
project(HeatDiffusionSim VERSION 1.0 LANGUAGES CXX)

# Enforce modern C++ standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enforce strict, high-performance compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall -Wextra -Werror -pedantic")

# Set include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

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

# ==============================================================================
# Future Phases Placeholder (Uncomment as you progress)
# ==============================================================================

# find_package(OpenMP)
# if(OpenMP_CXX_FOUND)
#     add_executable(heat_sim_omp src/main.cc src/grid.cc src/solver_omp.cc)
#     target_link_libraries(heat_sim_omp PUBLIC OpenMP::OpenMP_CXX)
# endif()

# find_package(MPI)
# if(MPI_CXX_FOUND)
#     add_executable(heat_sim_mpi src/main.cc src/grid.cc src/solver_mpi.cc)
#     target_link_libraries(heat_sim_mpi PUBLIC MPI::MPI_CXX)
# endif()

# enable_language(CUDA)
# if(CMAKE_CUDA_COMPILER)
#     add_executable(heat_sim_cuda src/main.cc src/grid.cc src/solver_cuda.cu)
#     target_compile_options(heat_sim_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -arch=native>)
# endif()