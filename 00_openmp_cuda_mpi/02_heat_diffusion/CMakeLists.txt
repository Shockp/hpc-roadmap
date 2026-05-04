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
# Simulation Executable Configuration
# ==============================================================================
# We use a single executable that can route to different solvers at runtime
# based on command-line arguments.

set(HEAT_SIM_SOURCES 
    src/main.cc 
    src/grid.cc 
    src/solver_seq.cc
)

if(CMAKE_CUDA_COMPILER)
    list(APPEND HEAT_SIM_SOURCES src/solver_cuda.cu)
endif()

# Check for OpenMP support
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    list(APPEND HEAT_SIM_SOURCES src/solver_omp.cc)
    if(CMAKE_CUDA_COMPILER)
        list(APPEND HEAT_SIM_SOURCES src/solver_hybrid.cu)
    endif()
endif()

# Check for MPI support
find_package(MPI)
if(MPI_CXX_FOUND)
    list(APPEND HEAT_SIM_SOURCES 
        src/solver_mpi_utils.cc
        src/solver_mpi_blocking.cc
        src/solver_mpi_nonblocking.cc
    )
endif()

add_executable(heat_sim ${HEAT_SIM_SOURCES})

if(MPI_CXX_FOUND)
    target_link_libraries(heat_sim PUBLIC MPI::MPI_CXX)
    target_compile_definitions(heat_sim PRIVATE ENABLE_MPI)
endif()

# Specify include directory for header files (.h).
target_include_directories(heat_sim PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Compilation flags oriented towards HPC and extreme performance.
target_compile_options(heat_sim PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra -Werror -pedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler -Wall,-Wextra>
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(heat_sim PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(heat_sim PRIVATE ENABLE_OPENMP)
endif()

if(CMAKE_CUDA_COMPILER)
    target_compile_definitions(heat_sim PRIVATE ENABLE_CUDA)
endif()

# -----------------------------------------------------------------------------
# OPTIONAL TARGET: Code formatter (Google C++ Style Guide).
# Allows running 'make format' to automatically standardize the code.
# -----------------------------------------------------------------------------
find_program(CLANG_FORMAT_EXE clang-format)
if(CLANG_FORMAT_EXE)
    file(GLOB_RECURSE ALL_SOURCE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/*.cuh"
    )
    add_custom_target(format
        COMMAND ${CLANG_FORMAT_EXE} -i -style=file ${ALL_SOURCE_FILES}
        COMMENT "Applying Google C++ Style Guide via .clang-format..."
    )
endif()

# -----------------------------------------------------------------------------
# HELPER TARGETS: Quick Run and Benchmarking.
# -----------------------------------------------------------------------------

# Run a small validation simulation (Sequential, 256x256, 100 iterations)
add_custom_target(run
    COMMAND heat_sim 256 100 seq
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS heat_sim
    COMMENT "Running validation simulation (Sequential, 256x256)..."
)

# Run the full automated benchmarking suite
add_custom_target(benchmark
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_benchmarks.sh
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS heat_sim
    COMMENT "Launching automated benchmark suite..."
)