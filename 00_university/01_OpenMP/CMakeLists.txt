# Minimum CMake version recommended for good C++17 support and modern targets.
cmake_minimum_required(VERSION 3.15)

# Project name and languages.
project(CAP_Practica_1 LANGUAGES CXX)

# Strictly require the C++17 standard.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Disable compiler-specific extensions (like gnu++17) to ensure portability.
set(CMAKE_CXX_EXTENSIONS OFF) 

# Export compilation command database for IDEs to locate headers.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compilation flags oriented towards HPC and extreme performance.
# -O3: Maximum safe optimization level.
# -march=native: Optimize for the specific CPU architecture where it is compiled (e.g., AVX).
# -DNDEBUG: Disable asserts to avoid performance penalties in production.
# -Wall -Wextra: Enable warnings to keep the code clean.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -march=native -DNDEBUG")

# Locate the OpenMPI and OpenMP packages.
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# List all source files.
set(SOURCES
    src/main.cpp
    src/io_utils.cpp
    src/kmeans.cpp
    src/stats.cpp
)

# Create the project executable.
add_executable(${PROJECT_NAME} ${SOURCES})

# Specify include directory for header files (.h).
target_include_directories(${PROJECT_NAME} PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    /usr/lib/x86_64-linux-gnu/openmpi/include
)

# Link the executable with MPI and OpenMP libraries using modern targets.
target_link_libraries(${PROJECT_NAME} PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)

# -----------------------------------------------------------------------------
# TOOL: Binary Dataset Generator.
# Creates the auxiliary executable to generate massive test data.
# -----------------------------------------------------------------------------
add_executable(generate_dataset tools/file_generator.cpp)

# -----------------------------------------------------------------------------
# OPTIONAL TARGET: Code formatter (Google C++ Style Guide).
# Allows running 'make format' to automatically standardize the code.
# -----------------------------------------------------------------------------
find_program(CLANG_FORMAT_EXE clang-format)
if(CLANG_FORMAT_EXE)
    file(GLOB_RECURSE ALL_SOURCE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
    )
    add_custom_target(format
        COMMAND ${CLANG_FORMAT_EXE} -i -style=Google ${ALL_SOURCE_FILES}
        COMMENT "Applying Google C++ Style Guide to all source code..."
    )
endif()