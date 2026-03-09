# Versión mínima de CMake recomendada para un buen soporte de C++17 y targets modernos
cmake_minimum_required(VERSION 3.15)

# Nombre del proyecto y lenguaje
project(CAP_Practica_1 LANGUAGES CXX)

# Requerir estrictamente el estándar C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Desactivar extensiones específicas del compilador (como gnu++17) para garantizar portabilidad
set(CMAKE_CXX_EXTENSIONS OFF) 

# Exportar base de datos de comandos de compilación para que el IDE encuentre los headers
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Banderas de compilación orientadas a HPC y rendimiento extremo
# -O3: Máximo nivel de optimización segura.
# -march=native: Optimiza para la arquitectura específica de la CPU donde se compila (uso de AVX, etc).
# -DNDEBUG: Desactiva los asserts para no penalizar el rendimiento en producción.
# -Wall -Wextra: Activa las advertencias para mantener el código limpio.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -march=native -DNDEBUG")

# Localizar los paquetes de OpenMPI y OpenMP
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Listar todos los archivos fuente
set(SOURCES
    src/main.cpp
    src/io_utils.cpp
    src/kmeans.cpp
    src/stats.cpp
)

# Crear el ejecutable del proyecto
add_executable(${PROJECT_NAME} ${SOURCES})

# Indicar dónde están los archivos de cabecera (.h)
target_include_directories(${PROJECT_NAME} PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    /usr/lib/x86_64-linux-gnu/openmpi/include
)

# Enlazar el ejecutable con las librerías de MPI y OpenMP usando targets modernos
target_link_libraries(${PROJECT_NAME} PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)

# -----------------------------------------------------------------------------
# TOOL: Generador de Datasets Binarios
# Crea el ejecutable auxiliar para generar los datos de prueba masivos.
# -----------------------------------------------------------------------------
add_executable(generate_dataset tools/file_generator.cpp)

# -----------------------------------------------------------------------------
# TARGET OPCIONAL: Formateador de código (Google C++ Style Guide)
# Permite ejecutar `make format` para estandarizar el código automáticamente.
# -----------------------------------------------------------------------------
find_program(CLANG_FORMAT_EXE clang-format)
if(CLANG_FORMAT_EXE)
    file(GLOB_RECURSE ALL_SOURCE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
    )
    add_custom_target(format
        COMMAND ${CLANG_FORMAT_EXE} -i -style=Google ${ALL_SOURCE_FILES}
        COMMENT "Aplicando Google C++ Style Guide a todo el código fuente..."
    )
endif()