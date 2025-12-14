cmake_minimum_required(VERSION 3.18)

project(2.1_dim_and_index LANGUAGES CXX CUDA)

# Find CUDA toolkit to enable CUDA::cudart and other imported targets
find_package(CUDAToolkit REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA separable compilation
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Set CUDA architecture (adjust based on your GPU)
# Common architectures: 60 (Pascal), 70 (Volta), 75 (Turing), 80 (Ampere), 86 (Ada), 89 (Hopper)
set(CMAKE_CUDA_ARCHITECTURES "60" CACHE STRING "CUDA architectures")

# Set output directory for executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

# Add executable
add_executable(print_index src/print_index.cu)

# Set CUDA properties for the target
set_property(TARGET print_index PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Include directories (if needed)
target_include_directories(print_index PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Link CUDA libraries
target_link_libraries(print_index PRIVATE 
    CUDA::cudart
)

# Print configuration info
message(STATUS "CUDA Toolkit found: ${CUDAToolkit_FOUND}")
message(STATUS "CUDA Toolkit version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
