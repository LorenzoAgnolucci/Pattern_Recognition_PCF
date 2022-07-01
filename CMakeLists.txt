cmake_minimum_required(VERSION 3.15)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(Pattern_Recognition LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -fopenmp")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(Pattern_Recognition_Cpp_benchmark cpp/main.cpp cpp/Matrix.h cpp/Utils.h cpp/PatternRecognition.h )
add_executable(Pattern_Recognition_Cpp_correctness cpp/check_correctness.cpp cpp/Matrix.h cpp/Utils.h cpp/PatternRecognition.h )
add_executable(Pattern_Recognition_CUDA_benchmark CUDA/main.cu CUDA/PatternRecognition.cu CUDA/PatternRecognition.h CUDA/Utils.h)
add_executable(Pattern_Recognition_CUDA_correctness CUDA/check_correctness.cu CUDA/PatternRecognition.cu CUDA/PatternRecognition.h CUDA/Utils.h)

target_link_libraries(Pattern_Recognition_Cpp_benchmark ${OpenCV_LIBS})
target_link_libraries(Pattern_Recognition_Cpp_correctness ${OpenCV_LIBS})
target_link_libraries(Pattern_Recognition_CUDA_correctness ${OpenCV_LIBS})

set_target_properties(Pattern_Recognition_CUDA_benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(Pattern_Recognition_CUDA_correctness PROPERTIES CUDA_SEPARABLE_COMPILATION ON)