cmake_minimum_required(VERSION 3.16)
project(meanrevML_cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/W4 /O2)
else()
    add_compile_options(-O2 -Wall -Wextra)
endif()

add_executable(meanrevML_demo src/main.cpp)
target_include_directories(meanrevML_demo PRIVATE include)
option(BUILD_PYTHON_BINDINGS "Build pybind11 Python bindings" OFF)

if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(meanrevML_py bindings.cpp)
    target_include_directories(meanrevML_py PRIVATE include)
    target_compile_features(meanrevML_py PRIVATE cxx_std_17)
endif()