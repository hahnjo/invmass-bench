cmake_minimum_required(VERSION 3.11)
project (invmass-benchmark CXX)
add_compile_options(-Wall -Wextra -Wpedantic)
set(CMAKE_CXX_STANDARD "17" CACHE STRING "")

if(NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
   message(STATUS "Build type not specified, defaulting to ${CMAKE_BUILD_TYPE}")
endif()

# Get Google benchmark -- either from the system or from github
find_package(benchmark QUIET)
if (NOT benchmark_FOUND)
   message(STATUS "Could not find a local installation of google benchmark.")
   message(STATUS "It will be fetched from github and built together with the project.")
   include(FetchContent)
   FetchContent_Declare(
      benchmark
      GIT_REPOSITORY https://github.com/google/benchmark
      GIT_TAG 1.7.1
   )
   FetchContent_Populate(benchmark)
   set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON CACHE BOOL "Download gtest if needed")
   set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable building of gbench tests")
   set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "Disable installation of gbench (it's a private dependency)")
   add_subdirectory(${benchmark_SOURCE_DIR} ${benchmark_BINARY_DIR})
   add_library(benchmark::benchmark ALIAS benchmark)

endif()

add_executable(bench bench.cpp)
target_link_libraries(bench PRIVATE benchmark::benchmark)
install(TARGETS bench RUNTIME)
