cmake_minimum_required(VERSION 3.22)
set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
SET(CMAKE_TOOLCHAIN_FILE "/home/fanghz/lib/vcpkg/scripts/buildsystems/vcpkg.cmake")
project(example)
set(CMAKE_CXX_STANDARD 20)
#SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_BUILD_TYPE "Release")
find_package(Threads REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(TBB CONFIG REQUIRED)
find_package(xsimd CONFIG REQUIRED)
find_package(eve CONFIG REQUIRED)
find_package(PkgConfig)
pkg_check_modules(liburing REQUIRED IMPORTED_TARGET GLOBAL liburing>=2.0)



add_executable(example test.cc)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(example PRIVATE -O3 -march=native -fopt-info-vec-optimized)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(example PRIVATE -O3 -march=native -Rpass=loop-vectorize -Rpass-analysis=loop-vectorize -Rpass-missed=loop-vectorize)
endif ()


target_link_libraries(example xsimd)
target_link_libraries(example eve::eve)
target_link_libraries(example fmt::fmt)
target_link_libraries(example PkgConfig::liburing)
target_link_libraries(example TBB::tbb TBB::tbbmalloc)

