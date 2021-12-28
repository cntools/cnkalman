cmake_minimum_required(VERSION 3.12)

project(cnkalman LANGUAGES C CXX)
include(CTest)

include(CheckIncludeFile)
include (CheckSymbolExists)

set(CMAKE_CXX_STANDARD 17)

SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -fsanitize=undefined")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize=undefined")

add_subdirectory(libs/cnmatrix)

add_subdirectory(src)

if(ENABLE_TESTS)
  add_subdirectory(tests)
endif()


