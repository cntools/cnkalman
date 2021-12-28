cmake_minimum_required(VERSION 3.12)

project(cnkalman LANGUAGES C CXX)
include(CTest)

include(CheckIncludeFile)
include (CheckSymbolExists)

set(CMAKE_CXX_STANDARD 17)

option(USE_ASAN "Use address sanitizer" OFF)

if(UNIX)
  add_compile_options(-fPIC -Wall -Wno-unused-variable -Wno-switch -Wno-parentheses -Wno-missing-braces -Werror=return-type -fvisibility=hidden -Werror=vla -fno-math-errno -Werror=pointer-arith)
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SHARED_FLAGS} -std=gnu99 -Werror=incompatible-pointer-types -Werror=implicit-function-declaration -Werror=missing-field-initializers ")

  if(USE_ASAN)
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
  endif()
endif()

add_subdirectory(libs/cnmatrix)
add_subdirectory(src)

if(ENABLE_TESTS)
  add_subdirectory(tests)
endif()


install(DIRECTORY include/cnkalman DESTINATION include)
include(GNUInstallDirs)
configure_file(cnkalman.pc.in cnkalman.pc @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/cnkalman.pc" DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig")


