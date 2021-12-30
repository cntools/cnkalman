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

set(cnkalman_root_source_dir ${CMAKE_SOURCE_DIR})
execute_process(COMMAND  ${CMAKE_COMMAND} -E env PYTHONPATH=${cnkalman_root_source_dir} python3 -c "import cnkalman.codegen as cg"
        RESULT_VARIABLE INVALID_PYTHON3 ERROR_QUIET)
set(INVALID_PYTHON3 ${INVALID_PYTHON3} CACHE BOOL "Whether the version of python supports generation or not" FORCE)
function(cnkalman_generate_code FILE)
if(NOT INVALID_PYTHON3)
  get_filename_component(DIR ${CMAKE_CURRENT_SOURCE_DIR}/${FILE} DIRECTORY)
  get_filename_component(NAME ${CMAKE_CURRENT_SOURCE_DIR}/${FILE} NAME_WE)
  message(${DIR}/${NAME}.gen.h)
  add_custom_command(OUTPUT ${DIR}/${NAME}.gen.h COMMAND
          ${CMAKE_COMMAND} -E env PYTHONPATH=${cnkalman_root_source_dir} python3 ${CMAKE_CURRENT_SOURCE_DIR}/${FILE} --cnkalman-generate-source
          COMMENT "Generating source...")
  endif()
endfunction()

if(${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
if(ENABLE_TESTS)
  add_subdirectory(tests)
endif()
endif()


install(DIRECTORY include/cnkalman DESTINATION include)
include(GNUInstallDirs)
configure_file(cnkalman.pc.in cnkalman.pc @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/cnkalman.pc" DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig")


