add_library(cnkalman ./survive_kalman.c ./model.cc ./numerical_diff.c ./iekf.c)

target_include_directories(cnkalman PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/>
  $<INSTALL_INTERFACE:include/>
  )

target_link_libraries(cnkalman cnmatrix)
