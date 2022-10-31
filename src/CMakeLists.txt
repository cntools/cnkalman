add_library(cnkalman STATIC kalman.c ./model.cc ./numerical_diff.c ./iekf.c ./ModelPlot.cc
        ../include/cnkalman/kalman.h)

target_include_directories(cnkalman PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/>
  $<INSTALL_INTERFACE:include/>
  )

target_link_libraries(cnkalman cnmatrix)

install(TARGETS cnkalman DESTINATION lib)

pybind11_add_module(filter cnkalman_python_bindings.cpp)
target_link_libraries(filter PUBLIC cnkalman)
install(TARGETS filter DESTINATION lib)
