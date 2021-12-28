include_directories(../include)

include_directories(.)

add_executable(TestKalmanModels ModelRunner.cc TestKalmanModels.cc models/LinearPoseVel.h model.h)
target_link_libraries(TestKalmanModels cnkalman)
target_link_libraries(TestKalmanModels boost_iostreams boost_system boost_filesystem survive)

add_test(NAME runbenchmarks
        COMMAND ${CMAKE_COMMAND}
        -DTEST_KALMAN_MODELS=$<TARGET_FILE:TestKalmanModels>
        -DBENCHMARK_SCRIPT=${CMAKE_CURRENT_SOURCE_DIR}/filterpy/benchmark.py
        -DCMAKE_CURRENT_BINARY_DIR1=${CMAKE_CURRENT_BINARY_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/runbenchmarks.cmake)

add_test(TestKalmanModels TestKalmanModels)