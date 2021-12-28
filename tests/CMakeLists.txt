include_directories(../include)

include_directories(.)

add_executable(TestKalmanModels ModelRunner.cc TestKalmanModels.cc models/LinearPoseVel.h)
target_link_libraries(TestKalmanModels cnkalman boost_iostreams boost_system boost_filesystem)

add_test(NAME runbenchmarks
        COMMAND ${CMAKE_COMMAND}
        -DTEST_KALMAN_MODELS=$<TARGET_FILE:TestKalmanModels>
        -DBENCHMARK_SCRIPT=${CMAKE_CURRENT_SOURCE_DIR}/filterpy/benchmark.py
        -P ${CMAKE_CURRENT_SOURCE_DIR}/runbenchmarks.cmake)

add_test(NAME TestKalmanModels COMMAND TestKalmanModels)

find_package(GTest)
if(GTest_FOUND)
    include(GoogleTest)

    add_executable(kalman_toy kalman_toy.cc)
    target_link_libraries(kalman_toy cnkalman GTest::gtest GTest::gtest_main)

    gtest_discover_tests(kalman_toy)
endif()
