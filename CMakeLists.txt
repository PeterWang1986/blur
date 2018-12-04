project(blur)

set(src_list "local_blur_based_edge.cc")

Find_package(OpenCV 3.4 REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")

add_library(blur STATIC ${src_list})
target_link_libraries(blur 
    ${OpenCV_LIBS}
)

add_executable(local_blur_based_edge_test "local_blur_based_edge_test.cc")
target_link_libraries(local_blur_based_edge_test
  blur
)

