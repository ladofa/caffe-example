cmake_minimum_required(VERSION 3.0.0)
set (CMAKE_CXX_STANDARD 11)
find_package(OpenCV)
find_package(Caffe REQUIRED)

add_executable(xor xor.cpp)
target_link_libraries(xor ${Caffe_LIBRARIES})
include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS}) # ex. -DCPU_ONLY


