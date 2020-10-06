cmake_minimum_required(VERSION 3.16)
project(ObjectDetection)
set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 11)

add_executable(ObjectDetection main.cpp)
target_link_libraries(ObjectDetection ${OpenCV_LIBS})