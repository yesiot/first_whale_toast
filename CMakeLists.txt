cmake_minimum_required(VERSION 3.10)
project(opencv_hist)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 17)

include_directories(${OpenCV_INCLUDES})

add_executable(opencv_hist main.cpp)

target_link_libraries(opencv_hist ${OpenCV_LIBS})
