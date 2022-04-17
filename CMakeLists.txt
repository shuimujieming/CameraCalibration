cmake_minimum_required(VERSION 3.17)
project(CameraCali)
set(CMAKE_CXX_STANDARD 17)
#设置当前OpenCV目录
set(OpenCV_DIR "E:\\OpenCV\\opencv_build")

add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")


add_executable(CameraCali main.cpp)
link_directories(${OpenCV_LIBRARY_DIRS})

#查找OpenCV依赖
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#链接OpenCV库
target_link_libraries(CameraCali ${OpenCV_LIBS})
