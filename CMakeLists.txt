cmake_minimum_required(VERSION 3.15)
project(MyProject)

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# 创建 demo 可执行文件
add_executable(demo src/demo.cpp)
target_include_directories(demo PRIVATE include)
target_compile_features(demo PRIVATE cxx_std_20)

# 创建 main 可执行文件
add_executable(main src/main.cpp)
target_include_directories(main PRIVATE include)
target_compile_features(main PRIVATE cxx_std_20)