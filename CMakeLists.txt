cmake_minimum_required(VERSION 3.10)
project(CNN)

set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address,undefined -fno-omit-frame-pointer -O1 -g")

find_package(OpenCV REQUIRED)

add_executable(main.out main.cpp)
target_include_directories(main.out PRIVATE ${OpenCV_INCLUDE_DIRS} /usr/include/eigen3)
target_link_libraries(main.out PRIVATE ${OpenCV_LIBS})

add_executable(test.out test.cpp)
target_include_directories(test.out PRIVATE ${OpenCV_INCLUDE_DIRS} /usr/include/eigen3)
target_link_libraries(test.out PRIVATE ${OpenCV_LIBS})

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
