# 0) creat source and cmakefile
# 1) makdir build
# 2) cd build
# 3) cmake ..
# 4) cmake --build .
# 5) ./Executable
cmake_minimum_required(VERSION 3.22)

set(executable "OpenCv_Basics")
project (
    CppProjectTemplate 
    VERSION 1.0.0
    LANGUAGES C CXX 
)
find_package(OpenCV REQUIRED)

add_executable (${executable} main.cc)

target_link_libraries(${executable} PRIVATE ${OpenCV_LIBS})

target_include_directories(${executable} PRIVATE ${OpenCV_INCLUDE_DIRS})