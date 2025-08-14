cmake_minimum_required(VERSION 3.22)

set(executable "OpenCv_Basics")
project (
    CppProjectTemplate 
    VERSION 1.0.0
    LANGUAGES C CXX 
)
find_package(OpenCV REQUIRED)
file(COPY resources DESTINATION ${CMAKE_BINARY_DIR})


add_executable (${executable} main.cc)

target_link_libraries(${executable} PRIVATE ${OpenCV_LIBS})

target_include_directories(${executable} PRIVATE ${OpenCV_INCLUDE_DIRS})