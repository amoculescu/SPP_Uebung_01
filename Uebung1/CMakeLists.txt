cmake_minimum_required(VERSION 3.11)
project(01 C)

set(CMAKE_C_STANDARD 11)

include_directories(src)

add_executable(01
        src/sorted_list.h
        src/sorted_list.c)
