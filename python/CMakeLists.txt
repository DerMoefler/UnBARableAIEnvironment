cmake_minimum_required(VERSION 3.20)
project(bar_ai LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(bar_ai
    ../cpp/src/bindings/bindings.cpp
    ../cpp/src/unit/pawn.cpp
)

target_include_directories(bar_ai PRIVATE
    ../cpp/src
)

install(TARGETS bar_ai DESTINATION .)