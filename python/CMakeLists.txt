cmake_minimum_required(VERSION 3.20)

# -----------------------------Pybind shit--------------------------------------
project(bar_ai LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(bar_ai
    ../cpp/src/bindings/bindings.cpp
    ../cpp/include/UnBARableAI/unit_data.h
)

target_include_directories(bar_ai PRIVATE
    ../cpp/src
)

install(TARGETS bar_ai DESTINATION .)

# ----------------------grpc-tools für Python---------------------------

find_package(Python REQUIRED)

# protoc über Python ausführen
set(GENERATED_DIR ${CMAKE_SOURCE_DIR}/generated)
set(PROTO_DIR ${CMAKE_SOURCE_DIR}/../cpp/proto)

file(MAKE_DIRECTORY ${GENERATED_DIR})


add_custom_command(
    OUTPUT
        ${GENERATED_DIR}/UnBARableAI_pb2.py
        ${GENERATED_DIR}/UnBARableAI_pb2_grpc.py
    COMMAND ${Python_EXECUTABLE} -m grpc_tools.protoc
        -I${PROTO_DIR}
        --python_out=${GENERATED_DIR}
        --grpc_python_out=${GENERATED_DIR}
        ${PROTO_DIR}/UnBARableAI.proto
    DEPENDS ${PROTO_DIR}/UnBARableAI.proto
)

add_custom_target(generate_grpc ALL
    DEPENDS
        ${GENERATED_DIR}/UnBARableAI_pb2.py
        ${GENERATED_DIR}/UnBARableAI_pb2_grpc.py
)