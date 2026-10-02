cmake_minimum_required(VERSION 3.20)

project(bar_ai LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ----------------------------------------------------------------------
# Python / pybind11
# ----------------------------------------------------------------------

set(PYBIND11_FINDPYTHON ON)

find_package(
    Python REQUIRED
    COMPONENTS
        Interpreter
        Development.Module
)

find_package(pybind11 CONFIG REQUIRED)

set(CPP_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../cpp"
)

# Nur die C++-Implementierungen einbinden, die das Binding tatsächlich nutzt.
pybind11_add_module(bar_ai
    "${CPP_DIR}/src/bindings/bindings.cpp"

    "${CPP_DIR}/src/id/id_allocator.cpp"

    "${CPP_DIR}/src/memory/bar_shared_memory.cpp"
    "${CPP_DIR}/src/memory/shared_memory_posix.cpp"
    "${CPP_DIR}/src/memory/partially_linked_list_information.cpp"
    "${CPP_DIR}/src/memory/segment_information.cpp"
    "${CPP_DIR}/src/memory/debug/hexdump.cpp"
)

target_compile_features(bar_ai
    PRIVATE
        cxx_std_20
)

target_include_directories(bar_ai
    PRIVATE
        # Für <UnBARableAI/...>
        "${CPP_DIR}/include"

        # Für <memory/...>, <serialization/...> und <id/...>
        "${CPP_DIR}/src"
)

install(
    TARGETS bar_ai
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
)

# ----------------------------------------------------------------------
# Python gRPC generation
# ----------------------------------------------------------------------

set(
    GENERATED_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/generated"
)

set(
    PROTO_DIR
    "${CPP_DIR}/proto"
)

set(
    PROTO_FILE
    "${PROTO_DIR}/UnBARableAI.proto"
)

file(MAKE_DIRECTORY "${GENERATED_DIR}")

add_custom_command(
    OUTPUT
        "${GENERATED_DIR}/UnBARableAI_pb2.py"
        "${GENERATED_DIR}/UnBARableAI_pb2_grpc.py"
    COMMAND
        "${Python_EXECUTABLE}" -m grpc_tools.protoc
        "-I${PROTO_DIR}"
        "--python_out=${GENERATED_DIR}"
        "--grpc_python_out=${GENERATED_DIR}"
        "${PROTO_FILE}"
    DEPENDS
        "${PROTO_FILE}"
    VERBATIM
)

add_custom_target(generate_grpc ALL
    DEPENDS
        "${GENERATED_DIR}/UnBARableAI_pb2.py"
        "${GENERATED_DIR}/UnBARableAI_pb2_grpc.py"
)