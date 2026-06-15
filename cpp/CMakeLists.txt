cmake_minimum_required(VERSION 3.20)
project(UnBARableAI)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Dependencies
find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)
find_package(pybind11 REQUIRED)

# ------------------------------ (start) gRPC: UnBARableAI proto ------------------------------------
set(PROTO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../proto")
set(GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")

file(MAKE_DIRECTORY "${GENERATED_DIR}")

set(PROTO_NAME "UnBARableAI")
set(PROTO_FILE "${PROTO_DIR}/${PROTO_NAME}.proto")

set(GENERATED_PROTO_SRCS
    "${GENERATED_DIR}/${PROTO_NAME}.pb.cc"
    "${GENERATED_DIR}/${PROTO_NAME}.grpc.pb.cc"
)

set(GENERATED_PROTO_HDRS
    "${GENERATED_DIR}/${PROTO_NAME}.pb.h"
    "${GENERATED_DIR}/${PROTO_NAME}.grpc.pb.h"
)

set(PROTOC_EXE $<TARGET_FILE:protobuf::protoc>)
set(GRPC_CPP_PLUGIN $<TARGET_FILE:gRPC::grpc_cpp_plugin>)

add_custom_command(
    OUTPUT ${GENERATED_PROTO_SRCS} ${GENERATED_PROTO_HDRS}
    COMMAND ${PROTOC_EXE}
    ARGS
        --proto_path=${PROTO_DIR}
        --cpp_out=${GENERATED_DIR}
        --grpc_out=${GENERATED_DIR}
        --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
        ${PROTO_FILE}
    DEPENDS ${PROTO_FILE}
    COMMENT "Generating C++ protobuf/gRPC sources"
    VERBATIM
)


add_library(${PROTO_NAME}_proto STATIC
    ${GENERATED_PROTO_SRCS}
    ${GENERATED_PROTO_HDRS}
)

target_include_directories(${PROTO_NAME}_proto PUBLIC
    ${GENERATED_DIR}
)

target_link_libraries(${PROTO_NAME}_proto PUBLIC
    gRPC::grpc++
    protobuf::libprotobuf
)
# ------------------------------ (end) gRPC: UnBARableAI proto ------------------------------------

add_executable(tests
    "tests/memory/shared_memory_test.cpp"
    "tests/memory/shared_memory_posix_test.cpp"
    "tests/memory/partially_linked_list_information_test.cpp"
    "tests/memory/segment_information_test.cpp"
    "src/memory/shared_memory_posix.cpp"
)

target_include_directories(tests PUBLIC
    "src/"
)

target_link_libraries(tests
    GTest::gtest_main
)
add_executable(send_event_client
    send_event_client.cpp
)

target_link_libraries(send_event_client
    PRIVATE
        ${PROTO_NAME}_proto
)

# ------------------------------ generate docs ------------------------------------
add_custom_target(generate_docs_images
    COMMAND uv run run_all.py
            --out-dir generated/
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/../doc/scripts
)

add_custom_target(docs
    COMMAND doxygen Doxyfile
    DEPENDS generate_docs_images
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

