cmake_minimum_required(VERSION 3.20)
project(UnBARableAI VERSION 0.1.0 LANGUAGES CXX)

option(UNBARABLEAI_BUILD_PYTHON "Build Python bindings with pybind11" ON)
option(UNBARABLEAI_BUILD_TESTS "Build tests" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# Dependencies
find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)

if(UNBARABLEAI_BUILD_TESTS)
    find_package(GTest CONFIG REQUIRED)
endif()

if(UNBARABLEAI_BUILD_PYTHON)
    set(PYBIND11_FINDPYTHON ON)
    find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development)
    find_package(pybind11 CONFIG REQUIRED)
endif()

# ----------------------------------------------------------------------
# Protobuf / gRPC code generation
# ----------------------------------------------------------------------

set(PROTO_DIR        "${CMAKE_CURRENT_SOURCE_DIR}/proto")
set(PROTO_FILE       "${PROTO_DIR}/UnBARableAI.proto")
set(GENERATED_DIR    "${CMAKE_CURRENT_BINARY_DIR}/generated")

file(MAKE_DIRECTORY "${GENERATED_DIR}")

# protobuf_generate() ist bei find_package(Protobuf CONFIG REQUIRED)
# im Config-Mode verfügbar und kann C++- sowie gRPC-Code erzeugen. [1](https://github.com/protocolbuffers/protobuf/blob/main/docs/cmake_protobuf_generate.md)[2](https://cmake.org/cmake/help/latest/module/FindProtobuf.html)
#
# Wir hängen die .proto-Datei an ein Objekt-Target, damit CMake die
# erzeugten Dateien automatisch bei Änderungen neu erzeugt. [1](https://github.com/protocolbuffers/protobuf/blob/main/docs/cmake_protobuf_generate.md)[4](https://cmake.org/cmake/help/latest/command/add_library.html)
add_library(unbarableai_proto OBJECT "${PROTO_FILE}")

set_target_properties(unbarableai_proto PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(unbarableai_proto
    PUBLIC
        "$<BUILD_INTERFACE:${GENERATED_DIR}>"
)

target_link_libraries(unbarableai_proto
    PUBLIC
        protobuf::libprotobuf
        gRPC::grpc++
)

# Normale protobuf-C++ Dateien erzeugen:
#   UnBARableAI.pb.h / UnBARableAI.pb.cc
protobuf_generate(
    TARGET unbarableai_proto
    IMPORT_DIRS "${PROTO_DIR}"
    PROTOC_OUT_DIR "${GENERATED_DIR}"
)

# gRPC-Dateien erzeugen:
#   UnBARableAI.grpc.pb.h / UnBARableAI.grpc.pb.cc
#
# grpc_cpp_plugin ist Teil der gRPC-Tooling/Codegen-Schiene; Conan Center
# führt grpc standardmäßig mit codegen/cpp_plugin-Optionen. [3](https://conan.io/center/recipes/grpc)
protobuf_generate(
    TARGET unbarableai_proto
    LANGUAGE grpc
    GENERATE_EXTENSIONS .grpc.pb.h .grpc.pb.cc
    IMPORT_DIRS "${PROTO_DIR}"
    PROTOC_OUT_DIR "${GENERATED_DIR}"
    PLUGIN "protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>"
)

# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

if(UNBARABLEAI_BUILD_TESTS)
    add_executable(tests
        "tests/memory/shared_memory_test.cpp"
        "tests/memory/shared_memory_posix_test.cpp"
        "tests/memory/partially_linked_list_information_test.cpp"
        "tests/memory/segment_information_test.cpp"
        "tests/serialization/serialization_test.cpp"
        "src/memory/shared_memory_posix.cpp"
    )

    target_include_directories(tests
        PUBLIC
            "src/"
    )

    target_link_libraries(tests
        PRIVATE
            GTest::gtest_main
    )

    add_executable(send_event_client
        tests/grpc/send_event_client.cpp
    )
endif()

# ----------------------------------------------------------------------
# Docs
# ----------------------------------------------------------------------

add_custom_target(generate_docs_images
    COMMAND uv run run_all.py --out-dir generated/
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/../doc/scripts
)

add_custom_target(docs
    COMMAND doxygen Doxyfile
    DEPENDS generate_docs_images
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# ----------------------------------------------------------------------
# Core library
# ----------------------------------------------------------------------

add_library(unbarableai_core STATIC
    src/grpc_client/UnBARableAIClient.cpp
)

# Die erzeugten Proto-Objekte direkt in die Core-Library einbinden.
# Objekt-Libraries sind genau dafür gedacht: eine Sammlung von Objekten,
# die von einem realen Target verwendet werden kann. [4](https://cmake.org/cmake/help/latest/command/add_library.html)[5](https://cmake.org/cmake/help/latest/guide/tutorial/In-Depth%20CMake%20Library%20Concepts.html)
target_sources(unbarableai_core
    PRIVATE
        $<TARGET_OBJECTS:unbarableai_proto>
)

set_target_properties(unbarableai_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

add_library(UnBARableAI::unbarableai_core ALIAS unbarableai_core)

target_include_directories(unbarableai_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${GENERATED_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${GENERATED_DIR}
)

target_link_libraries(unbarableai_core
    PUBLIC
        protobuf::libprotobuf
        gRPC::grpc++
)

if(UNBARABLEAI_BUILD_TESTS)
    target_link_libraries(send_event_client
        PRIVATE
            unbarableai_core
    )
endif()

# ----------------------------------------------------------------------
# Install / export
# ----------------------------------------------------------------------

install(TARGETS unbarableai_core
    EXPORT UnBARableAITargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)


install(FILES
    "${GENERATED_DIR}/UnBARableAI.pb.h"
    "${GENERATED_DIR}/UnBARableAI.grpc.pb.h"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/UnBARableAI
)

install(EXPORT UnBARableAITargets
    FILE UnBARableAITargets.cmake
    NAMESPACE UnBARableAI::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/UnBARableAI
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/UnBARableAIConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/UnBARableAIConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/UnBARableAI
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/UnBARableAIConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/UnBARableAIConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/UnBARableAIConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/UnBARableAI
)