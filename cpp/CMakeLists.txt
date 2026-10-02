cmake_minimum_required(VERSION 3.20)
project(UnBARableAI VERSION 0.1.0 LANGUAGES CXX)

option(UNBARABLEAI_BUILD_PYTHON "Build Python bindings with pybind11" ON)
option(UNBARABLEAI_BUILD_TESTS "Build tests" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CMakePackageConfigHelpers)
include(CTest)
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
    enable_testing()
    include(GoogleTest)

    add_executable(tests
        "tests/memory/bar_shared_memory_test.cpp"
        "tests/memory/shared_memory_test.cpp"
        "tests/memory/shared_memory_posix_test.cpp"
        "tests/memory/partially_linked_list_information_test.cpp"
        "tests/memory/segment_information_test.cpp"
        "tests/serialization/layout_test.cpp"
        "tests/serialization/serialization_test.cpp"
        "src/id/id_allocator.cpp"
        "src/memory/debug/hexdump.cpp"
        "src/memory/bar_shared_memory.cpp"
        "src/memory/partially_linked_list_information.cpp"
        "src/memory/segment_information.cpp"
        "src/memory/shared_memory_posix.cpp"
    )

    target_include_directories(tests
        PUBLIC
            "include/"
            "src/"
    )

    target_link_libraries(tests
        PRIVATE
            GTest::gtest_main
    )

    gtest_discover_tests(tests)

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
# ----------------------------------------------------------------------
# Core library
# ----------------------------------------------------------------------

# Nur Quellen aufnehmen, die tatsächlich zur Core-Library gehören.
# bindings/ und engine_bridge/ sind separate Komponenten.

file(
    GLOB_RECURSE UNBARABLEAI_CORE_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/grpc_client/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/memory/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/id/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/serialization/*.cpp"
)

add_library(unbarableai_core STATIC
    ${UNBARABLEAI_CORE_SOURCES}
    $<TARGET_OBJECTS:unbarableai_proto>
)

set_target_properties(unbarableai_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

add_library(
    UnBARableAI::unbarableai_core
    ALIAS unbarableai_core
)

# bar_shared_memory.h verwendet:
#   <UnBARableAI/src/memory/shared_memory.h>
#
# Deshalb wird für den Build dieselbe Struktur erzeugt,
# die später auch im Installationsverzeichnis vorhanden ist.
set(
    UNBARABLEAI_BUILD_INCLUDE_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/generated-include"
)

set(
    UNBARABLEAI_BUILD_SRC_LINK
    "${UNBARABLEAI_BUILD_INCLUDE_DIR}/UnBARableAI/src"
)

file(MAKE_DIRECTORY
    "${UNBARABLEAI_BUILD_INCLUDE_DIR}/UnBARableAI"
)

if(EXISTS "${UNBARABLEAI_BUILD_SRC_LINK}"
   OR IS_SYMLINK "${UNBARABLEAI_BUILD_SRC_LINK}")
    file(REMOVE_RECURSE "${UNBARABLEAI_BUILD_SRC_LINK}")
endif()

file(
    CREATE_LINK
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${UNBARABLEAI_BUILD_SRC_LINK}"
    SYMBOLIC
    RESULT UNBARABLEAI_SRC_LINK_RESULT
)

if(NOT UNBARABLEAI_SRC_LINK_RESULT STREQUAL "0")
    message(FATAL_ERROR
        "Failed to create UnBARableAI src include link: "
        "${UNBARABLEAI_SRC_LINK_RESULT}"
    )
endif()

target_include_directories(unbarableai_core
    PUBLIC
        # <UnBARableAI/...>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>

        # "serialization/...", "id/..." usw.
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>

        # <UnBARableAI/src/...>
        $<BUILD_INTERFACE:${UNBARABLEAI_BUILD_INCLUDE_DIR}>

        # Generierte *.pb.h und *.grpc.pb.h
        $<BUILD_INTERFACE:${GENERATED_DIR}>

        # Installierte Package-Struktur
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/UnBARableAI/src>
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

# Gesamten cpp/src-Verzeichnisbaum übernehmen
install(
    DIRECTORY
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
    DESTINATION
        "${CMAKE_INSTALL_INCLUDEDIR}/UnBARableAI"
    FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp"
        PATTERN "*.tpp"
        PATTERN "*.inl"
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
