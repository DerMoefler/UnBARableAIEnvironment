cmake_minimum_required(VERSION 3.20)
project(UnBARableAI VERSION 0.1.0 LANGUAGES CXX)
option(UNBARABLEAI_BUILD_PYTHON "Build Python bindings with pybind11" ON)
option(UNBARABLEAI_BUILD_TESTS "Build tests" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Dependencies
find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)

if(UNBARABLEAI_BUILD_TESTS)
    find_package(GTest CONFIG REQUIRED)
endif()

if(UNBARABLEAI_BUILD_PYTHON)
    # pybind11 3.x / aktuelle Releases erwarten modernes Python
    # und der Recoil-Buildcontainer hat nur Python 3.6.9.
    # Deshalb Python-Bindings nur optional in einer passenden Umgebung bauen.
    set(PYBIND11_FINDPYTHON ON)
    find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development)
    find_package(pybind11 CONFIG REQUIRED)
endif()

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# ------------------------------ (start) gRPC: UnBARableAI proto ------------------------------------
set(PROTO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/proto")
set(GENERATED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/generated")

set(PROTO_NAME "UnBARableAI")

set(GENERATED_PROTO_SRCS
    "${GENERATED_DIR}/${PROTO_NAME}.pb.cc"
    "${GENERATED_DIR}/${PROTO_NAME}.grpc.pb.cc"
)

set(GENERATED_PROTO_HDRS
    "${GENERATED_DIR}/${PROTO_NAME}.pb.h"
    "${GENERATED_DIR}/${PROTO_NAME}.grpc.pb.h"
)

foreach(_f IN LISTS GENERATED_PROTO_SRCS GENERATED_PROTO_HDRS)
    if(NOT EXISTS "${_f}")
        message(FATAL_ERROR
            "Missing generated protobuf/grpc file: ${_f}\n"
            "Generate the files on the host before building Recoil."
        )
    endif()
endforeach()



# ------------------------------ (end) gRPC: UnBARableAI proto ------------------------------------
if(UNBARABLEAI_BUILD_TESTS)
    add_executable(tests
        "tests/memory/shared_memory_test.cpp"
        "tests/memory/shared_memory_posix_test.cpp"
        "tests/memory/partially_linked_list_information_test.cpp"
        "tests/memory/segment_information_test.cpp"
        "tests/serialization/serialization_test.cpp"
        "src/memory/shared_memory_posix.cpp"
    )

    target_include_directories(tests PUBLIC
        "src/"
    )

    target_link_libraries(tests
        GTest::gtest_main
    )
    add_executable(send_event_client
        tests/grpc/send_event_client.cpp
    )
endif()


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


#-------------------------------------client library--------------------------------------
add_library(unbarableai_core STATIC
    src/grpc_client/UnBARableAIClient.cpp
    ${GENERATED_PROTO_SRCS}
    ${GENERATED_PROTO_HDRS}
)

if(UNBARABLEAI_BUILD_TESTS)
    target_link_libraries(send_event_client
        PRIVATE
            unbarableai_core
    )
endif()

set_target_properties(unbarableai_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)


add_library(UnBARableAI::unbarableai_core ALIAS unbarableai_core)

target_include_directories(unbarableai_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${GENERATED_DIR}
)

target_link_libraries(unbarableai_core
    PUBLIC
        gRPC::grpc++
        protobuf::libprotobuf
)

if(UNBARABLEAI_BUILD_TESTS)
    target_link_libraries(send_event_client
        PRIVATE
            unbarableai_core
    )
endif()

install(TARGETS unbarableai_core
    EXPORT UnBARableAITargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
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