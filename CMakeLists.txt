cmake_minimum_required(VERSION 3.20)
project(LinSolve VERSION 1.0 LANGUAGES CXX)

# Set the build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# C++20 Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the installation directories
set(CMAKE_INSTALL_INCLUDEDIR "include")
set(CMAKE_INSTALL_LIBDIR "lib")

# Define the source and header directories
set(LINSOLVE_SOURCE_DIR "${CMAKE_SOURCE_DIR}/source")
set(LINSOLVE_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/LinSolve")
set(LINSOLVE_TEST_DIR "${CMAKE_SOURCE_DIR}/test")
set(LINSOLVE_TENSORVIEW_DIR "${CMAKE_SOURCE_DIR}/LinSolve/TensorView")
set(LINSOLVE_NUMCEPTS_DIR "${CMAKE_SOURCE_DIR}/LinSolve/numcepts")

# Find LAPACK
find_package(LAPACK REQUIRED)

# TensorView is a CMake-based library, so add it as a subdirectory
add_subdirectory(${LINSOLVE_TENSORVIEW_DIR})

# Collect all the source files for the LinSolve library
file(GLOB LINSOLVE_SOURCES "${LINSOLVE_SOURCE_DIR}/*.cpp")

# Define the library
add_library(linsolve STATIC ${LINSOLVE_SOURCES})

# Set the include directories for both the library and the tests
target_include_directories(linsolve
    PRIVATE
        ${CMAKE_SOURCE_DIR}  # This makes the root project directory accessible for the source files
    PUBLIC
        ${LINSOLVE_INCLUDE_DIR}  # Public for use by other targets, including tests
)

# Link the LAPACK library
target_link_libraries(linsolve PRIVATE LAPACK::LAPACK)

# Set compile options
target_compile_options(linsolve
    PRIVATE
    $<$<CONFIG:Release>:-O3>
    -Wpedantic -Wall
)

# Install the LinSolve library and headers
install(TARGETS linsolve
        EXPORT LinSolveTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(FILES "${LINSOLVE_INCLUDE_DIR}/linsolve.hpp"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(DIRECTORY "${LINSOLVE_INCLUDE_DIR}/LinSolve/"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Test setup
enable_testing()

# Add the test executables
file(GLOB TEST_SOURCES "${LINSOLVE_TEST_DIR}/*.cpp")

foreach(TEST_FILE ${TEST_SOURCES})
    # Generate an executable for each test file
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_FILE})

    # Set the include directories for the test executables
    target_include_directories(${TEST_NAME}
        PRIVATE
            ${CMAKE_SOURCE_DIR}      # Ensure test executables can access the root project directory
    )

    # Link test executables with the LinSolve library and LAPACK
    target_link_libraries(${TEST_NAME} PRIVATE linsolve LAPACK::LAPACK)

    # Add the test
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()

# Optionally, you can add a custom target to run all tests
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND}
    DEPENDS ${TEST_SOURCES}
)
