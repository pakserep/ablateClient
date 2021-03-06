cmake_minimum_required(VERSION 3.14)

# FetContent will allow downloading of other libraries
include(FetchContent)

# ABLATE must be available for the project.  If the ABLATE_PATH is not specified it will download the version specified in FetchContent_Declare
set(ABLATE_PATH "" CACHE PATH "Custom path to local ABLATE project.  If specified this will be used instead of downloading ABLATE.")
message(ABLATE_PATH ${ABLATE_PATH})

if("${ABLATE_PATH}" STREQUAL "")
    message(STATUS "ABLATE_PATH was not specified.  ABLATE is being downloaded and built.")

    # Download and make ablate available to the client
    FetchContent_Declare(
            ablate
            GIT_REPOSITORY https://github.com/UBCHREST/ablate.git
            GIT_TAG v0.3.8
    )
    FetchContent_MakeAvailable(ablate)

else()
    message(STATUS "Configuring with ABLATE_PATH: ${ABLATE_PATH}")
    add_subdirectory(${ABLATE_PATH} ${CMAKE_CURRENT_BINARY_DIR}/ablate-build-directory)
endif()

# Set the project details
project(pedramClient VERSION 0.0.1)

# Set the c/c++ Standards
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# define the two sample programs and link to ablate
add_executable(ablateLibraryClient ablateLibraryClient.cpp)
target_link_libraries(ablateLibraryClient PUBLIC ablateLibrary)

# define the two sample programs
add_executable(ablateCoreClient ablateCoreClient.c)
target_link_libraries(ablateCoreClient PUBLIC ablateCore)

# define cavity example
add_executable(cavity_case cavity.c)
target_link_libraries(cavity_case PUBLIC ablateCore)

# define a vector test case
add_executable(test_vec test_vec.c)
target_link_libraries(test_vec PUBLIC ablateCore)

# define ex4 given by Dr. Knepley
add_executable(ex4_case ex4.c)
target_link_libraries(ex4_case PUBLIC ablateCore)

# define quiescent fluid case
add_executable(quiescentFluid_case quiescentFluid.c)
target_link_libraries(quiescentFluid_case PUBLIC ablateCore)

# define some examples by Matt McGurn
add_executable(pointerCase pointerExp.c)
add_executable(structCase structExp.c)
#target_link_libraries(mattTest_case PUBLIC ablateCore)

add_executable(cavity cavity.c)
target_link_libraries(cavity PUBLIC ablateCore)

