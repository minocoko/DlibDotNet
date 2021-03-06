cmake_minimum_required(VERSION 3.0.0)
set(PROJ_NAME DlibDotNet.Native)

project(${PROJ_NAME} VERSION 0.0.0)

# Only GCC requires -fPIC
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
	set(CMAKE_CXX_FLAGS "-fPIC")
endif()

# Select the release build type by default
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

if(MSVC)
	if (MSVC_VERSION GREATER_EQUAL "1900")
		include(CheckCXXCompilerFlag)
		CHECK_CXX_COMPILER_FLAG("/std:c++latest" _cpp_latest_flag_supported)
		if (_cpp_latest_flag_supported)
			add_compile_options("/std:c++latest")
		endif()
	else()
		set(CMAKE_CXX_STANDARD "17")
	endif()
else()
	set(CMAKE_CXX_STANDARD "17")
endif()

add_definitions(-DUNICODE -D_UNICODE)
add_subdirectory(${DLIB_PATH}/dlib dlib_build)

FILE(GLOB_RECURSE HEADERS ${PROJECT_SOURCE_DIR}/dlib/*.h)
FILE(GLOB_RECURSE SOURCES ${PROJECT_SOURCE_DIR}/dlib/*.cpp)
add_library(${PROJ_NAME} SHARED ${HEADERS} ${SOURCES})

target_link_libraries(${PROJ_NAME} dlib::dlib )

set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "/MP")
set(CompilerFlags
    CMAKE_CXX_FLAGS
    CMAKE_CXX_FLAGS_DEBUG
    CMAKE_CXX_FLAGS_RELEASE
    CMAKE_C_FLAGS
    CMAKE_C_FLAGS_DEBUG
    CMAKE_C_FLAGS_RELEASE
)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)