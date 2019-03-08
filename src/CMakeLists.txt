project(CudaRT)
set(CMAKE_PROJECT_VERSION 0)
set(CMAKE_PROJECT_VERSION_MAJOR 0)
set(CMAKE_PROJECT_VERSION_MINOR 1)

cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
set(CMAKE_INCLUDE_CURRENT_DIR TRUE)

set(CUDA_RT_VERSION_MAJOR ${CMAKE_PROJECT_VERSION_MAJOR})
set(CUDA_RT_VERSION_MINOR ${CMAKE_PROJECT_VERSION_MINOR})

message("source: ${PROJECT_SOURCE_DIR}")
message("binary: ${PROJECT_BINARY_DIR}")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(CUDA 9.0 REQUIRED)

message("CUDA NVCC FLAGS: ${CUDA_NVCC_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(OpenGL_GL_PREFERENCE "GLVND")
find_package(OpenGL)
find_package(glfw3)
find_package(GLEW)

if (OPENGL_FOUND AND GLFW_FOUND AND GLEW_FOUND)
    include_directories(${OPENGL_INCLUDE_DIR})
    include_directories(${GLFW_INCLUDE_DIRS})
else()
	set(SKIP_OPENGL "#define SKIP_OPENGL")
    message("One of the OpenGL libraries was not found!")
endif()

set(CONTEXT "context")
set(GEOMETRY "geometry")
set(RENDERER "renderer")
set(CUDA_RT_INCLUDE "include")

configure_file(
	"${PROJECT_SOURCE_DIR}/include/config.h.in"
	"${PROJECT_SOURCE_DIR}/include/config.h"
)

include_directories(${CUDA_RT_INCLUDE}
				${CUDA_INCLUDE_DIRS}
				${CONTEXT}
				${GEOMETRY}
				${RENDERER})

set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_STANDARD 11)

list (APPEND CONTEXTLIB_FILES ${CONTEXT}/env_loader.cpp
                        ${CONTEXT}/json_loader.cpp
                        ${CONTEXT}/launcher.cpp
                        ${CONTEXT}/terminal_launcher.cpp
                        ${CONTEXT}/texture_store.cpp)
if (NOT SKIP_OPENGL)
	list(APPEND CONTEXTLIB_FILES ${CONTEXT}/window.cpp ${CONTEXT}/windowed_launcher.cpp)
endif()

cuda_add_library(contextlib STATIC ${CONTEXTLIB_FILES})

add_compile_options(-std=c++11)

target_link_libraries(contextlib ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GLFW_LIBRARIES})

if (${CMAKE_VERSION} VERSION_LESS "3.0.0")
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
endif()

cuda_add_executable(CudaRT
					${GEOMETRY}/bvh.cpp
					${GEOMETRY}/scene.cpp
					${GEOMETRY}/camera.cu
					${GEOMETRY}/geometry.cu
					${RENDERER}/parallel_renderer.cu
					${RENDERER}/renderer.cu
					${RENDERER}/sequential_renderer.cu
					main.cpp
					OPTIONS -arch=sm_52 -maxrregcount 64)
set_target_properties( CudaRT
			PROPERTIES CUDA_SEPARABLE_COMPILATION ON
			CUDA_STANDARD 11)
target_link_libraries(CudaRT contextlib
		 ${CUDA_LIBRARIES})
