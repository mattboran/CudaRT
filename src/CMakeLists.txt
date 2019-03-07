cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(CudaRT
	VERSION 0.1
	DESCRIPTION "CUDART"
	LANGUAGES CXX CUDA)

set(OpenGL_GL_PREFERENCE "GLVND")

message("source: ${PROJECT_SOURCE_DIR}")
message("binary: ${PROJECT_BINARY_DIR}")

find_package(PkgConfig REQUIRED)
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED)
pkg_search_module(GLFW REQUIRED glfw3)
if (GL_FOUND)
    include_directories(${GLFW_INCLUDE_DIRS})
endif()
include_directories(${OPENGL_INCLUDE_DIR} ${GLEW_INCLUDE_DIRS})

set(CONTEXT "context")
set(GEOMETRY "geometry")
set(RENDERER "renderer")
set(INCLUDE "include")

add_library(contextlib STATIC
	${CONTEXT}/env_loader.cpp
	${CONTEXT}/json_loader.cpp
	${CONTEXT}/launcher.cpp
	${CONTEXT}/terminal_launcher.cpp
	${CONTEXT}/texture_store.cpp
	${CONTEXT}/window.cpp
	${CONTEXT}/windowed_launcher.cpp)
target_include_directories(contextlib PUBLIC include /usr/include /usr/local/cuda/include)
target_compile_features(contextlib PUBLIC cxx_std_11)
target_link_libraries(contextlib ${OPENGL_STATIC_LIBRARIES} ${GLEW_STATIC_LIBRARIES} ${GLFW_STATIC_LIBRARIES} GLEW GL)

add_library(rendererlib STATIC
	${RENDERER}/parallel_renderer.cu
	${RENDERER}/renderer.cu
	${RENDERER}/sequential_renderer.cpp)

target_include_directories(rendererlib PUBLIC include /usr/include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_compile_features(rendererlib PUBLIC cxx_std_11)
target_compile_options(rendererlib PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_52,code=sm_52>)
set_target_properties(rendererlib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(rendererlib PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(geometrylib STATIC
	${GEOMETRY}/bvh.cpp
	${GEOMETRY}/scene.cpp
	${GEOMETRY}/camera.cu
	${GEOMETRY}/geometry.cu)

target_include_directories(geometrylib PUBLIC include /usr/include /usr/local/cuda/include)
target_compile_features(geometrylib PUBLIC cxx_std_11)
set_target_properties(geometrylib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(geometrylib PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(geometrylib contextlib)

add_executable(CudaRT main.cpp)
set_property(TARGET CudaRT PROPERTY CUDA_STANDARD 11)
set_target_properties( CudaRT
			PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(CudaRT contextlib)
target_link_libraries(CudaRT geometrylib)
target_link_libraries(CudaRT rendererlib)
