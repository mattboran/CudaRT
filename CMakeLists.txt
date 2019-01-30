cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(CudaRT
	VERSION 0.1
	DESCRIPTION "CUDART"
	LANGUAGES CXX CUDA)

set(OpenGL_GL_PREFERENCE "GLVND")
set(glfw_DIR "/usr/lib/x86_64-linux-gnu/cmake/glfw3")

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
include_directories(${GLFW_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIR} ${GLEW_INCLUDE_DIRS})

set(SRC "src/")
set(CONTEXT "src/context")
set(GEOMETRY "src/geometry")
set(RENDERER "src/renderer")
set(INCLUDE "src/include")

add_library(contextlib STATIC
	${CONTEXT}/launcher.cpp
	${CONTEXT}/terminal_launcher.cpp
	${CONTEXT}/window.cpp
	${CONTEXT}/windowed_launcher.cpp)
target_include_directories(contextlib PUBLIC src/include /usr/include /usr/local/cuda/include)
target_compile_features(contextlib PUBLIC cxx_std_11)
set_target_properties(contextlib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(contextlib PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(contextlib ${OPENGL_STATIC_LIBRARIES} ${GLEW_STATIC_LIBRARIES} ${GLFW_STATIC_LIBRARIES})

add_library(rendererlib STATIC
	${RENDERER}/parallel_renderer.cu
	${RENDERER}/renderer.cu
	${RENDERER}/sequential_renderer.cpp)

target_include_directories(rendererlib PUBLIC src/include /usr/include /usr/local/cuda/include)
target_compile_features(rendererlib PUBLIC cxx_std_11)
set_target_properties(rendererlib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(rendererlib PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(geometrylib STATIC
	${GEOMETRY}/bvh.cpp
	${GEOMETRY}/scene.cpp
	${GEOMETRY}/camera.cu
	${GEOMETRY}/geometry.cu)

target_include_directories(geometrylib PUBLIC src/include /usr/include /usr/local/cuda/include)
target_compile_features(geometrylib PUBLIC cxx_std_11)
set_target_properties(geometrylib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(geometrylib PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(CudaRT src/main.cpp)
set_property(TARGET CudaRT PROPERTY CUDA_STANDARD 11)
set_target_properties( CudaRT
			PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(CudaRT contextlib)
target_link_libraries(CudaRT geometrylib)
target_link_libraries(CudaRT rendererlib)
target_link_libraries(CudaRT ${OPENGL_STATIC_LIBRARIES} ${GLEW_STATIC_LIBRARIES} ${GLFW_STATIC_LIBRARIES} GLEW GL)
