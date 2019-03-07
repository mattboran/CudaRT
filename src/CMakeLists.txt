cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(CudaRT
	VERSION 0.1
	DESCRIPTION "CUDART"
	LANGUAGES CXX CUDA)

message("source: ${PROJECT_SOURCE_DIR}")
message("binary: ${PROJECT_BINARY_DIR}")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(CUDA REQUIRED)
string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_52,code=sm_52")
set(OpenGL_GL_PREFERENCE "GLVND")
find_package(OpenGL)
find_package(glfw3)
find_package(GLEW)

if (OPENGL_FOUND AND GLFW_FOUND AND GLEW_FOUND)
    include_directories(${OPENGL_INCLUDE_DIR})
    include_directories(${GLFW_INCLUDE_DIRS})
else()
    message("One of the OpenGL libraries was not found!")
endif()

set(CONTEXT "context")
set(GEOMETRY "geometry")
set(RENDERER "renderer")
set(CUDA_RT_INCLUDE "include")

add_library(contextlib STATIC
	${CONTEXT}/env_loader.cpp
	${CONTEXT}/json_loader.cpp
	${CONTEXT}/launcher.cpp
	${CONTEXT}/terminal_launcher.cpp
	${CONTEXT}/texture_store.cpp
	${CONTEXT}/window.cpp
	${CONTEXT}/windowed_launcher.cpp)
target_include_directories(contextlib PUBLIC ${CUDA_RT_INCLUDE} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_compile_features(contextlib PUBLIC cxx_std_11)
target_link_libraries(contextlib ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GLFW_LIBRARIES})

add_library(rendererlib STATIC
	${RENDERER}/parallel_renderer.cu
	${RENDERER}/renderer.cu
	${RENDERER}/sequential_renderer.cpp)

target_include_directories(rendererlib PUBLIC ${CUDA_RT_INCLUDE} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_compile_features(rendererlib PUBLIC cxx_std_11)
set_target_properties(rendererlib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_library(geometrylib STATIC
	${GEOMETRY}/bvh.cpp
	${GEOMETRY}/scene.cpp
	${GEOMETRY}/camera.cu
	${GEOMETRY}/geometry.cu)

target_include_directories(geometrylib PUBLIC ${CUDA_RT_INCLUDE} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_compile_features(geometrylib PUBLIC cxx_std_11)
set_target_properties(geometrylib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(geometrylib contextlib)

add_executable(CudaRT main.cpp)
set_target_properties( CudaRT
						PROPERTIES CUDA_SEPARABLE_COMPILATION ON
						CUDA_STANDARD 11)
target_link_libraries(CudaRT contextlib
							 geometrylib
							 rendererlib)
