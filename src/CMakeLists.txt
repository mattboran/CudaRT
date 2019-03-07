#cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(CudaRT)
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
set(CMAKE_INCLUDE_CURRENT_DIR TRUE)

set(CUDA_RT_VERSION_MAJOR 0)
set(CUDA_RT_VERSION_MINOR 1)

message("source: ${PROJECT_SOURCE_DIR}")
message("binary: ${PROJECT_BINARY_DIR}")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS
	${CUDA_NVCC_FLAGS};
	-O3 -gencode arch=compute_52,code=sm_52
)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#message("CUDA include dirs: ${CUDA_INCLUDE_DIRS}")
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

if (SKIP_OPENGL)
cuda_add_library(contextlib STATIC
	${CONTEXT}/env_loader.cpp
	${CONTEXT}/json_loader.cpp
	${CONTEXT}/launcher.cpp
	${CONTEXT}/terminal_launcher.cpp
	${CONTEXT}/texture_store.cpp
)
else ()
cuda_add_library(contextlib STATIC
	${CONTEXT}/env_loader.cpp
	${CONTEXT}/json_loader.cpp
	${CONTEXT}/launcher.cpp
	${CONTEXT}/terminal_launcher.cpp
	${CONTEXT}/texture_store.cpp
	${CONTEXT}/window.cpp
	${CONTEXT}/windowed_launcher.cpp
)
endif()

add_compile_options(-std=c++11)
target_link_libraries(contextlib ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GLFW_LIBRARIES})

cuda_add_executable(CudaRT 
					${GEOMETRY}/bvh.cpp
					${GEOMETRY}/scene.cpp
					${GEOMETRY}/camera.cu
					${GEOMETRY}/geometry.cu
					${RENDERER}/parallel_renderer.cu
					${RENDERER}/renderer.cu
					${RENDERER}/sequential_renderer.cu
					main.cpp)
set_target_properties( CudaRT
						PROPERTIES CUDA_SEPARABLE_COMPILATION ON
						CUDA_STANDARD 11)
target_link_libraries(CudaRT contextlib
							 ${CUDA_LIBRARIES})
