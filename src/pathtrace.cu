/*
 ============================================================================
 Name        : pathtrace.cu
 Author      : Tudor Matei Boran
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA raytracer
 ============================================================================
 */
#include "cuda_error_check.h" // includes cuda.h and cuda_runtime_api.h
#include "pathtrace.h"
#include <cuda.h>

using namespace geom;
using namespace scene;

__global__ void renderKernel(geom::Triangle* d_triPtr, int numTriangles, Vector3Df* d_imgPtr, int width, int height);

Vector3Df* pathtraceWrapper(Scene& scene, int width, int height, int samples) {
	int pixels = width * height;
	size_t triangleBytes = sizeof(Triangle) * scene.getNumTriangles();
	size_t imageBytes = sizeof(Vector3Df) * width * height;

	// Initialize CUDA memory

	Triangle* triPtr = scene.getTriPtr();
	Triangle* d_triPtr = NULL;
	Vector3Df* imgDataPtr = new Vector3Df[pixels];
	Vector3Df* d_imgDataPtr = NULL;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_triPtr, triangleBytes));
	CUDA_CHECK_RETURN(cudaMemcpy((void*)d_triPtr, (void*)triPtr, triangleBytes, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_imgDataPtr, imageBytes));
	CUDA_CHECK_RETURN(cudaMemcpy((void*)d_imgDataPtr, (void*)imgDataPtr, imageBytes, cudaMemcpyHostToDevice));

	// Launch kernel
	dim3 block(blockWidth, blockWidth, 1);
	dim3 grid(width/blockWidth, height/blockWidth, 1);

	for (int s = 0; s < samples; s++)
	{
		// TODO: Pixels isn't required, or should be width and height if we want to support
		// frame sizes not multiple of blockSize^2
		renderKernel <<<grid, block>>>(d_triPtr, scene.getNumTriangles(), d_imgDataPtr, width, height);
	}

	CUDA_CHECK_RETURN(cudaMemcpy((void*)imgDataPtr, (void*)d_imgDataPtr, imageBytes, cudaMemcpyDeviceToHost));
	cudaFree((void*)d_triPtr);
	cudaFree((void*)d_imgDataPtr);
	return imgDataPtr;
}

__global__ void renderKernel(geom::Triangle* d_triPtr, int numTriangles, Vector3Df* d_imgPtr, int width, int height) {
	unsigned int i, j;
	i = blockIdx.x*blockDim.x + threadIdx.x;
	j = blockIdx.y*blockDim.y + threadIdx.y;
	float r = (float)i/(float)width;
	float g = (float)j/(float)height;
	d_imgPtr[j * width + i] = Vector3Df(r, g, 0.5f);
}

