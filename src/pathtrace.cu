/*
 ============================================================================
 Name        : pathtrace.cu
 Author      : Tudor Matei Boran
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA raytracer
 ============================================================================
 */
#include "camera.cuh"
#include "pathtrace.h"
#include "cuda_error_check.h" // includes cuda.h and cuda_runtime_api.h
#include <cuda.h>

using namespace geom;

__global__ void renderKernel(geom::Triangle* d_triPtr, int numTriangles, Camera* d_camPtr, Vector3Df* d_imgPtr, int width, int height);
__device__ float intersectTriangles(geom::Triangle* d_triPtr, int numTriangles, unsigned int& hitId, const Ray& ray);

Vector3Df* pathtraceWrapper(Scene& scene, int width, int height, int samples) {
	int pixels = width * height;
	size_t triangleBytes = sizeof(Triangle) * scene.getNumTriangles();
	size_t imageBytes = sizeof(Vector3Df) * width * height;

	// Initialize CUDA memory

	Triangle* triPtr = scene.getTriPtr();
	Triangle* d_triPtr = NULL;
	Vector3Df* imgDataPtr = new Vector3Df[pixels];
	Vector3Df* d_imgDataPtr = NULL;
	Camera* camPtr = scene.getCameraPtr();
	Camera* d_camPtr = NULL;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_triPtr, triangleBytes));
	CUDA_CHECK_RETURN(cudaMemcpy((void*)d_triPtr, (void*)triPtr, triangleBytes, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_imgDataPtr, imageBytes));
	CUDA_CHECK_RETURN(cudaMemcpy((void*)d_imgDataPtr, (void*)imgDataPtr, imageBytes, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_camPtr, sizeof(Camera)));
	CUDA_CHECK_RETURN(cudaMemcpy((void*)d_camPtr, (void*)camPtr, sizeof(Camera), cudaMemcpyHostToDevice));

	// Launch kernel
	dim3 block(blockWidth, blockWidth, 1);
	dim3 grid(width/blockWidth, height/blockWidth, 1);

	for (int s = 0; s < samples; s++)
	{
		renderKernel <<<grid, block>>>(d_triPtr, scene.getNumTriangles(), d_camPtr, d_imgDataPtr, width, height);
	}

	CUDA_CHECK_RETURN(cudaMemcpy((void*)imgDataPtr, (void*)d_imgDataPtr, imageBytes, cudaMemcpyDeviceToHost));
	cudaFree((void*)d_triPtr);
	cudaFree((void*)d_imgDataPtr);
	return imgDataPtr;
}

__global__ void renderKernel(geom::Triangle* d_triPtr, int numTriangles, Camera* d_camPtr, Vector3Df* d_imgPtr, int width, int height) {
	unsigned int i, j;
	i = blockIdx.x*blockDim.x + threadIdx.x;
	j = blockIdx.y*blockDim.y + threadIdx.y;

	Ray camRay = d_camPtr->computeCameraRay(i, j);
	unsigned int hitId = 0;
	float t = intersectTriangles(d_triPtr, numTriangles, hitId, camRay);
	Vector3Df light(0.0f, 10.0f, 1.0f);
	if (t < MAX_DISTANCE) {
		Vector3Df hitPt = camRay.pointAlong(t);
		Vector3Df lightDir = light - hitPt;
		Vector3Df normal = d_triPtr[hitId]._normal;
		d_imgPtr[j * width + i] = Vector3Df(d_triPtr[hitId]._colorDiffuse * dot(lightDir, normal));
	}
}

__device__ float intersectTriangles(geom::Triangle* d_triPtr, int numTriangles, unsigned int& hitId, const Ray& ray) {
	float t = MAX_DISTANCE, tprime = MAX_DISTANCE;
	for (unsigned i = 0; i < numTriangles; i++)
	{
		tprime = d_triPtr[i].intersect(ray);
		if (tprime < t && tprime > 0.f)
		{
			t = tprime;
			hitId = i;
		}
	}
	return t;
}

