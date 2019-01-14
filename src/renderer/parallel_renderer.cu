/*
 * parallel_renderer.cpp
 *
 *  Created on: Dec 22, 2018
 *      Author: matt
 */

#include "renderer.h"
#include "cuda_error_check.h"
#include <curand_kernel.h>
#include <iostream>

using std::cout;

#define BLOCK_WIDTH 16u

// Kernels
__global__ void initializeCurandKernel(curandState* p_curandState);
__global__ void renderKernel(SettingsData settings,
		Vector3Df* p_imgBuffer,
		uchar4* p_outImg,
		Camera* p_camera,
		TrianglesData* p_tris,
		LightsData* p_lights,
		curandState *p_curandState,
		int sampleNumber);

__host__ ParallelRenderer::ParallelRenderer(Scene* _scenePtr, int _width, int _height, int _samples, bool _useBVH) :
	Renderer(_scenePtr, _width, _height, _samples, _useBVH) {
	// CUDA settings
	useCuda = true;
	threadsPerBlock = BLOCK_WIDTH * BLOCK_WIDTH;
	gridBlocks = width / BLOCK_WIDTH * height / BLOCK_WIDTH;

	int pixels = width * height;
	unsigned int numTris = p_scene->getNumTriangles();
	unsigned int numBvhNodes = p_scene->getNumBvhNodes();
	unsigned int numLights = p_scene->getNumLights();
	size_t trianglesBytes = sizeof(Triangle) * numTris;
	size_t bvhBytes = sizeof(LinearBVHNode) * numBvhNodes;
	size_t lightsBytes = sizeof(Triangle) * numLights;
	size_t curandBytes = sizeof(curandState) * threadsPerBlock * gridBlocks;

	d_imgVectorPtr = NULL;
	d_imgBytesPtr = NULL;
	d_camPtr = NULL;
	d_triPtr = NULL;
	d_bvhPtr = NULL;
	d_lightsPtr = NULL;
	d_trianglesData = NULL;
	d_lightsData = NULL;
	d_curandStatePtr = NULL;

	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_imgVectorPtr, sizeof(Vector3Df) * pixels));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_imgBytesPtr, sizeof(uchar4) * pixels));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_camPtr, sizeof(Camera)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_triPtr, trianglesBytes));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_bvhPtr, bvhBytes));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_lightsPtr, lightsBytes));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_trianglesData, sizeof(TrianglesData) + trianglesBytes));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_lightsData, sizeof(LightsData) + lightsBytes));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_curandStatePtr, curandBytes));

	createSettingsData(&d_settingsData);
	copyMemoryToCuda();

	initializeCurand();
}

__host__ ParallelRenderer::~ParallelRenderer() {
	cudaFree(d_imgVectorPtr);
	cudaFree(d_camPtr);
	cudaFree(d_triPtr);
	cudaFree(d_bvhPtr);
	cudaFree(d_lightsPtr);
	cudaFree(d_trianglesData);
	cudaFree(d_lightsData);
	cudaFree(d_curandStatePtr);
}

__host__ void ParallelRenderer::copyMemoryToCuda() {
	Scene* scenePtr = getScenePtr();
	int numTris = scenePtr->getNumTriangles();
	int numLights = scenePtr->getNumLights();
	int numBvhNodes = scenePtr->getNumBvhNodes();
	float lightsSurfaceArea = scenePtr->getLightsSurfaceArea();
	size_t trianglesBytes = sizeof(Triangle) * numTris;
	size_t bvhBytes = sizeof(LinearBVHNode) * numBvhNodes;
	size_t lightsBytes = sizeof(Triangle) * numLights;
	size_t trianglesDataBytes = sizeof(TrianglesData) + trianglesBytes;

	Camera* h_camPtr = scenePtr->getCameraPtr();
	Triangle* h_triPtr = scenePtr->getTriPtr();
	LinearBVHNode* h_bvhPtr = scenePtr->getBvhPtr();
	Triangle* h_lightsPtr = scenePtr->getLightsPtr();
	TrianglesData* h_trianglesData = (TrianglesData*)malloc(sizeof(TrianglesData) + trianglesBytes + bvhBytes);
	LightsData* h_lightsData = (LightsData*)malloc(sizeof(LightsData) + lightsBytes);

	CUDA_CHECK_RETURN(cudaMemcpy(d_camPtr, h_camPtr, sizeof(Camera), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_triPtr, h_triPtr, trianglesBytes, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_bvhPtr, h_bvhPtr, bvhBytes, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_lightsPtr, h_lightsPtr, sizeof(Triangle) * numLights, cudaMemcpyHostToDevice));

	createTrianglesData(h_trianglesData, d_triPtr, d_bvhPtr);
	CUDA_CHECK_RETURN(cudaMemcpy(d_trianglesData, h_trianglesData, trianglesDataBytes, cudaMemcpyHostToDevice));

	createLightsData(h_lightsData, d_lightsPtr);
	CUDA_CHECK_RETURN(cudaMemcpy(d_lightsData, h_lightsData, sizeof(LightsData) + lightsBytes, cudaMemcpyHostToDevice));

	free(h_trianglesData);
	free(h_lightsData);
}

__host__ void ParallelRenderer::initializeCurand() {
	dim3 block = dim3(BLOCK_WIDTH, BLOCK_WIDTH, 1);
	dim3 grid = dim3(width/BLOCK_WIDTH, height/BLOCK_WIDTH, 1);

	initializeCurandKernel<<<grid, block, 0>>>(d_curandStatePtr);
}

__host__ void ParallelRenderer::renderOneSamplePerPixel(uchar4* p_img) {
	dim3 block = dim3(BLOCK_WIDTH, BLOCK_WIDTH, 1);
	dim3 grid = dim3(width/BLOCK_WIDTH, height/BLOCK_WIDTH, 1);
	samplesRendered++;
	renderKernel<<<grid, block, 0>>>(d_settingsData,
			d_imgVectorPtr,
			p_img,
			d_camPtr,
			d_trianglesData,
			d_lightsData,
			d_curandStatePtr,
			samplesRendered);
}

__host__ void ParallelRenderer::copyImageBytes() {
	int pixels = width * height;
	size_t imgBytes = sizeof(uchar4) * pixels;
	CUDA_CHECK_RETURN(cudaMemcpy(h_imgPtr, d_imgBytesPtr, imgBytes, cudaMemcpyDeviceToHost));
	for (unsigned i = 0; i < pixels; i++) {
		gammaCorrectPixel(h_imgPtr[i]);
	}
}

__global__ void initializeCurandKernel(curandState* p_curandState) {
	int idx = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y)
				+ (threadIdx.y * blockDim.x) + threadIdx.x;
	curand_init(1234, idx, 0, &p_curandState[idx]);
}

__global__ void renderKernel(SettingsData settings,
		Vector3Df* p_imgBuffer,
		uchar4* p_outImg,
		Camera* p_camera,
		TrianglesData* p_tris,
		LightsData* p_lights,
		curandState *p_curandState,
		int sampleNumber) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = y * settings.width + x;
	curandState* p_threadCurand = &p_curandState[idx];
	Sampler sampler(p_threadCurand);
	Vector3Df color = samplePixel(x, y, p_camera, p_tris, p_lights, &sampler);
	p_imgBuffer[idx] += color;
	p_outImg[idx] = vector3ToUchar4(p_imgBuffer[idx]/(float)sampleNumber);
}
