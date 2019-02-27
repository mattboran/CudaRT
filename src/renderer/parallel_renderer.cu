/*
 * parallel_renderer.cpp
 *
 *  Created on: Dec 22, 2018
 *      Author: matt
 */

#include "renderer.h"
#include "cuda_error_check.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <iostream>

using std::cout;

#define USE_SHARED_MEMORY
#define BLOCK_WIDTH 16u

// Kernels
__global__ void initializeCurandKernel(curandState* p_curandState);
__global__ void renderKernel(SettingsData settings,
		Vector3Df* p_imgBuffer,
		uchar4* p_outImg,
		Camera* p_camera,
		SceneData* p_tris,
		LightsData* p_lights,
		curandState *p_curandState,
		int sampleNumber);

__host__ ParallelRenderer::ParallelRenderer(Scene* _scenePtr, pixels_t _width, pixels_t _height, uint _samples) :
	Renderer(_scenePtr, _width, _height, _samples) {
	// CUDA settings
	useCuda = true;
	threadsPerBlock = BLOCK_WIDTH * BLOCK_WIDTH;
	gridBlocks = width / BLOCK_WIDTH * height / BLOCK_WIDTH;

	pixels_t pixels = width * height;
	uint numTris = p_scene->getNumTriangles();
	uint numMaterials = p_scene->getNumMaterials();
	uint numBvhNodes = p_scene->getNumBvhNodes();
	uint numLights = p_scene->getNumLights();
	uint numTextures = p_scene->getNumTextures();
	size_t trianglesBytes = sizeof(Triangle) * numTris;
	size_t materialsBytes = sizeof(Material) * numMaterials;
	size_t bvhBytes = sizeof(LinearBVHNode) * numBvhNodes;
	size_t lightsBytes = sizeof(Triangle) * numLights;
	size_t curandBytes = sizeof(curandState) * threadsPerBlock * gridBlocks;

	d_imgVectorPtr = NULL;
	d_imgBytesPtr = NULL;
	d_camPtr = NULL;
	d_triPtr = NULL;
	d_bvhPtr = NULL;
	d_materials = NULL;
	dd_textureData = NULL;
	d_textureDimensions = NULL;
	d_lightsPtr = NULL;
	d_sceneData = NULL;
	d_lightsData = NULL;
	d_curandStatePtr = NULL;

	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_imgVectorPtr, sizeof(Vector3Df) * pixels));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_imgBytesPtr, sizeof(uchar4) * pixels));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_camPtr, sizeof(Camera)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_triPtr, trianglesBytes));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_materials, materialsBytes));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_bvhPtr, bvhBytes));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_lightsPtr, lightsBytes));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_sceneData, sizeof(SceneData)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_lightsData, sizeof(LightsData)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_curandStatePtr, curandBytes));

	CUDA_CHECK_RETURN(cudaMalloc((void**)&dd_textureData, sizeof(Vector3Df*) * numTextures));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_textureDimensions, sizeof(pixels_t) * numTextures * 2));
	createSettingsData(&d_settingsData);
	copyMemoryToCuda();

	initializeCurand();
}

__host__ ParallelRenderer::~ParallelRenderer() {
	cudaFree(d_imgVectorPtr);
	cudaFree(d_camPtr);
	cudaFree(d_triPtr);
	cudaFree(d_bvhPtr);
	cudaFree(d_materials);
	cudaFree(dd_textureData);
	cudaFree(d_textureDimensions);
	cudaFree(d_lightsPtr);
	cudaFree(d_sceneData);
	cudaFree(d_lightsData);
	cudaFree(d_curandStatePtr);
}

__host__ void ParallelRenderer::copyMemoryToCuda() {
	uint numTris = p_scene->getNumTriangles();
	uint numLights = p_scene->getNumLights();
	uint numBvhNodes = p_scene->getNumBvhNodes();
	uint numMaterials = p_scene->getNumMaterials();
	uint numTextures = p_scene->getNumTextures();

	float lightsSurfaceArea = p_scene->getLightsSurfaceArea();
	size_t trianglesBytes = sizeof(Triangle) * numTris;
	size_t materialsBytes = sizeof(Material) * numMaterials;
	size_t bvhBytes = sizeof(LinearBVHNode) * numBvhNodes;
	size_t lightsBytes = sizeof(Triangle) * numLights;

	Camera* h_camPtr = p_scene->getCameraPtr();
	Triangle* h_triPtr = p_scene->getTriPtr();
	LinearBVHNode* h_bvhPtr = p_scene->getBvhPtr();
	Triangle* h_lightsPtr = p_scene->getLightsPtr();
	Material* h_materialsPtr = p_scene->getMaterialsPtr();
	SceneData* h_SceneData = (SceneData*)malloc(sizeof(SceneData));
	LightsData* h_lightsData = (LightsData*)malloc(sizeof(LightsData));

	CUDA_CHECK_RETURN(cudaMemcpy(d_camPtr, h_camPtr, sizeof(Camera), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_triPtr, h_triPtr, trianglesBytes, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_materials, h_materialsPtr, materialsBytes, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_bvhPtr, h_bvhPtr, bvhBytes, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_lightsPtr, h_lightsPtr, lightsBytes, cudaMemcpyHostToDevice));

	for (uint i = 0; i < numTextures; i++) {
		pixels_t* p_texDimensions = p_scene->getTextureDimensionsPtr();
		pixels_t width = p_texDimensions[i * 2];
		pixels_t height = p_texDimensions[i * 2 + 1];
		size_t pixelsBytes = width * height * sizeof(Vector3Df);
		Vector3Df* d_currentTexture;
		Vector3Df* h_currentTexture = p_scene->getTexturePtr(i);
		CUDA_CHECK_RETURN(cudaMalloc((void**)&d_currentTexture, pixelsBytes));
		CUDA_CHECK_RETURN(cudaMemcpy(&dd_textureData[i], d_currentTexture, sizeof(Vector3Df*), cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(dd_textureData[i], h_currentTexture, pixelsBytes, cudaMemcpyHostToDevice));
	}
	createSceneData(h_SceneData, d_triPtr, d_bvhPtr, d_materials, dd_textureData, d_textureDimensions);
	CUDA_CHECK_RETURN(cudaMemcpy(d_sceneData, h_SceneData, sizeof(SceneData), cudaMemcpyHostToDevice));

	createLightsData(h_lightsData, d_lightsPtr);
	CUDA_CHECK_RETURN(cudaMemcpy(d_lightsData, h_lightsData, sizeof(LightsData), cudaMemcpyHostToDevice));

	free(h_SceneData);
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
	size_t sharedMemory = sizeof(Material) * p_scene->getNumMaterials();
	renderKernel<<<grid, block, sharedMemory>>>(d_settingsData,
			d_imgVectorPtr,
			p_img,
			d_camPtr,
			d_sceneData,
			d_lightsData,
			d_curandStatePtr,
			samplesRendered);
}

__host__ void ParallelRenderer::copyImageBytes(uchar4* p_img) {
	pixels_t pixels = width * height;
	size_t imgBytes = sizeof(uchar4) * pixels;
	CUDA_CHECK_RETURN(cudaMemcpy(h_imgPtr, p_img, imgBytes, cudaMemcpyDeviceToHost));
	for (uint i = 0; i < pixels; i++) {
		gammaCorrectPixel(h_imgPtr[i]);
	}
}

__global__ void initializeCurandKernel(curandState* p_curandState) {
	uint idx = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y)
				+ (threadIdx.y * blockDim.x) + threadIdx.x;
	curand_init(1234, idx, 0, &p_curandState[idx]);
}

__global__ void renderKernel(SettingsData settings,
		Vector3Df* p_imgBuffer,
		uchar4* p_outImg,
		Camera* p_camera,
		SceneData* p_tris,
		LightsData* p_lights,
		curandState *p_curandState,
		int sampleNumber) {

#ifdef USE_SHARED_MEMORY
	unsigned int numMaterials = p_tris->numMaterials;
	extern __shared__ Material d_materials[];
	if (threadIdx.x + threadIdx.y == 0) {
		for (int i = 0; i < numMaterials; i++) {
			d_materials[i] = p_tris->p_materials[i];
		}
	}
	__syncthreads();
#else
	Material* d_materials = p_tris->p_materials;
#endif
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint idx = y * settings.width + x;
	curandState* p_threadCurand = &p_curandState[idx];
	Sampler sampler(p_threadCurand);
	Vector3Df color = samplePixel(x, y, p_camera, p_tris, p_lights, d_materials, &sampler);
	p_imgBuffer[idx] += color;
	p_outImg[idx] = vector3ToUchar4(p_imgBuffer[idx]/(float)sampleNumber);
}
