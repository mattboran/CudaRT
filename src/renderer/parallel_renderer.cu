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
	pixels_t totalTexturePixels = p_scene->getTotalTexturePixels();
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
	d_textureData = NULL;
	d_textureDimensions = NULL;
	d_textureOffsets = NULL;
	d_cudaTexObjects = NULL;
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

	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_textureData, sizeof(Vector3Df) * totalTexturePixels));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_textureDimensions, sizeof(pixels_t) * numTextures * 2));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_textureOffsets, sizeof(pixels_t) * numTextures + sizeof(pixels_t)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_cudaTexObjects, sizeof(cudaTextureObject_t) * numTextures));

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
	cudaFree(d_textureData);
	cudaFree(d_textureDimensions);
	cudaFree(d_textureOffsets);
	cudaFree(d_cudaTexObjects);
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
	pixels_t numTotalTexturePixels = p_scene->getTotalTexturePixels();

	float lightsSurfaceArea = p_scene->getLightsSurfaceArea();
	size_t trianglesBytes = sizeof(Triangle) * numTris;
	size_t materialsBytes = sizeof(Material) * numMaterials;
	size_t bvhBytes = sizeof(LinearBVHNode) * numBvhNodes;
	size_t lightsBytes = sizeof(Triangle) * numLights;
	size_t textureBytes = sizeof(Vector3Df) * numTotalTexturePixels;
	size_t textureObjectBytes = sizeof(cudaTextureObject_t) * numTextures;

	Camera* h_camPtr = p_scene->getCameraPtr();
	Triangle* h_triPtr = p_scene->getTriPtr();
	LinearBVHNode* h_bvhPtr = p_scene->getBvhPtr();
	Triangle* h_lightsPtr = p_scene->getLightsPtr();
	Material* h_materialsPtr = p_scene->getMaterialsPtr();
	SceneData* h_sceneData = (SceneData*)malloc(sizeof(SceneData));
	LightsData* h_lightsData = (LightsData*)malloc(sizeof(LightsData));
	Vector3Df* h_textureData = p_scene->getTexturePtr();
	pixels_t* h_textureDimensions = p_scene->getTextureDimensionsPtr();
	pixels_t* h_textureOffsets = p_scene->getTextureOffsetsPtr();

	CUDA_CHECK_RETURN(cudaMemcpy(d_camPtr, h_camPtr, sizeof(Camera), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_triPtr, h_triPtr, trianglesBytes, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_materials, h_materialsPtr, materialsBytes, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_bvhPtr, h_bvhPtr, bvhBytes, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_lightsPtr, h_lightsPtr, lightsBytes, cudaMemcpyHostToDevice));

	CUDA_CHECK_RETURN(cudaMemcpy(d_textureData, h_textureData, textureBytes, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_textureDimensions, h_textureDimensions, sizeof(pixels_t) * 2 * numTextures, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_textureOffsets, h_textureOffsets, sizeof(pixels_t) * numTextures + sizeof(pixels_t), cudaMemcpyHostToDevice));

	// Create texture objects for textures
	cudaTextureObject_t* p_cudaTexObjects = new cudaTextureObject_t[numTextures];
	cudaResourceDesc* p_resDesc = new cudaResourceDesc[numTextures];
	cudaTextureDesc* p_texDesc = new cudaTextureDesc[numTextures];
	for (uint i = 0; i < numTextures; i++) {
		Vector3Df* p_currentTextureData = p_scene->getTexturePtr(i);
		pixels_t numPixels = h_textureOffsets[i+1] - h_textureOffsets[i];
		float4* p_currentTextureFormattedData = new float4[numPixels];
		for (pixels_t j = 0; j < numPixels; j++) {
			p_currentTextureFormattedData[j] = make_float4(p_currentTextureData[j]);
		}
		float4* d_currentTextureData = NULL;
		CUDA_CHECK_RETURN(cudaMalloc((void**)&d_currentTextureData, sizeof(float4) * numPixels));
		CUDA_CHECK_RETURN(cudaMemcpy(d_currentTextureData, p_currentTextureFormattedData, sizeof(float4) * numPixels, cudaMemcpyHostToDevice));

		memset(&p_resDesc[i], 0, sizeof(cudaResourceDesc));
		p_resDesc[i].resType = cudaResourceTypeLinear;
		p_resDesc[i].res.linear.devPtr = d_currentTextureData;
		p_resDesc[i].res.linear.desc.f = cudaChannelFormatKindFloat;
		p_resDesc[i].res.linear.desc.x = 32;
		p_resDesc[i].res.linear.desc.z = 32;
		p_resDesc[i].res.linear.desc.y = 32;
		p_resDesc[i].res.linear.desc.w = 32; //todo: should this be 0??
		p_resDesc[i].res.linear.sizeInBytes = numPixels * sizeof(float4);

		memset(&p_texDesc[i], 0, sizeof(cudaTextureDesc));
		p_texDesc[i].readMode = cudaReadModeElementType;
		p_texDesc[i].addressMode[0] = cudaAddressModeWrap;
		p_texDesc[i].filterMode = cudaFilterModeLinear;

		cudaTextureObject_t* p_currentTexObject = new cudaTextureObject_t();
		cudaCreateTextureObject(p_currentTexObject,
								&p_resDesc[i],
								&p_texDesc[i],
								NULL);
		p_cudaTexObjects[i] = *p_currentTexObject;
	}
	CUDA_CHECK_RETURN(cudaMemcpy(d_cudaTexObjects, p_cudaTexObjects, sizeof(cudaTextureObject_t) * numTextures, cudaMemcpyHostToDevice));
	h_sceneData->p_cudaTexObjects = d_cudaTexObjects;

	createSceneData(h_sceneData, d_triPtr, d_bvhPtr, d_materials, d_textureData, d_textureDimensions, d_textureOffsets);
	CUDA_CHECK_RETURN(cudaMemcpy(d_sceneData, h_sceneData, sizeof(SceneData), cudaMemcpyHostToDevice));

	createLightsData(h_lightsData, d_lightsPtr);
	CUDA_CHECK_RETURN(cudaMemcpy(d_lightsData, h_lightsData, sizeof(LightsData), cudaMemcpyHostToDevice));

	free(h_sceneData);
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
