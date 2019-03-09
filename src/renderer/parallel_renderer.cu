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

#define BLOCK_WIDTH 16u

__constant__ float3 c_materialFloats[MAX_MATERIALS * MATERIALS_FLOAT_COMPONENTS];
__constant__ int2 c_materialIndices[MAX_MATERIALS];

// Kernels
__global__ void initializeCurandKernel(curandState* p_curandState);
__global__ void renderKernel(SettingsData settings,
		Vector3Df* p_imgBuffer,
		uchar4* p_outImg,
		Camera* p_camera,
		SceneData* p_sceneData,
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
	size_t lightsBytes = sizeof(Triangle) * numLights;
	size_t curandBytes = sizeof(curandState) * threadsPerBlock;
	size_t textureObjectBytes = sizeof(cudaTextureObject_t) * (numTextures + TEXTURES_OFFSET);

	d_imgVectorPtr = NULL;
	d_imgBytesPtr = NULL;
	d_camPtr = NULL;
	d_triPtr = NULL;
	d_cudaTexObjects = NULL;
	d_lightsPtr = NULL;
	d_sceneData = NULL;
	d_lightsData = NULL;
	d_curandStatePtr = NULL;

	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_imgVectorPtr, sizeof(Vector3Df) * pixels));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_imgBytesPtr, sizeof(uchar4) * pixels));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_camPtr, sizeof(Camera)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_triPtr, trianglesBytes));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_lightsPtr, lightsBytes));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_sceneData, sizeof(SceneData)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_lightsData, sizeof(LightsData)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_curandStatePtr, curandBytes));

	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_cudaTexObjects, textureObjectBytes));

	createSettingsData(&d_settingsData);
	copyMemoryToCuda();

	initializeCurand();
}

__host__ ParallelRenderer::~ParallelRenderer() {
	cudaFree(d_imgVectorPtr);
	cudaFree(d_camPtr);
	cudaFree(d_triPtr);
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
	size_t lightsBytes = sizeof(Triangle) * numLights;
	size_t textureObjectBytes = sizeof(cudaTextureObject_t) * (numTextures + TEXTURES_OFFSET);

	Camera* h_camPtr = p_scene->getCameraPtr();
	Triangle* h_triPtr = p_scene->getTriPtr();
	Triangle* h_lightsPtr = p_scene->getLightsPtr();
	Material* h_materialsPtr = p_scene->getMaterialsPtr();
	SceneData* h_sceneData = (SceneData*)malloc(sizeof(SceneData));
	LightsData* h_lightsData = (LightsData*)malloc(sizeof(LightsData));
	Vector3Df* h_textureData = p_scene->getTexturePtr();

	CUDA_CHECK_RETURN(cudaMemcpy(d_camPtr, h_camPtr, sizeof(Camera), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_triPtr, h_triPtr, trianglesBytes, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_lightsPtr, h_lightsPtr, lightsBytes, cudaMemcpyHostToDevice));

	cudaTextureObject_t* h_textureObjects = createTextureObjects();
	CUDA_CHECK_RETURN(cudaMemcpy(d_cudaTexObjects, h_textureObjects, textureObjectBytes, cudaMemcpyHostToDevice));
	h_sceneData->p_cudaTexObjects = d_cudaTexObjects;

	createSceneData(h_sceneData, d_triPtr, NULL, NULL, NULL, NULL);
	CUDA_CHECK_RETURN(cudaMemcpy(d_sceneData, h_sceneData, sizeof(SceneData), cudaMemcpyHostToDevice));

	createLightsData(h_lightsData, d_lightsPtr);
	CUDA_CHECK_RETURN(cudaMemcpy(d_lightsData, h_lightsData, sizeof(LightsData), cudaMemcpyHostToDevice));

	createMaterialsData(NULL, NULL);
	free(h_sceneData);
	free(h_lightsData);
}

__host__ cudaTextureObject_t* ParallelRenderer::createTextureObjects() {
	uint numTextures = p_scene->getNumTextures();
	cudaTextureObject_t* p_cudaTexObjects = new cudaTextureObject_t[numTextures + TEXTURES_OFFSET];
	//
	// BVH
	//
	LinearBVHNode* h_bvh = p_scene->getBvhPtr();
	size_t numBvhNodes = p_scene->getNumBvhNodes();
	// Copy min and max
	{
		size_t size = numBvhNodes * 2 * sizeof(float4);
		float4* h_buffer = new float4[numBvhNodes * 2];
		for (uint i = 0; i < numBvhNodes; i++) {
			h_buffer[2*i] = make_float4(h_bvh[i].min);
			h_buffer[2*i + 1] = make_float4(h_bvh[i].max);
		}
		float4* d_buffer = NULL;
		CUDA_CHECK_RETURN(cudaMalloc((void**)&d_buffer, size));
		CUDA_CHECK_RETURN(cudaMemcpy(d_buffer, h_buffer, size, cudaMemcpyHostToDevice));

		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(cudaResourceDesc));
		resDesc.resType = cudaResourceTypeLinear;
		resDesc.res.linear.devPtr = d_buffer;
		resDesc.res.linear.desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
		resDesc.res.linear.sizeInBytes = size;

		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(cudaTextureDesc));
		texDesc.addressMode[0] = cudaAddressModeClamp;
		texDesc.filterMode = cudaFilterModePoint;

		cudaTextureObject_t currentTexObject = 0;
		cudaCreateTextureObject(&currentTexObject,
								&resDesc,
								&texDesc,
								NULL);
		p_cudaTexObjects[BVH_BOUNDS_OFFSET] = currentTexObject;
		delete h_buffer;
	}
	// Copy indexes, numTriangles, and axis
	{
		size_t size = numBvhNodes * sizeof(int2);
		int2* h_buffer = new int2[numBvhNodes];
		for (uint i = 0; i < numBvhNodes; i++) {
			h_buffer[i].x = h_bvh->secondChildOffset;
			//
			int32_t yValue = ((int32_t)(h_bvh->numTriangles) << 16) | ((int32_t)(h_bvh->axis));
			h_buffer[i].y = yValue;
			h_bvh++;
		}
		int2* d_buffer = NULL;
		CUDA_CHECK_RETURN(cudaMalloc((void**)&d_buffer, size));
		CUDA_CHECK_RETURN(cudaMemcpy(d_buffer, h_buffer, size, cudaMemcpyHostToDevice));

		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(cudaResourceDesc));
		resDesc.resType = cudaResourceTypeLinear;
		resDesc.res.linear.devPtr = d_buffer;
		resDesc.res.linear.desc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindSigned);
		resDesc.res.linear.sizeInBytes = size;

		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(cudaTextureDesc));
		texDesc.addressMode[0] = cudaAddressModeClamp;
		texDesc.filterMode = cudaFilterModePoint;

		cudaTextureObject_t currentTexObject = 0;
		cudaCreateTextureObject(&currentTexObject,
								&resDesc,
								&texDesc,
								NULL);
		p_cudaTexObjects[BVH_INDEX_OFFSET] = currentTexObject;
		delete h_buffer;
	}

	//
	// Actual Textures
	//
	pixels_t* h_textureDimensions = p_scene->getTextureDimensionsPtr();
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
	for (uint i = 0; i < numTextures; i++) {
		Vector3Df* p_currentTextureData = p_scene->getTexturePtr(i);
		pixels_t width = h_textureDimensions[2*i];
		pixels_t height = h_textureDimensions[2*i + 1];
		pixels_t numPixels = width * height;
		size_t size = numPixels * sizeof(float4);
		float4* p_currentTextureFormattedData = new float4[numPixels];
		for (pixels_t j = 0; j < numPixels; j++) {
			p_currentTextureFormattedData[j] = make_float4(p_currentTextureData[j]);
		}
		cudaArray* cuArray = NULL;
		CUDA_CHECK_RETURN(cudaMallocArray(&cuArray, &channelDesc, width, height));
		CUDA_CHECK_RETURN(cudaMemcpyToArray(cuArray,
											0,
											0,
											p_currentTextureFormattedData,
											size,
											cudaMemcpyHostToDevice));

		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(cudaResourceDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = cuArray;

		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(cudaTextureDesc));
		texDesc.addressMode[0] = cudaAddressModeWrap;
		texDesc.addressMode[1] = cudaAddressModeWrap;
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.readMode = cudaReadModeElementType;
		texDesc.normalizedCoords = 1;

		cudaTextureObject_t currentTexObject = 0;
		cudaCreateTextureObject(&currentTexObject,
								&resDesc,
								&texDesc,
								NULL);
		p_cudaTexObjects[i + TEXTURES_OFFSET] = currentTexObject;
		delete p_currentTextureFormattedData;
	}
	return p_cudaTexObjects;
}

__host__ void ParallelRenderer::createMaterialsData(float3* matFloats, int2* matIndices) {
	Material* p_materials = p_scene->getMaterialsPtr();
	uint numMaterials = p_scene->getNumMaterials();
	float3* p_floatBuffer = new float3[MAX_MATERIALS * MATERIALS_FLOAT_COMPONENTS];
	int2* p_intBuffer = new int2[MAX_MATERIALS];
	float3* p_currentFloat = p_floatBuffer;
	int2* p_currentIndex = p_intBuffer;
	for (uint i = 0; i < numMaterials; i++) {
		*p_currentFloat++ = make_float3(p_materials[i].kd);
		*p_currentFloat++ = make_float3(p_materials[i].ka);
		*p_currentFloat++ = make_float3(p_materials[i].ks);
		*p_currentFloat++ = make_float3(p_materials[i].ns,
										p_materials[i].ni,
										p_materials[i].diffuseCoefficient);
		*p_currentIndex++ = make_int2((int32_t)p_materials[i].bsdf,
									  (int32_t)p_materials[i].texKdIdx);

	}
	cudaMemcpyToSymbol(c_materialFloats,
					   p_floatBuffer,
					   numMaterials * MATERIALS_FLOAT_COMPONENTS * sizeof(float3));
	cudaMemcpyToSymbol(c_materialIndices,
					   p_intBuffer,
					   numMaterials * sizeof(int2));

	delete p_floatBuffer;
	delete p_intBuffer;
}

__host__ void ParallelRenderer::initializeCurand() {
	dim3 block = dim3(BLOCK_WIDTH, BLOCK_WIDTH, 1);
	dim3 grid = dim3(width/BLOCK_WIDTH, height/BLOCK_WIDTH, 1);

	initializeCurandKernel<<<1, block, 0>>>(d_curandStatePtr);
}

__host__ void ParallelRenderer::renderOneSamplePerPixel(uchar4* p_img) {
	dim3 block = dim3(BLOCK_WIDTH, BLOCK_WIDTH, 1);
	dim3 grid = dim3(width/BLOCK_WIDTH, height/BLOCK_WIDTH, 1);
	samplesRendered++;
	renderKernel<<<grid, block, 0>>>(d_settingsData,
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
		SceneData* p_sceneData,
		LightsData* p_lights,
		curandState *p_curandState,
		int sampleNumber) {

	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint blockOnlyIdx = threadIdx.x * blockDim.x + threadIdx.y;
	uint idx = y * settings.width + x;
	curandState* p_threadCurand = &p_curandState[blockOnlyIdx];
	Sampler sampler(p_threadCurand);
	Vector3Df color = samplePixel(x, y,
								  p_camera,
								  p_sceneData,
								  p_lights,
								  &sampler,
								  c_materialFloats,
								  c_materialIndices);
	p_imgBuffer[idx] += color;
	p_outImg[idx] = vector3ToUchar4(p_imgBuffer[idx]/(float)sampleNumber);
}
