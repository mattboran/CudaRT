/*
 * renderer.h
 *
 *  Created on: Dec 22, 2018
 *      Author: matt
 */

#ifndef RENDERER_H_
#define RENDERER_H_

 #include "bvh.h"
#include "scene.h"

#include <cuda.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

#define BVH_BOUNDS_OFFSET 0
#define BVH_INDEX_OFFSET 1
#define TEXTURES_OFFSET 2

#define MAX_MATERIALS 128
#define KD_OFFSET 0
#define KA_OFFSET 1
#define KS_OFFSET 2
#define AUX_OFFSET 3
#define MATERIALS_FLOAT_COMPONENTS (AUX_OFFSET + 1)

typedef unsigned int pixels_t;

struct SceneData {
	Triangle* p_triangles;
	cudaTextureObject_t* p_cudaTexObjects;
#ifndef __CUDA_ARCH__
	LinearBVHNode* p_bvh;
	float3* p_textureData;
	pixels_t* p_textureDimensions;
	pixels_t* p_textureOffsets;
	uint numBVHNodes;
	uint numTextures;
#endif
}__attribute((aligned(8)));

struct TextureContainer {
#ifdef __CUDA_ARCH__
	__host__ __device__ TextureContainer(cudaTextureObject_t* p_texObject) :
			p_textureObject(p_texObject) {}
	cudaTextureObject_t* p_textureObject = NULL;

#else
	__host__ __device__ TextureContainer(float3* p_texData, pixels_t* p_texDims) :
				p_textureData(p_texData), p_textureDimensions(p_texDims) {}

	float3* p_textureData = NULL;
	pixels_t* p_textureDimensions = NULL;
#endif
};

struct Sampler {
	curandState* p_curandState = NULL;
	__host__ Sampler() {}
	__device__ Sampler(curandState* p_curand) : p_curandState(p_curand) {}
	__host__ __device__ float getNextFloat();
};

__host__ __device__ float3 samplePixel(int x, int y,
                                          Camera p_camera,
                                          SceneData* p_SceneData,
                                          uint* p_lightsIndices,
                  	  				      uint numLights,
                  					      float lightsSurfaceArea,
                                          Sampler* p_sampler,
                                          float3* p_matFloats,
                                          int2* p_matIndices);
__host__ __device__ void gammaCorrectPixel(uchar4 &p);
__host__ __device__ float3 sampleTexture(TextureContainer* p_textureContainer, float u, float v);

class Renderer {
public:
	bool useCuda = false;
	uchar4* h_imgPtr;
	__host__ virtual ~Renderer() { delete[] h_imgPtr; }
	__host__ virtual void renderOneSamplePerPixel(uchar4* p_img) = 0;
	__host__ virtual void copyImageBytes(uchar4* p_img) = 0;
	__host__ virtual uchar4* getImgBytesPointer() = 0;
    __host__ virtual void createMaterialsData() = 0;
	__host__ cudaTextureObject_t* getCudaTextureObjectPtr() { return NULL; }
	__host__ pixels_t getWidth() { return width; }
	__host__ pixels_t getHeight() { return height; }
	__host__ int getSamples() { return samples; }
	__host__ int getSamplesRendered() { return samplesRendered; }
protected:
	__host__ Renderer() {}
	__host__ Renderer(Scene* _scenePtr, pixels_t _width, pixels_t _height, uint _samples);
	Scene* p_scene;
	pixels_t width;
	pixels_t height;
	uint samples;
	volatile uint samplesRendered;
};

class ParallelRenderer : public Renderer {
public:
	__host__ ParallelRenderer() : Renderer() {}
	__host__ ParallelRenderer(Scene* _scenePtr, pixels_t _width, pixels_t _height, uint _samples);
	__host__ void renderOneSamplePerPixel(uchar4* p_img);
	__host__ void copyImageBytes(uchar4* p_img);
	__host__ uchar4* getImgBytesPointer() { return d_imgBytesPtr; }
    __host__ void createMaterialsData();
	__host__ ~ParallelRenderer();
private:
	float3* d_imgVectorPtr;
	uchar4* d_imgBytesPtr;
	SceneData* d_sceneData;
	Triangle* d_triPtr;
    uint* d_lightsIndices;
	cudaTextureObject_t* d_cudaTexObjects;
	Camera* d_camPtr;
	curandState* d_curandStatePtr;
	// TODO: Consider storing block, grid instead
	uint threadsPerBlock;
	uint gridBlocks;
	__host__ void copyMemoryToCuda();
	__host__ void initializeCurand();
	__host__ cudaTextureObject_t* createTextureObjects();
};

class SequentialRenderer : public Renderer {
public:
	SequentialRenderer() : Renderer() {}
	SequentialRenderer(Scene* _scenePtr, pixels_t _width, pixels_t _height, uint _samples);
	__host__ void renderOneSamplePerPixel(uchar4* p_img);
	__host__ void copyImageBytes(uchar4* p_img);
	__host__ uchar4* getImgBytesPointer() { return h_imgBytesPtr; }
    __host__ void createSceneData(SceneData* p_SceneData,
                                  Triangle* p_triangles,
                                  LinearBVHNode* p_bvh,
                                  float3* p_textureData,
                                  pixels_t* p_textureDimensions,
                                  pixels_t* p_textureOffsets);
    __host__ void createMaterialsData();
	__host__ ~SequentialRenderer();
private:
	uchar4* h_imgBytesPtr;
	float3* h_imgVectorPtr;
	SceneData* h_sceneData;
};

__host__ __device__ inline uchar4 float3ToUchar4(const float3& v) {
    uchar4 retVal;
	retVal.x = (unsigned char)((v.x > 1.0f ? 1.0f: v.x)*(255.f));
	retVal.y = (unsigned char)((v.y > 1.0f ? 1.0f: v.y)*(255.f));
	retVal.z = (unsigned char)((v.z > 1.0f ? 1.0f: v.z)*(255.f));
	retVal.w = 255u;
	return retVal;
}

#endif /* RENDERER_H_ */
