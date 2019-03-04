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

typedef unsigned int pixels_t;

struct LightsData {
	Triangle* lightsPtr;
	uint numLights;
	float totalSurfaceArea;
};

struct SceneData {
	Triangle* p_triangles;
	LinearBVHNode* p_bvh;
	Material* p_materials;
	Vector3Df* p_textureData;
	pixels_t* p_textureDimensions;
	pixels_t* p_textureOffsets;
	uint numTriangles;
	uint numBVHNodes;
	uint numMaterials;
	uint numTextures;
};

struct SettingsData {
	uint width;
	uint height;
	uint samples;
};

struct TextureContainer {
	__host__ __device__ TextureContainer(Vector3Df* p_texData, pixels_t* p_texDims) :
		p_textureData(p_texData), p_textureDimensions(p_texDims) {}
	Vector3Df* p_textureData = NULL;
	pixels_t* p_textureDimensions = NULL;
};

struct Sampler {
	curandState* p_curandState = NULL;
	__host__ Sampler() {}
	__device__ Sampler(curandState* p_curand) : p_curandState(p_curand) {}
	__host__ __device__ float getNextFloat();
};

__host__ __device__ Vector3Df samplePixel(int x, int y, Camera* p_camera, SceneData* p_SceneData, LightsData *p_lightsData, Material* p_materials, Sampler* p_sampler);
__host__ __device__ void gammaCorrectPixel(uchar4 &p);
__host__ __device__ Vector3Df sampleTexture(TextureContainer* p_textureContainer, float u, float v);

class Renderer {
public:
	bool useCuda = false;
	uchar4* h_imgPtr;
	virtual ~Renderer() { delete[] h_imgPtr; }
	__host__ virtual void renderOneSamplePerPixel(uchar4* p_img) = 0;
	__host__ virtual void copyImageBytes(uchar4* p_img) = 0;
	__host__ virtual uchar4* getImgBytesPointer() = 0;
	__host__ Scene* getScenePtr() { return p_scene; }
	__host__ pixels_t getWidth() { return width; }
	__host__ pixels_t getHeight() { return height; }
	__host__ int getSamples() { return samples; }
	__host__ int getSamplesRendered() { return samplesRendered; }
	__host__ void createSettingsData(SettingsData* p_settingsData);
	__host__ void createSceneData(SceneData* p_SceneData,
                                  Triangle* p_triangles,
                                  LinearBVHNode* p_bvh,
                                  Material* p_materials,
                                  Vector3Df* p_textureData,
                                  pixels_t* p_textureDimensions,
                                  pixels_t* p_textureOffsets);
	__host__ void createLightsData(LightsData* p_lightsData, Triangle* p_triangles);
protected:
	__host__ Renderer() {}
	__host__ Renderer(Scene* _scenePtr, pixels_t _width, pixels_t _height, uint _samples);
	Scene* p_scene;
	pixels_t width;
	pixels_t height;
	uint samples;
	uint samplesRendered;
};

class ParallelRenderer : public Renderer {
public:
	__host__ ParallelRenderer() : Renderer() {}
	__host__ ParallelRenderer(Scene* _scenePtr, pixels_t _width, pixels_t _height, uint _samples);
	__host__ void renderOneSamplePerPixel(uchar4* p_img);
	__host__ void copyImageBytes(uchar4* p_img);
	__host__ uchar4* getImgBytesPointer() { return d_imgBytesPtr; }
	~ParallelRenderer();
private:
	Vector3Df* d_imgVectorPtr;
	uchar4* d_imgBytesPtr;
	LightsData* d_lightsData;
	SceneData* d_sceneData;
	SettingsData d_settingsData;
	Triangle* d_triPtr;
	LinearBVHNode* d_bvhPtr;
	Triangle* d_lightsPtr;
	Material* d_materials;
	Vector3Df* d_textureData;
	pixels_t* d_textureDimensions;
	pixels_t* d_textureOffsets;
	Camera* d_camPtr;
	curandState* d_curandStatePtr;
	// TODO: Consider storing block, grid instead
	uint threadsPerBlock;
	uint gridBlocks;
	__host__ void copyMemoryToCuda();
	__host__ void initializeCurand();
};

class SequentialRenderer : public Renderer {
public:
	SequentialRenderer() : Renderer() {}
	SequentialRenderer(Scene* _scenePtr, pixels_t _width, pixels_t _height, uint _samples);
	__host__ void renderOneSamplePerPixel(uchar4* p_img);
	__host__ void copyImageBytes(uchar4* p_img);
	__host__ uchar4* getImgBytesPointer() { return h_imgBytesPtr; }
	~SequentialRenderer();
private:
	uchar4* h_imgBytesPtr;
	Vector3Df* h_imgVectorPtr;
	SettingsData h_settingsData;
	SceneData* h_sceneData;
	LightsData* h_lightsData;
};

__host__ __device__ inline uchar4 vector3ToUchar4(const Vector3Df& v) {
	uchar4 retVal;
	retVal.x = (unsigned char)((v.x > 1.0f ? 1.0f: v.x)*(255.f));
	retVal.y = (unsigned char)((v.y > 1.0f ? 1.0f: v.y)*(255.f));
	retVal.z = (unsigned char)((v.z > 1.0f ? 1.0f: v.z)*(255.f));
	retVal.w = 255u;
	return retVal;
}

__host__ __device__ inline bool sameTriangle(Triangle* p_a, Triangle* p_b) {
	return p_a->_triId == p_b->_triId;
}

#endif /* RENDERER_H_ */
