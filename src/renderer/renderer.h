/*
 * renderer.h
 *
 *  Created on: Dec 22, 2018
 *      Author: matt
 */

#ifndef RENDERER_H_
#define RENDERER_H_

#include "scene.h"
#include "linalg.h"

#include <curand.h>

struct LightsData {
	geom::Triangle* lightsPtr;
	unsigned numLights;
	float totalSurfaceArea;
};

struct TrianglesData {
	geom::Triangle* triPtr;
	CacheFriendlyBVHNode* bvhPtr;
	unsigned *bvhIndexPtr;
	unsigned numTriangles;
	unsigned numBVHNodes;
};

struct SettingsData {
	int width;
	int height;
	int samples;
	bool useBVH;
};

__host__ __device__ inline uchar4 vector3ToUchar4(const Vector3Df& v) {
	uchar4 retVal;
	float invGamma = 1.f/2.2f;
	retVal.x = (unsigned char)(powf(clamp(v.x, 0.0f, 1.0f), invGamma)*(255.f));
	retVal.y = (unsigned char)(powf(clamp(v.y, 0.0f, 1.0f), invGamma)*(255.f));
	retVal.z = (unsigned char)(powf(clamp(v.z, 0.0f, 1.0f), invGamma)*(255.f));
	retVal.w = 255u;
	return retVal;
}
__host__ __device__ Vector3Df testSamplePixel(int x, int y, int width, int height);
__host__ __device__ float intersectAllTriangles(geom::Triangle* p_triangles, int numTriangles, geom::RayHit &hitData, const geom::Ray& ray);
__host__ __device__ Vector3Df radiance(TrianglesData* p_triData, LightsData* p_lightData, curandState* p_randState);

class Renderer {
protected:
	__host__ Renderer() {}
	__host__ Renderer(Scene* _scenePtr, int _width, int _height, int _samples, bool _useBVH);
	Scene* p_scene;
	int width;
	int height;
	int samples;
	int useBVH;
public:
	uchar4* h_imgPtr;
	virtual ~Renderer() { delete[] h_imgPtr;	}
	__host__ virtual void renderOneSamplePerPixel() = 0;
	__host__ virtual void copyImageBytes() = 0;
	__host__ Scene* getScenePtr() { return p_scene; }
	__host__ int getWidth() { return width; }
	__host__ int getHeight() { return height; }
	__host__ int getSamples() { return samples; }
	__host__ bool getUseBVH() { return useBVH; }
	__host__ void createSettingsData(SettingsData* p_settingsData);
	__host__ void createTrianglesData(TrianglesData* p_trianglesData, geom::Triangle* p_triangles);
	__host__ void createLightsData(LightsData* p_lightsData, geom::Triangle* p_triangles);
};

class ParallelRenderer : public Renderer {
public:
	__host__ ParallelRenderer() : Renderer() {}
	__host__ ParallelRenderer(Scene* _scenePtr, int _width, int _height, int _samples, bool _useBVH);
	__host__ void renderOneSamplePerPixel();
	~ParallelRenderer();
private:
	Vector3Df* d_imgVectorPtr;
	uchar4* d_imgBytesPtr;
	LightsData* d_lightsData;
	TrianglesData* d_trianglesData;
	SettingsData d_settingsData;
	geom::Triangle* d_triPtr;
	geom::Triangle* d_lightsPtr;
	Camera* d_camPtr;
	curandState* d_curandStatePtr;
	// TODO: Consider storing block, grid instead
	unsigned int threadsPerBlock;
	unsigned int gridBlocks;

	__host__ void copyMemoryToCuda();
	__host__ void initializeCurand();
	__host__ void copyImageBytes();
};

class SequentialRenderer : public Renderer {
public:
	SequentialRenderer() : Renderer() {}
	SequentialRenderer(Scene* _scenePtr, int _width, int _height, int _samples, bool _useBVH);
	__host__ void renderOneSamplePerPixel();
	__host__ void copyImageBytes();
	~SequentialRenderer();
private:
	uchar4* h_imgBytesPtr;
	Vector3Df* h_imgVectorPtr;
	SettingsData h_settingsData;
	TrianglesData* h_trianglesData;
	LightsData* h_lightsData;
};

#endif /* RENDERER_H_ */
