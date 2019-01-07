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
	geom::Triangle* p_triangles;
	unsigned numTriangles;
};

struct SettingsData {
	int width;
	int height;
	int samples;
	bool useBVH;
};

__host__ __device__ inline uchar4 vector3ToUchar4(const Vector3Df& v) {
	uchar4 retVal;
	retVal.x = (unsigned char)(clamp(v.x, 0.0f, 1.0f)*(255.f));
	retVal.y = (unsigned char)(clamp(v.y, 0.0f, 1.0f)*(255.f));
	retVal.z = (unsigned char)(clamp(v.z, 0.0f, 1.0f)*(255.f));
	retVal.w = 255u;
	return retVal;
}
__host__ __device__ inline bool sameTriangle(geom::Triangle* p_a, geom::Triangle* p_b) {
	return p_a->_triId == p_b->_triId;
}
__host__ __device__ Vector3Df testSamplePixel(int x, int y, int width, int height);
__host__ __device__ float intersectAllTriangles(geom::Triangle* p_triangles, int numTriangles, geom::RayHit &hitData, geom::Ray& ray);
__host__ __device__ void gammaCorrectPixel(uchar4 &p);
class Renderer {
protected:
	__host__ Renderer() {}
	__host__ Renderer(Scene* _scenePtr, int _width, int _height, int _samples, bool _useBVH);
	Scene* p_scene;
	int width;
	int height;
	int samples;
	int useBVH;
	int samplesRendered = 0;
public:
	bool useCuda = false;
	uchar4* h_imgPtr;
	virtual ~Renderer() { delete[] h_imgPtr;	}
	__host__ virtual void renderOneSamplePerPixel(uchar4* p_img) = 0;
	__host__ virtual void copyImageBytes() = 0;
	__host__ virtual uchar4* getImgBytesPointer() = 0;
	__host__ Scene* getScenePtr() { return p_scene; }
	__host__ int getWidth() { return width; }
	__host__ int getHeight() { return height; }
	__host__ int getSamples() { return samples; }
	__host__ bool getUseBVH() { return useBVH; }
	__host__ int getSamplesRendered() { return samplesRendered; }
	__host__ void createSettingsData(SettingsData* p_settingsData);
	__host__ void createTrianglesData(TrianglesData* p_trianglesData, geom::Triangle* p_triangles);
	__host__ void createLightsData(LightsData* p_lightsData, geom::Triangle* p_triangles);
};

class ParallelRenderer : public Renderer {
public:
	__host__ ParallelRenderer() : Renderer() {}
	__host__ ParallelRenderer(Scene* _scenePtr, int _width, int _height, int _samples, bool _useBVH);
	__host__ void renderOneSamplePerPixel(uchar4* p_img);
	__host__ void copyImageBytes();
	__host__ uchar4* getImgBytesPointer() { return d_imgBytesPtr; }
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
};

class SequentialRenderer : public Renderer {
public:
	SequentialRenderer() : Renderer() {}
	SequentialRenderer(Scene* _scenePtr, int _width, int _height, int _samples, bool _useBVH);
	__host__ void renderOneSamplePerPixel(uchar4* p_img);
	__host__ void copyImageBytes();
	__host__ uchar4* getImgBytesPointer() { return h_imgBytesPtr; }
	~SequentialRenderer();
private:
	uchar4* h_imgBytesPtr;
	Vector3Df* h_imgVectorPtr;
	SettingsData h_settingsData;
	TrianglesData* h_trianglesData;
	LightsData* h_lightsData;
	__host__ __device__ Vector3Df samplePixel(int x, int y);
};

#endif /* RENDERER_H_ */
