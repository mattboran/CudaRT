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
	// TODO: Decomission numStreams
	int numStreams;
	bool useBVH;
} __attribute__ ((aligned (32)));

class Renderer {
protected:
	__host__ Renderer(Scene* _scenePtr, int _width, int _height, int _samples, bool _useBVH) :
		scenePtr(_scenePtr), width(_width), height(_height), samples(_samples), useBVH(_useBVH) {
		h_imgPtr = (Vector3Df*)malloc(sizeof(Vector3Df) * width * height);
	}
	Scene* scenePtr;
	int width;
	int height;
	int samples;
	int numStreams = 1;
	int useBVH;
public:
	Vector3Df* h_imgPtr;
	virtual ~Renderer() { free(h_imgPtr);	}
	__host__ virtual void renderOneSamplePerPixel() = 0;
	__host__ Scene* getScenePtr() { return scenePtr; }
	__host__ int getWidth() { return width; }
	__host__ int getHeight() { return height; }
	__host__ int getSamples() { return samples; }
	__host__ int getNumStreams() { return numStreams; }
	__host__ bool getUseBVH() { return useBVH; }
};

class ParallelRenderer : public Renderer {
public:
	__host__ ParallelRenderer(Scene* _scenePtr, int _width, int _height, int _samples, bool _useBVH);
	__host__ void renderOneSamplePerPixel();
	~ParallelRenderer();
private:
	Vector3Df* d_imgPtr;
	LightsData* d_lightsData;
	TrianglesData* d_trianglesData;
	SettingsData d_settingsData;
	geom::Triangle* d_triPtr;
	geom::Triangle* d_lightsPtr;
	Camera* d_camPtr;

	__host__ void copyMemoryToCuda();
	__host__ void createSettingsData(SettingsData* p_settingsData);
};

class SequentialRenderer : public Renderer {
	SequentialRenderer(Scene* _scenePtr, int _width, int _height, int _samples, bool _useBVH) :
		Renderer(_scenePtr, _width, _height, _samples, _useBVH) {}
	void renderOneSamplePerPixel() {
		// not implemented yet
	}
	~SequentialRenderer();
};

#endif /* RENDERER_H_ */
