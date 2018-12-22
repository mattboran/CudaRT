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
	Renderer(Scene* _scenePtr, int _width, int _height, int _samples, bool _useBVH) :
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
	virtual void renderOneSamplePerPixel() = 0;
	int getWidth() { return width; }
	int getHeight() { return height; }
	int getSamples() { return samples; }
	int getNumStreams() { return numStreams; }
	bool getUseBVH() { return useBVH; }
};

class ParallelRenderer : public Renderer {
public:
	ParallelRenderer(Scene* _scenePtr, int _width, int _height, int _samples, bool _useBVH) :
		Renderer(_scenePtr, _width, _height, _samples, _useBVH) {
		h_lightsData = NULL;
		h_trianglesData = NULL;
		h_settingsData = NULL;
	}
	void renderOneSample();
	~ParallelRenderer();
private:
	LightsData* h_lightsData;
	TrianglesData* h_trianglesData;
	SettingsData* h_settingsData;
	void createHostDataStructures();

};

class SequentialRenderer : public Renderer {
	SequentialRenderer(Scene* _scenePtr, int _width, int _height, int _samples, bool _useBVH) :
		Renderer(_scenePtr, _width, _height, _samples, _useBVH) {}
	void renderOneSample() {
		// not implemented yet
	}
	~SequentialRenderer();
};

#endif /* RENDERER_H_ */
