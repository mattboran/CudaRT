#ifndef PATHTRACE_CU
#define PATHTRACE_CU

#include "bvh.h"
#include "scene.h"
#include <ctime>

const unsigned blockWidth = 16u;

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

// TODO: Move image, camera, and curandState pointers into here
struct SettingsData {
	int width;
	int height;
	int samples;
	bool useTexMem;
	int numStreams;
} __attribute__ ((aligned (32)));

struct Clock {
	unsigned firstValue;
	Clock() { reset(); }
	void reset() { firstValue = clock(); }
	double readS() { return (double)(clock() - firstValue) / (double)(CLOCKS_PER_SEC); }
};

Vector3Df* pathtraceWrapper(Scene& scene, int width, int height, int samples, int numStreams, bool &useTexMemory);
__global__ void setupCurandKernel(curandState *randState, int streamOffset);
__global__ void renderKernel(TrianglesData* d_tris, Camera* d_camPtr, Vector3Df* d_imgPtr, LightsData* d_lights, SettingsData* d_settings, curandState *randState, int streamId);
__global__ void averageSamplesAndGammaCorrectKernel(Vector3Df* d_streamImgDataPtr, Vector3Df* d_imgPtr, SettingsData* d_settings);
__device__ float intersectTriangles(geom::Triangle* d_triPtr, int numTriangles, geom::RayHit& hitData, const geom::Ray& ray, bool useTexMem);
__device__ bool rayIntersectsBox(const geom::Ray& ray, CacheFriendlyBVHNode *bvhNode);
__device__ float intersectBVH(CacheFriendlyBVHNode* d_bvh, geom::Triangle* d_triPtr, unsigned* d_triIndexPtr, geom::RayHit& hitData, const geom::Ray& ray, bool useTexMemory);

#endif
