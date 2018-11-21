#ifndef PATHTRACE_CU
#define PATHTRACE_CU

#include "scene.h"

const unsigned blockWidth = 16u;

struct LightsData {
	geom::Triangle* lightsPtr;
	unsigned numLights;
	float totalSurfaceArea;
} __attribute__ ((aligned (32)));

struct TrianglesData {
	geom::Triangle* triPtr;
	unsigned numTriangles;
} __attribute__ ((aligned (32)));

// TODO: Move image, camera, and curandState pointers into here
struct SettingsData {
	int width;
	int height;
	int samples;
	bool useTexMem;
	int numStreams;
} __attribute__ ((aligned (32)));

Vector3Df* pathtraceWrapper(Scene& scene, int width, int height, int samples, int numStreams, bool &useTexMemory);

__global__ void debugRenderKernel(geom::Triangle* d_triPtr, int numTriangles,
		Camera* d_camPtr, Vector3Df* d_imgPtr, int width, int height,
		bool useTexMem);
__global__ void setupCurandKernel(curandState *randState, int streamOffset);
__global__ void renderKernel(TrianglesData* d_tris, Camera* d_camPtr, Vector3Df* d_imgPtr, LightsData* d_lights, SettingsData* d_settings, curandState *randState, int streamId);
__global__ void averageSamplesKernel(Vector3Df* d_streamImgDataPtr, Vector3Df* d_imgPtr, SettingsData* d_settings);
__device__ float intersectTriangles(geom::Triangle* d_triPtr, int numTriangles, geom::RayHit& hitData, const geom::Ray& ray, bool useTexMem);
__device__ inline geom::Triangle getTriangleFromTexture(unsigned i);


#endif
