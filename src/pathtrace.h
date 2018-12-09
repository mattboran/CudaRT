#ifndef PATHTRACE_CU
#define PATHTRACE_CU

#include "bvh.h"
#include "scene.h"
#include <ctime>

// TODO: Move to util.h
struct Clock {
	unsigned firstValue;
	Clock() { reset(); }
	void reset() { firstValue = clock(); }
	double readS() { return (double)(clock() - firstValue) / (double)(CLOCKS_PER_SEC); }
};

namespace Parallel {
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
		// TODO: Decomission numStreams
		int numStreams;
		bool useBVH;

	} __attribute__ ((aligned (32)));

	Vector3Df* pathtraceWrapper(Scene& scene, int width, int height, int samples, int numStreams, bool useBVH);
}
#endif
