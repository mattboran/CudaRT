#ifndef __TEST_RENDER__
#define __TEST_RENDER__
#include "pathtrace.h"
struct BBox;

namespace Sequential {
	Vector3Df* pathtraceWrapper(Scene& scene, int width, int height, int samples, int numStreams, bool useBVH);
}
#endif
