#ifndef __SEQUENTIAL_H__
#define __SEQUENTIAL_H__
#include "pathtrace.h"
struct BBox;

namespace Sequential {
	Vector3Df* pathtraceWrapper(Scene& scene, int width, int height, int samples, int numStreams, bool useBVH);
}
#endif
