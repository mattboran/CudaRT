#ifndef PATHTRACE_CU
#define PATHTRACE_CU

#include "scene.h"

const unsigned blockWidth = 16u;

Vector3Df* pathtraceWrapper(Scene& scene, int width, int height, int samples, bool useTexMemory);

#endif
