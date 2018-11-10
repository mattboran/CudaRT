#ifndef PATHTRACE_CU
#define PATHTRACE_CU

#include "scene.h"

const unsigned blockWidth = 16u;

Vector3Df* pathtraceWrapper(scene::Scene& scene, int width, int height, int samples);

#endif
