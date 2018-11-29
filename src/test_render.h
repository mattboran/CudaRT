#ifndef __TEST_RENDER__
#define __TEST_RENDER__
#include "pathtrace.h"
struct BBox;

Vector3Df* testRenderWrapper(Scene& scene, int width, int height, int samples, int numStreams, bool &useTexMemory, int argc, char** argv);
void testRender(BBox* bboxPtr, Camera* camPtr, Vector3Df* imgPtr, int width, int height);
#endif
