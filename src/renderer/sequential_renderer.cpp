#include "renderer.h"

#include <algorithm>
#include <iterator>
#include <string.h>

using namespace geom;

SequentialRenderer::SequentialRenderer(Scene* _scenePtr, int _width, int _height, int _samples, bool _useBVH) :
  Renderer(_scenePtr, _width, _height, _samples, _useBVH)
{
    int numTriangles = p_scene->getNumTriangles();
    int numLights = p_scene->getNumLights();
    Triangle* p_triangles = p_scene->getTriPtr();
    Triangle* p_lights = p_scene->getLightsPtr();

    size_t trianglesBytes = sizeof(Triangle) * numTriangles;
    size_t lightsBytes = sizeof(Triangle) * numLights;
    size_t trianglesDataBytes = sizeof(TrianglesData) + trianglesBytes;
    size_t lightsDataBytes = sizeof(LightsData) + lightsBytes;
    h_trianglesData = (TrianglesData*)malloc(trianglesDataBytes);
    h_lightsData = (LightsData*)malloc(lightsDataBytes);
    h_imgBytesPtr = new uchar4[width * height];
    h_imgVectorPtr = new Vector3Df[width * height];

    createTrianglesData(h_trianglesData, p_triangles);
    createLightsData(h_lightsData, p_lights);
    createSettingsData(&h_settingsData);
}

SequentialRenderer::~SequentialRenderer() {
    free(h_trianglesData);
    free(h_lightsData);
    delete[] h_imgBytesPtr;
    delete[] h_imgVectorPtr;
}

__host__ void SequentialRenderer::renderOneSamplePerPixel() {
    for (unsigned x = 0; x < width; x++) {
        for (unsigned y = 0; y < height; y++) {
            int idx = y * width + x;
            Vector3Df color = testSamplePixel(x, y, width, height);
            h_imgVectorPtr[idx] = color;
            h_imgBytesPtr[idx] = vector3ToUchar4(color);
        }
    }
}

__host__ void SequentialRenderer::copyImageBytes() {
  int pixels = width * height;
	size_t imgBytes = sizeof(uchar4) * pixels;
  memcpy(h_imgPtr, h_imgBytesPtr, imgBytes);
}
