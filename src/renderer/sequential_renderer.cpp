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
            Vector3Df color = samplePixel(x, y);
            h_imgVectorPtr[idx] = color;
            h_imgBytesPtr[idx] = vector3ToUchar4(color);
        }
    }
}


__host__ __device__  Vector3Df SequentialRenderer::samplePixel(int x, int y) {
    Ray ray = p_scene->getCameraPtr()->computeSequentialCameraRay(x, y);

    Vector3Df color(0.f, 0.f, 0.f);
    Vector3Df mask(1.f, 1.f, 1.f);
    RayHit hitData;
    float t = 0.0f;

    Triangle* p_triangles = h_trianglesData->triPtr;
    Triangle* p_hitTriangle = NULL;
    int numTriangles = h_trianglesData->numTriangles;

    const unsigned maxBounces = 1;
    for (unsigned bounces = 0; bounces < maxBounces; bounces++) {
        t = intersectAllTriangles(p_triangles, numTriangles, hitData, ray);
        if (t == FLT_MAX) {
            break;
        }
        p_hitTriangle = hitData.pHitTriangle;
        if (p_hitTriangle->isEmissive()) {
            color += mask * p_hitTriangle->_colorEmit;
        }
        color = p_hitTriangle->_colorDiffuse;
    }
    return color;
}

__host__ void SequentialRenderer::copyImageBytes() {
  int pixels = width * height;
	size_t imgBytes = sizeof(uchar4) * pixels;
  memcpy(h_imgPtr, h_imgBytesPtr, imgBytes);
}
