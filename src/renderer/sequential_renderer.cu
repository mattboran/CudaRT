#include "renderer.h"

#include <algorithm>
#include <iterator>
#include <string.h>

SequentialRenderer::SequentialRenderer(Scene* _scenePtr, int _width, int _height, int _samples) :
  Renderer(_scenePtr, _width, _height, _samples)
{
    unsigned int numTriangles = p_scene->getNumTriangles();
    unsigned int numLights = p_scene->getNumLights();
    unsigned int numBvhNodes = p_scene->getNumBvhNodes();
    Triangle* p_triangles = p_scene->getTriPtr();
    LinearBVHNode* p_bvh = p_scene->getBvhPtr();
    Triangle* p_lights = p_scene->getLightsPtr();

    size_t trianglesBytes = sizeof(Triangle) * numTriangles;
    size_t lightsBytes = sizeof(Triangle) * numLights;
    size_t bvhBytes = sizeof(LinearBVHNode) * numBvhNodes;
    size_t trianglesDataBytes = sizeof(TrianglesData) + trianglesBytes + bvhBytes;
    size_t lightsDataBytes = sizeof(LightsData) + lightsBytes;
    h_trianglesData = (TrianglesData*)malloc(trianglesDataBytes);
    h_lightsData = (LightsData*)malloc(lightsDataBytes);
    h_imgBytesPtr = new uchar4[width * height]();
    h_imgVectorPtr = new Vector3Df[width * height]();

    createTrianglesData(h_trianglesData, p_triangles, p_bvh);
    createLightsData(h_lightsData, p_lights);
    createSettingsData(&h_settingsData);
}

SequentialRenderer::~SequentialRenderer() {
    free(h_lightsData);
    free(h_trianglesData);
    delete[] h_imgBytesPtr;
    delete[] h_imgVectorPtr;
}

__host__ void SequentialRenderer::renderOneSamplePerPixel(uchar4* p_img) {
	samplesRendered++;
	Camera* p_camera = p_scene->getCameraPtr();
	Sampler* p_sampler = new Sampler();
    for (unsigned x = 0; x < width; x++) {
        for (unsigned y = 0; y < height; y++) {
            int idx = y * width + x;
            std::cout<<"On sampleRendered = " << samplesRendered << "\n";
            Vector3Df sample = samplePixel(x, y, p_camera, h_trianglesData, h_lightsData, p_sampler);
            h_imgVectorPtr[idx] += sample;
            std::cout<<"Got the sample: x: " << sample.x << " y: " << sample.y << " z: "<< sample.z<<"\n";
            p_img[idx] = vector3ToUchar4(sample/samplesRendered);
            std::cout<<"p_img[idx]= x: " << p_img[idx].x << " y: " << p_img[idx].y << " z: "<< p_img[idx].z<<"\n";
        }
    }
	delete p_sampler;
}

__host__ void SequentialRenderer::copyImageBytes(uchar4* p_img) {
	int pixels = width * height;
	size_t imgBytes = sizeof(uchar4) * pixels;
	memcpy(h_imgPtr, p_img, imgBytes);
	for (unsigned i = 0; i < pixels; i++) {
		gammaCorrectPixel(h_imgPtr[i]);
	}
}
