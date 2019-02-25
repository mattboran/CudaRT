#include "renderer.h"
#include <algorithm>
#include <iterator>
#include <string.h>
#ifdef _OPENMP
 #include <omp.h>
#endif

SequentialRenderer::SequentialRenderer(Scene* _scenePtr, pixels_t _width, pixels_t _height, uint _samples) :
  Renderer(_scenePtr, _width, _height, _samples)
{
    uint numTriangles = p_scene->getNumTriangles();
    uint numLights = p_scene->getNumLights();
    uint numBvhNodes = p_scene->getNumBvhNodes();
    uint numMaterials = p_scene->getNumMaterials();
    Triangle* p_triangles = p_scene->getTriPtr();
    LinearBVHNode* p_bvh = p_scene->getBvhPtr();
    Triangle* p_lights = p_scene->getLightsPtr();
    Material* p_materials = p_scene->getMaterialsPtr();

    size_t trianglesBytes = sizeof(Triangle) * numTriangles;
    size_t lightsBytes = sizeof(Triangle) * numLights;
    size_t bvhBytes = sizeof(LinearBVHNode) * numBvhNodes;
    size_t materialsBytes = sizeof(Material) * numMaterials;
    size_t SceneDataBytes = sizeof(SceneData) + trianglesBytes + bvhBytes + materialsBytes;
    size_t lightsDataBytes = sizeof(LightsData) + lightsBytes;
    h_SceneData = (SceneData*)malloc(SceneDataBytes);
    h_lightsData = (LightsData*)malloc(lightsDataBytes);
    h_imgBytesPtr = new uchar4[width * height]();
    h_imgVectorPtr = new Vector3Df[width * height]();

    createSceneData(h_SceneData, p_triangles, p_bvh, p_materials);
    createLightsData(h_lightsData, p_lights);
    createSettingsData(&h_settingsData);
}

SequentialRenderer::~SequentialRenderer() {
    free(h_lightsData);
    free(h_SceneData);
    delete[] h_imgBytesPtr;
    delete[] h_imgVectorPtr;
}

void SequentialRenderer::renderOneSamplePerPixel(uchar4* p_img) {
	samplesRendered++;
	Camera* p_camera = p_scene->getCameraPtr();
	Material* p_materials = p_scene->getMaterialsPtr();
	Sampler* p_sampler = new Sampler();
    #pragma omp parallel for
    for (pixels_t x = 0; x < width; x++) {
        for (pixels_t y = 0; y < height; y++) {
            int idx = y * width + x;
            Vector3Df sample = samplePixel(x, y, p_camera, h_SceneData, h_lightsData, p_materials, p_sampler);
            h_imgVectorPtr[idx] += sample;
            p_img[idx] = vector3ToUchar4(h_imgVectorPtr[idx]/samplesRendered);
        }
    }
	delete p_sampler;
}

void SequentialRenderer::copyImageBytes(uchar4* p_img) {
	pixels_t pixels = width * height;
	size_t imgBytes = sizeof(uchar4) * pixels;
	memcpy(h_imgPtr, p_img, imgBytes);
	for (pixels_t i = 0; i < pixels; i++) {
		gammaCorrectPixel(h_imgPtr[i]);
	}
}
