#include "renderer.h"
#include <algorithm>
#include <iterator>
#include <string.h>
#ifdef _OPENMP
 #include <omp.h>
#endif

// Sequential version of constant memory for materials
static float3 materialFloats[MAX_MATERIALS * MATERIALS_FLOAT_COMPONENTS];
static int2 materialIndices[MAX_MATERIALS];

SequentialRenderer::SequentialRenderer(Scene* _scenePtr, pixels_t _width, pixels_t _height, uint _samples) :
  Renderer(_scenePtr, _width, _height, _samples)
{
    uint numTriangles = p_scene->getNumTriangles();
    uint numLights = p_scene->getNumLights();
    uint numBvhNodes = p_scene->getNumBvhNodes();
    uint numMaterials = p_scene->getNumMaterials();
    uint numTextures = p_scene->getNumTextures();
    pixels_t numTexturePixels = p_scene->getTotalTexturePixels();

    Triangle* p_triangles = p_scene->getTriPtr();
    LinearBVHNode* p_bvh = p_scene->getBvhPtr();
    Triangle* p_lights = p_scene->getLightsPtr();
    Material* p_materials = p_scene->getMaterialsPtr();
    Vector3Df* p_textureData = p_scene->getTexturePtr();
    pixels_t* p_textureDimensions = p_scene->getTextureDimensionsPtr();
    pixels_t* p_textureOffsets = p_scene->getTextureOffsetsPtr();

    size_t trianglesBytes = sizeof(Triangle) * numTriangles;
    size_t lightsBytes = sizeof(Triangle) * numLights;
    size_t bvhBytes = sizeof(LinearBVHNode) * numBvhNodes;
    size_t materialsBytes = sizeof(Material) * numMaterials;
    size_t texturePixelsBytes = sizeof(Vector3Df) * numTexturePixels;
    size_t textureOffsetsBytes = sizeof(pixels_t) * numTextures + sizeof(pixels_t);
    size_t textureDimensionsBytes = textureOffsetsBytes * 2;
    size_t totalTextureBytes = texturePixelsBytes + textureOffsetsBytes + textureDimensionsBytes;

    size_t SceneDataBytes = sizeof(SceneData) + trianglesBytes + bvhBytes + materialsBytes + totalTextureBytes;
    size_t lightsDataBytes = sizeof(LightsData) + lightsBytes;
    h_sceneData = (SceneData*)malloc(SceneDataBytes);
    h_lightsData = (LightsData*)malloc(lightsDataBytes);
    h_imgBytesPtr = new uchar4[width * height]();
    h_imgVectorPtr = new Vector3Df[width * height]();

    createSceneData(h_sceneData, p_triangles, p_bvh, p_textureData, p_textureDimensions, p_textureOffsets);
    createLightsData(h_lightsData, p_lights);
    createSettingsData(&h_settingsData);

    createMaterialsData(materialFloats, materialIndices);
}

__host__ void SequentialRenderer::createMaterialsData(float3* matFloats, int2* matIndices) {
    Material* p_materials = p_scene->getMaterialsPtr();
    uint numMaterials = p_scene->getNumMaterials();
    float3* p_currentFloat = matFloats;
    int2* p_currentIndex = matIndices;
    for (uint i = 0; i < numMaterials; i++) {
        *p_currentFloat++ = make_float3(p_materials[i].kd);
        *p_currentFloat++ = make_float3(p_materials[i].ka);
        *p_currentFloat++ = make_float3(p_materials[i].ks);
        *p_currentFloat++ = make_float3(p_materials[i].ns,
                                        p_materials[i].ni,
                                        p_materials[i].diffuseCoefficient);
        *p_currentIndex++ = make_int2((int32_t)p_materials[i].bsdf,
                                      (int32_t)p_materials[i].texKdIdx);

    }
}

SequentialRenderer::~SequentialRenderer() {
    free(h_lightsData);
    free(h_sceneData);
    delete[] h_imgBytesPtr;
    delete[] h_imgVectorPtr;
}

void SequentialRenderer::renderOneSamplePerPixel(uchar4* p_img) {
	samplesRendered++;
	Camera camera = *p_scene->getCameraPtr();
	Sampler* p_sampler = new Sampler();
    #pragma omp parallel for
    for (pixels_t x = 0; x < width; x++) {
        for (pixels_t y = 0; y < height; y++) {
            int idx = y * width + x;
            Vector3Df sample = samplePixel(x, y,
                                           camera,
                                           h_sceneData,
                                           h_lightsData,
                                           p_sampler,
                                           materialFloats,
                                           materialIndices);
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
