#include "renderer.h"

#include <algorithm>
#include <iterator>
#include <string.h>

using namespace geom;

__host__ Vector3Df sampleDiffuseBSDF(SurfaceInteraction* p_interaction, const RayHit& rayHit) {
	float r1 = 2 * M_PI * (rand() / (RAND_MAX + 1.f));
	float r2 = (rand() / (RAND_MAX + 1.f));
	float r2sq = sqrtf(r2);
	// calculate orthonormal coordinates u, v, w, at hitpt
	Vector3Df w = p_interaction->normal;
	Vector3Df u = normalize(cross( (fabs(w.x) > 0.1f ?
				Vector3Df(0.f, 1.f, 0.f) :
				Vector3Df(1.f, 0.f, 0.f)), w));
	Vector3Df v = cross(w, u);
	p_interaction->inputDirection = normalize(u * cosf(r1) * r2sq + v * sinf(r1) * r2sq + w * sqrtf(1.f - r2));
	p_interaction->pdf = 0.5f;
	float cosineWeight = dot(p_interaction->inputDirection, p_interaction->normal);
	return rayHit.pHitTriangle->_colorDiffuse * cosineWeight;
}

__host__ Vector3Df estimateDirectLighting(Triangle* p_light, TrianglesData* p_trianglesData, const SurfaceInteraction &interaction) {
	Vector3Df directLighting(0.0f, 0.0f, 0.0f);
	if (sameTriangle(interaction.pHitTriangle, p_light)) {
		return directLighting;
	}
	//if specular, return directLighting
	Ray ray(interaction.position,  normalize(p_light->getRandomPointOn() - interaction.position));
	RayHit rayHit;
	// Sample the light
	Triangle* p_triangles = p_trianglesData->triPtr;
	float t = intersectAllTriangles(p_triangles, p_trianglesData->numTriangles, rayHit, ray);
	if (t < FLT_MAX && sameTriangle(rayHit.pHitTriangle, p_light)) {
		float surfaceArea = p_light->_surfaceArea;
		float distanceSquared = t*t;
		float incidenceAngle = fabs(dot(p_light->getNormal(rayHit), -ray.dir));
		float weightFactor = surfaceArea/distanceSquared * incidenceAngle;
		directLighting += p_light->_colorEmit * weightFactor;
	}
	return directLighting;
}

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
	samplesRendered++;
    for (unsigned x = 0; x < width; x++) {
        for (unsigned y = 0; y < height; y++) {
            int idx = y * width + x;
            h_imgVectorPtr[idx] += samplePixel(x, y);
            h_imgBytesPtr[idx] = vector3ToUchar4(h_imgVectorPtr[idx]/samplesRendered);
        }
    }
}

__host__ __device__  Vector3Df SequentialRenderer::samplePixel(int x, int y) {
    Ray ray = p_scene->getCameraPtr()->computeSequentialCameraRay(x, y);

    Vector3Df color(0.f, 0.f, 0.f);
    Vector3Df mask(1.f, 1.f, 1.f);
    RayHit rayHit;
    float t = 0.0f;
    SurfaceInteraction interaction;
    Triangle* p_triangles = h_trianglesData->triPtr;
    Triangle* p_hitTriangle = NULL;
    int numTriangles = h_trianglesData->numTriangles;
    for (unsigned bounces = 0; bounces < 6; bounces++) {
        t = intersectAllTriangles(p_triangles, numTriangles, rayHit, ray);
        if (t >= FLT_MAX) {
            break;
        }
        p_hitTriangle = rayHit.pHitTriangle;
        if (bounces == 0) {
        	color += mask * p_hitTriangle->_colorEmit;
        }
        interaction.position = ray.pointAlong(ray.tMax);
        interaction.normal = p_hitTriangle->getNormal(rayHit);
        interaction.outputDirection = normalize(ray.dir);
        interaction.pHitTriangle = p_hitTriangle;

        //IF DIFFUSE
		{
			bool clampRadiance = true;
			Triangle* p_light = &h_lightsData->lightsPtr[rand() % h_lightsData->numLights];
			Vector3Df directLighting = estimateDirectLighting(p_light, h_trianglesData, interaction);
			if (clampRadiance){
				// This introduces bias!!!
				directLighting.x = clamp(directLighting.x, 0.0f, 1.0f);
				directLighting.y = clamp(directLighting.y, 0.0f, 1.0f);
				directLighting.z = clamp(directLighting.z, 0.0f, 1.0f);
			}
			color += mask * directLighting;
		}
        mask *= sampleDiffuseBSDF(&interaction, rayHit) / interaction.pdf;

        ray.origin = interaction.position;
        ray.dir = interaction.inputDirection;
        ray.tMin = EPSILON;
        ray.tMax = FLT_MAX;
    }
    return color;
}

__host__ void SequentialRenderer::copyImageBytes() {
	int pixels = width * height;
	size_t imgBytes = sizeof(uchar4) * pixels;
	memcpy(h_imgPtr, h_imgBytesPtr, imgBytes);
	for (unsigned i = 0; i < pixels; i++) {
		gammaCorrectPixel(h_imgPtr[i]);
	}
}
