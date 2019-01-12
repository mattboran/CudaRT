/*
 * renderer.cpp

 *
 *  Created on: Dec 23, 2018
 *      Author: matt
 *
 *  This file contains functions that both the sequential
 *  and parallel renderers use
 */

#include "renderer.h"

#include <float.h>
#include <math.h>

__host__ __device__ float Sampler::getNextFloat() {
	#ifdef __CUDA_ARCH__
		return curand_uniform(p_curandState);
	#else
		return rand() / (float)RAND_MAX;
	#endif
}

__host__ Renderer::Renderer(Scene* _scenePtr, int _width, int _height, int _samples, bool _useBVH) :
	p_scene(_scenePtr), width(_width), height(_height), samples(_samples), useBVH(_useBVH)
{
	h_imgPtr = new uchar4[width*height];
}


__host__ void Renderer::createSettingsData(SettingsData* p_settingsData){
	p_settingsData->width = getWidth();
	p_settingsData->height = getHeight();
	p_settingsData->samples = getSamples();
	p_settingsData->useBVH = getUseBVH();
}

__host__ void Renderer::createTrianglesData(TrianglesData* p_trianglesData, Triangle* p_triangles, BVHBuildNode* p_bvh) {
	p_trianglesData->p_triangles = p_triangles;
	p_trianglesData->p_bvh = p_bvh;
	p_trianglesData->numTriangles = p_scene->getNumTriangles();
	p_trianglesData->numBVHNodes = p_scene->getNumBvhNodes();
}

__host__ void Renderer::createLightsData(LightsData* p_lightsData, Triangle* p_triangles) {
	p_lightsData->lightsPtr = p_triangles;
	p_lightsData->numLights = p_scene->getNumLights();
	p_lightsData->totalSurfaceArea = p_scene->getLightsSurfaceArea();
}

__host__ __device__ Vector3Df samplePixel(int x, int y, Camera* p_camera, TrianglesData* p_trianglesData, LightsData *p_lightsData, Sampler* p_sampler) {
	Ray ray = p_camera->computeCameraRay(x, y, p_sampler);

    Vector3Df color(0.f, 0.f, 0.f);
    Vector3Df mask(1.f, 1.f, 1.f);
    SurfaceInteraction interaction;
    Triangle* p_triangles = p_trianglesData->p_triangles;
    Triangle* p_hitTriangle = NULL;
    int numTriangles = p_trianglesData->numTriangles;
    for (unsigned bounces = 0; bounces < 6; bounces++) {
        if (!intersectTriangles(p_triangles, numTriangles, interaction, ray)) {
            break;
        }
        p_hitTriangle = interaction.p_hitTriangle;
        if (bounces == 0) {
        	color += mask * p_hitTriangle->_colorEmit;
        }
        interaction.position = ray.origin + ray.dir * ray.tMax;
        interaction.normal = p_hitTriangle->getNormal(interaction.u, interaction.v);
        interaction.outputDirection = normalize(ray.dir);
        interaction.p_hitTriangle = p_hitTriangle;

        //IF DIFFUSE
		{
			float randomNumber = p_sampler->getNextFloat() * ((float)p_lightsData->numLights - 1.0f + 0.9999999f);
			int selectedLightIdx = (int)truncf(randomNumber);
			Triangle* p_light = &p_lightsData->lightsPtr[selectedLightIdx];
			Vector3Df directLighting = estimateDirectLighting(p_light, p_trianglesData, interaction, p_sampler);

			bool clampRadiance = true;
			if (clampRadiance){
				// This introduces bias!!!
				directLighting.x = clamp(directLighting.x, 0.0f, 1.0f);
				directLighting.y = clamp(directLighting.y, 0.0f, 1.0f);
				directLighting.z = clamp(directLighting.z, 0.0f, 1.0f);
			}
			color += mask * directLighting;
		}
        mask *= sampleDiffuseBSDF(&interaction, p_hitTriangle, p_sampler) / interaction.pdf;

        ray.origin = interaction.position;
        ray.dir = interaction.inputDirection;
        ray.tMin = EPSILON;
        ray.tMax = FLT_MAX;
    }
    return color;
}

__host__ __device__ bool intersectTriangles(Triangle* p_triangles, int numTriangles, SurfaceInteraction &interaction, Ray& ray) {
	const float tInitial = ray.tMax;
	float t;
	float u, v;
	Triangle* p_current = p_triangles;
	while(numTriangles--) {
		t = p_current->intersect(ray, u, v);
		if (t < ray.tMax && t > ray.tMin) {
			ray.tMax = t;
			interaction.p_hitTriangle = p_current;
			interaction.u = u;
			interaction.v = v;
		}
		p_current++;
	}
	return ray.tMax < tInitial;
}

//__host__ __device__ bool traver

__host__ __device__ bool rayIntersectsBox(const Ray& ray, const Vector3Df& min, const Vector3Df& max) {
	float t0 = -FLT_MAX, t1 = FLT_MAX;

	float invRayDir = 1.f/ray.dir.x;
	float tNear = (min.x - ray.origin.x) * invRayDir;
	float tFar = (max.x - ray.origin.x) * invRayDir;
	if (tNear > tFar) {
		float tmp = tNear;
		tNear = tFar;
		tFar = tmp;
	}
	t0 = tNear > t0 ? tNear : t0;
	t1 = tFar < t1 ? tFar : t1;
	if (t0 > t1) return false;

	invRayDir = 1.f/ray.dir.y;
	tNear = (min.y - ray.origin.y) * invRayDir;
	tFar = (max.y - ray.origin.y) * invRayDir;
	if (tNear > tFar) {
		float tmp = tNear;
		tNear = tFar;
		tFar = tmp;
	}
	t0 = tNear > t0 ? tNear : t0;
	t1 = tFar < t1 ? tFar : t1;
	if (t0 > t1) return false;

	invRayDir = 1.f/ray.dir.z;
	tNear = (min.z - ray.origin.z) * invRayDir;
	tFar = (max.z - ray.origin.z) * invRayDir;
	if (tNear > tFar) {
		float tmp = tNear;
		tNear = tFar;
		tFar = tmp;
	}
	t0 = tNear > t0 ? tNear : t0;
	t1 = tFar < t1 ? tFar : t1;
	if (t0 > t1) return false;

	return true;
}

__host__ __device__ Vector3Df sampleDiffuseBSDF(SurfaceInteraction* p_interaction, Triangle* p_hitTriangle, Sampler* p_sampler) {
   float r1 = 2 * M_PI * p_sampler->getNextFloat();
   float r2 = p_sampler->getNextFloat();
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
   return p_hitTriangle->_colorDiffuse * cosineWeight;
}

__host__ __device__ Vector3Df estimateDirectLighting(Triangle* p_light, TrianglesData* p_trianglesData, const SurfaceInteraction &interaction, Sampler* p_sampler) {
	Vector3Df directLighting(0.0f, 0.0f, 0.0f);
	if (sameTriangle(interaction.p_hitTriangle, p_light)) {
		return directLighting;
	}
	//if specular, return directLighting
	Ray ray(interaction.position,  normalize(p_light->getRandomPointOn(p_sampler) - interaction.position));
	SurfaceInteraction lightInteraction;
	// Sample the light
	Triangle* p_triangles = p_trianglesData->p_triangles;
	bool intersectsLight = intersectTriangles(p_triangles, p_trianglesData->numTriangles, lightInteraction, ray);
	if (intersectsLight && sameTriangle(lightInteraction.p_hitTriangle, p_light)) {
		float surfaceArea = p_light->_surfaceArea;
		float distanceSquared = ray.tMax*ray.tMax;
		float incidenceAngle = fabs(dot(p_light->getNormal(lightInteraction.u, lightInteraction.v), -ray.dir));
		float weightFactor = surfaceArea/distanceSquared * incidenceAngle;
		directLighting += p_light->_colorEmit * weightFactor;
	}
	return directLighting;
}

__host__ __device__ void gammaCorrectPixel(uchar4 &p) {
	float invGamma = 1.f/2.2f;
	float3 fp;
	fp.x = pow((float)p.x * 1.f/255.f, invGamma);
	fp.y = pow((float)p.y * 1.f/255.f, invGamma);
	fp.z = pow((float)p.z * 1.f/255.f, invGamma);
	p.x = (unsigned char)(fp.x * 255.f);
	p.y = (unsigned char)(fp.y * 255.f);
	p.z = (unsigned char)(fp.z * 255.f);
}
