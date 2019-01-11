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

using namespace geom;

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

__host__ void Renderer::createTrianglesData(TrianglesData* p_trianglesData, Triangle* p_triangles) {
	p_trianglesData->p_triangles = p_triangles;
	p_trianglesData->numTriangles = p_scene->getNumTriangles();
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
    RayHit rayHit;
    float t = 0.0f;
    SurfaceInteraction interaction;
    Triangle* p_triangles = p_trianglesData->p_triangles;
    Triangle* p_hitTriangle = NULL;
    int numTriangles = p_trianglesData->numTriangles;
    for (unsigned bounces = 0; bounces < 6; bounces++) {
        t = intersectAllTriangles(p_triangles, numTriangles, rayHit, ray);
        if (t >= FLT_MAX) {
            break;
        }
        p_hitTriangle = rayHit.p_hitTriangle;
        if (bounces == 0) {
        	color += mask * p_hitTriangle->_colorEmit;
        }
        interaction.position = ray.pointAlong(ray.tMax);
        interaction.normal = p_hitTriangle->getNormal(rayHit);
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
        mask *= sampleDiffuseBSDF(&interaction, rayHit, p_sampler) / interaction.pdf;

        ray.origin = interaction.position;
        ray.dir = interaction.inputDirection;
        ray.tMin = EPSILON;
        ray.tMax = FLT_MAX;
    }
    return color;
}

__host__ __device__ float intersectAllTriangles(Triangle* p_triangles, int numTriangles, RayHit &hitData, Ray& ray) {
	float t = ray.tMax;
	float tprime = ray.tMax;
	float u, v;
	Triangle* p_current = p_triangles;
	while(numTriangles--) {
		tprime = p_current->intersect(ray, u, v);
		if (tprime < t && tprime > ray.tMin) {
			t = tprime;
			hitData.p_hitTriangle = p_current;
			hitData.u = u;
			hitData.v = v;
		}
		p_current++;
	}
	ray.tMax = t;
	return t;
}

__host__ __device__ Vector3Df sampleDiffuseBSDF(SurfaceInteraction* p_interaction, const RayHit& rayHit, Sampler* p_sampler) {
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
   return rayHit.p_hitTriangle->_colorDiffuse * cosineWeight;
}

__host__ __device__ Vector3Df estimateDirectLighting(Triangle* p_light, TrianglesData* p_trianglesData, const SurfaceInteraction &interaction, Sampler* p_sampler) {
	Vector3Df directLighting(0.0f, 0.0f, 0.0f);
	if (sameTriangle(interaction.p_hitTriangle, p_light)) {
		return directLighting;
	}
	//if specular, return directLighting
	Ray ray(interaction.position,  normalize(p_light->getRandomPointOn(p_sampler) - interaction.position));
	RayHit rayHit;
	// Sample the light
	Triangle* p_triangles = p_trianglesData->p_triangles;
	float t = intersectAllTriangles(p_triangles, p_trianglesData->numTriangles, rayHit, ray);
	if (t < FLT_MAX && sameTriangle(rayHit.p_hitTriangle, p_light)) {
		float surfaceArea = p_light->_surfaceArea;
		float distanceSquared = t*t;
		float incidenceAngle = fabs(dot(p_light->getNormal(rayHit), -ray.dir));
		float weightFactor = surfaceArea/distanceSquared * incidenceAngle;
		directLighting += p_light->_colorEmit * weightFactor;
	}
	return directLighting;
}

// __host__ __device__ bool rayIntersectsBox(const Ray& ray, BVHNode* bbox) {
// 	float t0 = -FLT_MAX, t1 = FLT_MAX;
//
// 	float invRayDir = 1.f/ray.dir.x;
// 	float tNear = (bbox->min.x - ray.origin.x) * invRayDir;
// 	float tFar = (bbox->max.x - ray.origin.x) * invRayDir;
// 	if (tNear > tFar) {
// 		float tmp = tNear;
// 		tNear = tFar;
// 		tFar = tmp;
// 	}
// 	t0 = tNear > t0 ? tNear : t0;
// 	t1 = tFar < t1 ? tFar : t1;
// 	if (t0 > t1) return false;
//
// 	invRayDir = 1.f/ray.dir.y;
// 	tNear = (bbox->min.y - ray.origin.y) * invRayDir;
// 	tFar = (bbox->max.y - ray.origin.y) * invRayDir;
// 	if (tNear > tFar) {
// 		float tmp = tNear;
// 		tNear = tFar;
// 		tFar = tmp;
// 	}
// 	t0 = tNear > t0 ? tNear : t0;
// 	t1 = tFar < t1 ? tFar : t1;
// 	if (t0 > t1) return false;
//
// 	invRayDir = 1.f/ray.dir.z;
// 	tNear = (bbox->min.z - ray.origin.z) * invRayDir;
// 	tFar = (bbox->max.z - ray.origin.z) * invRayDir;
// 	if (tNear > tFar) {
// 		float tmp = tNear;
// 		tNear = tFar;
// 		tFar = tmp;
// 	}
// 	t0 = tNear > t0 ? tNear : t0;
// 	t1 = tFar < t1 ? tFar : t1;
// 	if (t0 > t1) return false;
//
// 	return true;
// }

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
