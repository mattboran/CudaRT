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
/*
bool hitsBox(const Ray& ray, BVHNode* bbox) {
	float t0 = -FLT_MAX, t1 = FLT_MAX;
	//axes

	float invRayDir = 1.f/ray.dir.x;
	float tNear = (bbox->_bottom.x - ray.origin.x) * invRayDir;
	float tFar = (bbox->_top.x - ray.origin.x) * invRayDir;
	if (tNear > tFar) {
		float tmp = tNear;
		tNear = tFar;
		tFar = tmp;
	}
	t0 = tNear > t0 ? tNear : t0;
	t1 = tFar < t1 ? tFar : t1;
	if (t0 > t1) return false;

	invRayDir = 1.f/ray.dir.y;
	tNear = (bbox->_bottom.y - ray.origin.y) * invRayDir;
	tFar = (bbox->_top.y - ray.origin.y) * invRayDir;
	if (tNear > tFar) {
		float tmp = tNear;
		tNear = tFar;
		tFar = tmp;
	}
	t0 = tNear > t0 ? tNear : t0;
	t1 = tFar < t1 ? tFar : t1;
	if (t0 > t1) return false;

	invRayDir = 1.f/ray.dir.z;
	tNear = (bbox->_bottom.z - ray.origin.z) * invRayDir;
	tFar = (bbox->_top.z - ray.origin.z) * invRayDir;
	if (tNear > tFar) {
		float tmp = tNear;
		tNear = tFar;
		tFar = tmp;
	}
	t0 = tNear > t0 ? tNear : t0;
	t1 = tFar < t1 ? tFar : t1;
	if (t0 > t1) return false;

	return true;
}*/

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
