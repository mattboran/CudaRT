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

__host__ void Renderer::createTrianglesData(TrianglesData* p_trianglesData) {

}

__host__ void Renderer::createLightsData(LightsData* p_lightsData) {

}

__host__ __device__ Vector3Df testSamplePixel(int x, int y, int width, int height) {
	Vector3Df retVal;
	retVal.x = (float)x/(float)width;
	retVal.y = (float)y/(float)height;
	retVal.z = 1.0f;
	return retVal;
}

__host__ __device__ float intersectAllTriangles(Triangle* p_triangles, int numTriangles, RayHit &hitData, const Ray& ray) {
	float t = FLT_MAX;
	float tprime = FLT_MAX;
	float u, v;
	Triangle* p_current = p_triangles;
	while(numTriangles--) {
		tprime = p_current->intersect(ray, u, v);
		if (tprime < t && tprime > 0.f) {
			t = tprime;
			hitData.pHitTriangle = p_current;
			hitData.u = u;
			hitData.v = v;
		}
		p_current++;
	}
	return t;
}
