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

#include <math.h>

__host__ Renderer::Renderer(Scene* _scenePtr, int _width, int _height, int _samples, bool _useBVH) :
	p_scene(_scenePtr), width(_width), height(_height), samples(_samples), useBVH(_useBVH)
{
	h_imgPtr = (uchar4*)malloc(sizeof(Vector3Df) * width * height);
}

__host__ __device__ Vector3Df testSamplePixel(int x, int y, int width, int height) {
	Vector3Df retVal;
	retVal.x = (float)x/(float)width;
	retVal.y = (float)y/(float)height;
	retVal.z = 1.0f;
	return retVal;
}
