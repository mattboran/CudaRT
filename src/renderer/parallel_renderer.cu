/*
 * parallel_renderer.cpp
 *
 *  Created on: Dec 22, 2018
 *      Author: matt
 */

#include "renderer.h"
#include "cuda_error_check.h"

using namespace geom;

__host__ ParallelRenderer::ParallelRenderer(Scene* _scenePtr, int _width, int _height, int _samples, bool _useBVH) :
	Renderer(_scenePtr, _width, _height, _samples, _useBVH) {
	int pixels = width * height;
	unsigned numTris = scenePtr->getNumTriangles();
	unsigned numLights = scenePtr->getNumLights();
	unsigned numBVHNodes = scenePtr->getNumBVHNodes();

	d_camPtr = NULL;
	d_triPtr = NULL;
	d_lightsPtr = NULL;
	d_imgPtr = NULL;
	d_lightsData = NULL;
	d_trianglesData = NULL;
	h_imgPtr = (Vector3Df*)malloc(sizeof(Vector3Df) * pixels);

	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_camPtr, sizeof(Camera)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_triPtr, sizeof(Triangle) * numTris));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_lightsPtr, sizeof(Triangle) * numLights));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_imgPtr, sizeof(Vector3Df) * pixels));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_lightsData, sizeof(LightsData) + sizeof(Triangle) * numLights));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_trianglesData, sizeof(TrianglesData) + sizeof(Triangle) * numTris));

	createSettingsData(&d_settingsData);
	copyMemoryToCuda();
}

__host__ ParallelRenderer::~ParallelRenderer() {
	free(h_imgPtr);
	cudaFree(d_lightsData);
	cudaFree(d_trianglesData);
	cudaFree(d_camPtr);
	cudaFree(d_triPtr);
	cudaFree(d_lightsPtr);
	cudaFree(d_imgPtr);
}

__host__ void ParallelRenderer::createSettingsData(SettingsData* p_settingsData){
	p_settingsData->width = getWidth();
	p_settingsData->height = getHeight();
	p_settingsData->samples = getSamples();
	p_settingsData->numStreams = getNumStreams();
	p_settingsData->useBVH = getUseBVH();
}

__host__ void ParallelRenderer::copyMemoryToCuda() {
	Scene* p_scene = getScenePtr();
	int numTriangles = p_scene->getNumTriangles();
	int numLights = p_scene->getNumLights();
	float lightsSurfaceArea = p_scene->getLightsSurfaceArea();

	Camera* h_camPtr = p_scene->getCameraPtr();
	Triangle* h_triPtr = p_scene->getTriPtr();
	Triangle* h_lightsPtr = p_scene->getLightsPtr();
	TrianglesData* h_trianglesData = (TrianglesData*)malloc(sizeof(TrianglesData) + sizeof(Triangle) * numTriangles);
	LightsData* h_lightsData = (LightsData*)malloc(sizeof(LightsData) + sizeof(Triangle) * numLights);

	CUDA_CHECK_RETURN(cudaMemcpy(d_camPtr, h_camPtr, sizeof(Camera), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_triPtr, h_triPtr, sizeof(Triangle) * numTriangles, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_lightsPtr, h_lightsPtr, sizeof(Triangle) * numLights, cudaMemcpyHostToDevice));

	h_trianglesData->triPtr = d_triPtr;
	h_trianglesData->numTriangles = numTriangles;
	h_trianglesData->bvhPtr = NULL;
	h_trianglesData->bvhIndexPtr = NULL;
	h_trianglesData->numBVHNodes = 0;
	CUDA_CHECK_RETURN(cudaMemcpy(d_trianglesData, h_triPtr, sizeof(TrianglesData) + sizeof(Triangle) * numTriangles, cudaMemcpyHostToDevice));

	h_lightsData->lightsPtr = d_lightsPtr;
	h_lightsData->numLights = numLights;
	h_lightsData->totalSurfaceArea = lightsSurfaceArea;
	CUDA_CHECK_RETURN(cudaMemcpy(d_lightsData, h_lightsData, sizeof(LightsData) + sizeof(Triangle) * numLights, cudaMemcpyHostToDevice));

	free(h_trianglesData);
	free(h_lightsData);
}

__host__ void ParallelRenderer::renderOneSamplePerPixel() {
	// First setup settings data

}


