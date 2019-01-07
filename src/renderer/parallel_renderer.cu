/*
 * parallel_renderer.cpp
 *
 *  Created on: Dec 22, 2018
 *      Author: matt
 */

#include "renderer.h"
#include "cuda_error_check.h"

#include <curand_kernel.h>

#define BLOCK_WIDTH 16u

using namespace geom;
__device__ Vector3Df sampleDiffuseBSDF(SurfaceInteraction* p_interaction, const RayHit& rayHit, curandState* p_curandState);
__device__ Vector3Df estimateDirectLighting(Triangle* p_light, TrianglesData* p_trianglesData, const SurfaceInteraction &interaction, curandState* p_curandState);
__device__ Vector3Df samplePixel(int x, int y, Camera* p_camera, TrianglesData* p_trianglesData, LightsData *p_lightsData, curandState *p_curandState);
// Kernels
__global__ void initializeCurandKernel(curandState* p_curandState);
__global__ void renderKernel(SettingsData settings,
		Vector3Df* p_imgBuffer,
		uchar4* p_outImg,
		Camera* p_camera,
		TrianglesData* p_tris,
		LightsData* p_lights,
		curandState *p_curandState,
		int sampleNumber);

__host__ ParallelRenderer::ParallelRenderer(Scene* _scenePtr, int _width, int _height, int _samples, bool _useBVH) :
	Renderer(_scenePtr, _width, _height, _samples, _useBVH) {
	// CUDA settings
	useCuda = true;
	threadsPerBlock = BLOCK_WIDTH * BLOCK_WIDTH;
	gridBlocks = width / BLOCK_WIDTH * height / BLOCK_WIDTH;

	int pixels = width * height;
	unsigned numTris = p_scene->getNumTriangles();
	unsigned numLights = p_scene->getNumLights();
	unsigned numBVHNodes = p_scene->getNumBVHNodes();
	size_t trianglesBytes = sizeof(Triangle) * numTris;
	size_t lightsBytes = sizeof(Triangle) * numLights;
	size_t curandBytes = sizeof(curandState) * threadsPerBlock * gridBlocks;

	d_imgVectorPtr = NULL;
	d_imgBytesPtr = NULL;
	d_camPtr = NULL;
	d_triPtr = NULL;
	d_lightsPtr = NULL;
	d_trianglesData = NULL;
	d_lightsData = NULL;
	d_curandStatePtr = NULL;

	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_imgVectorPtr, sizeof(Vector3Df) * pixels));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_imgBytesPtr, sizeof(uchar4) * pixels));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_camPtr, sizeof(Camera)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_triPtr, trianglesBytes));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_lightsPtr, lightsBytes));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_trianglesData, sizeof(TrianglesData) + trianglesBytes));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_lightsData, sizeof(LightsData) + lightsBytes));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_curandStatePtr, curandBytes));

	createSettingsData(&d_settingsData);
	copyMemoryToCuda();

	initializeCurand();
}

__host__ ParallelRenderer::~ParallelRenderer() {
	cudaFree(d_imgVectorPtr);
	cudaFree(d_camPtr);
	cudaFree(d_triPtr);
	cudaFree(d_lightsPtr);
	cudaFree(d_trianglesData);
	cudaFree(d_lightsData);
	cudaFree(d_curandStatePtr);
}

__host__ void ParallelRenderer::copyMemoryToCuda() {
	Scene* scenePtr = getScenePtr();
	int numTris = scenePtr->getNumTriangles();
	int numLights = scenePtr->getNumLights();
	float lightsSurfaceArea = scenePtr->getLightsSurfaceArea();
	size_t trianglesBytes = sizeof(Triangle) * numTris;
	size_t lightsBytes = sizeof(Triangle) * numLights;

	Camera* h_camPtr = scenePtr->getCameraPtr();
	Triangle* h_triPtr = scenePtr->getTriPtr();
	Triangle* h_lightsPtr = scenePtr->getLightsPtr();
	TrianglesData* h_trianglesData = (TrianglesData*)malloc(sizeof(TrianglesData) + trianglesBytes);
	LightsData* h_lightsData = (LightsData*)malloc(sizeof(LightsData) + lightsBytes);

	CUDA_CHECK_RETURN(cudaMemcpy(d_camPtr, h_camPtr, sizeof(Camera), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_triPtr, h_triPtr, sizeof(Triangle) * numTris, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_lightsPtr, h_lightsPtr, sizeof(Triangle) * numLights, cudaMemcpyHostToDevice));

	createTrianglesData(h_trianglesData, d_triPtr);
	CUDA_CHECK_RETURN(cudaMemcpy(d_trianglesData, h_trianglesData, sizeof(TrianglesData) + trianglesBytes, cudaMemcpyHostToDevice));

	createLightsData(h_lightsData, d_lightsPtr);
	CUDA_CHECK_RETURN(cudaMemcpy(d_lightsData, h_lightsData, sizeof(LightsData) + lightsBytes, cudaMemcpyHostToDevice));

	free(h_trianglesData);
	free(h_lightsData);
}

__host__ void ParallelRenderer::initializeCurand() {
	dim3 block = dim3(BLOCK_WIDTH, BLOCK_WIDTH, 1);
	dim3 grid = dim3(width/BLOCK_WIDTH, height/BLOCK_WIDTH, 1);

	initializeCurandKernel<<<grid, block, 0>>>(d_curandStatePtr);
}

__host__ void ParallelRenderer::renderOneSamplePerPixel(uchar4* p_img) {
	dim3 block = dim3(BLOCK_WIDTH, BLOCK_WIDTH, 1);
	dim3 grid = dim3(width/BLOCK_WIDTH, height/BLOCK_WIDTH, 1);
	samplesRendered++;
	renderKernel<<<grid, block, 0>>>(d_settingsData,
			d_imgVectorPtr,
			p_img,
			d_camPtr,
			d_trianglesData,
			d_lightsData,
			d_curandStatePtr,
			samplesRendered);
}

__host__ void ParallelRenderer::copyImageBytes() {
	int pixels = width * height;
	size_t imgBytes = sizeof(uchar4) * pixels;
	CUDA_CHECK_RETURN(cudaMemcpy(h_imgPtr, d_imgBytesPtr, imgBytes, cudaMemcpyDeviceToHost));
	for (unsigned i = 0; i < pixels; i++) {
		gammaCorrectPixel(h_imgPtr[i]);
	}
}

__global__ void initializeCurandKernel(curandState* p_curandState) {
	int idx = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y)
				+ (threadIdx.y * blockDim.x) + threadIdx.x;
	curand_init(1234, idx, 0, &p_curandState[idx]);
}

__global__ void renderKernel(SettingsData settings,
		Vector3Df* p_imgBuffer,
		uchar4* p_outImg,
		Camera* p_camera,
		TrianglesData* p_tris,
		LightsData* p_lights,
		curandState *p_curandState,
		int sampleNumber) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = y * settings.width + x;
	curandState* p_threadCurand = &p_curandState[idx];
	Vector3Df color = samplePixel(x, y, p_camera, p_tris, p_lights, p_threadCurand);
	p_imgBuffer[idx] += color;
	p_outImg[idx] = vector3ToUchar4(p_imgBuffer[idx]/(float)sampleNumber);
}

__device__ Vector3Df samplePixel(int x, int y, Camera* p_camera, TrianglesData* p_trianglesData, LightsData *p_lightsData, curandState *p_curandState) {
    Ray ray = p_camera->computeCameraRay(x, y, p_curandState);

    Vector3Df color(0.f, 0.f, 0.f);
    Vector3Df mask(1.f, 1.f, 1.f);
    RayHit rayHit;
    float t = 0.0f;
    SurfaceInteraction interaction;
    Triangle* p_triangles = p_trianglesData->triPtr;
    Triangle* p_hitTriangle = NULL;
    int numTriangles = p_trianglesData->numTriangles;
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
			float randomNumber = curand_uniform(p_curandState) * ((float)p_lightsData->numLights - 1.0f + 0.9999999f);
			int selectedLightIdx = (int)truncf(randomNumber);
			Triangle* p_light = &p_lightsData->lightsPtr[selectedLightIdx];
			Vector3Df directLighting = estimateDirectLighting(p_light, p_trianglesData, interaction, p_curandState);

			bool clampRadiance = true;
			if (clampRadiance){
				// This introduces bias!!!
				directLighting.x = clamp(directLighting.x, 0.0f, 1.0f);
				directLighting.y = clamp(directLighting.y, 0.0f, 1.0f);
				directLighting.z = clamp(directLighting.z, 0.0f, 1.0f);
			}
			color += mask * directLighting;
		}
        mask *= sampleDiffuseBSDF(&interaction, rayHit, p_curandState) / interaction.pdf;

        ray.origin = interaction.position;
        ray.dir = interaction.inputDirection;
        ray.tMin = EPSILON;
        ray.tMax = FLT_MAX;
    }
    return color;
}

__device__ Vector3Df sampleDiffuseBSDF(SurfaceInteraction* p_interaction, const RayHit& rayHit, curandState* p_curandState) {
	float r1 = 2 * M_PI * curand_uniform(p_curandState);
	float r2 = curand_uniform(p_curandState);
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

__device__ Vector3Df estimateDirectLighting(Triangle* p_light, TrianglesData* p_trianglesData, const SurfaceInteraction &interaction, curandState* p_curandState) {
	Vector3Df directLighting(0.0f, 0.0f, 0.0f);
	if (sameTriangle(interaction.pHitTriangle, p_light)) {
		return directLighting;
	}
	//if specular, return directLighting
	Ray ray(interaction.position,  normalize(p_light->getRandomPointOn(p_curandState) - interaction.position));
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
