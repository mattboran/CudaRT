/*
 ============================================================================
 Name        : pathtrace.cu
 Author      : Tudor Matei Boran
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA raytracer
 ============================================================================
 */
#include "camera.cuh"
#include "cuda_textures.cuh"
#include "pathtrace.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <curand_kernel.h>

#include <iostream>

#include "cuda_error_check.h" // includes cuda.h and cuda_runtime_api.h

using namespace geom;

struct LightsData {
	Triangle* lightsPtr;
	unsigned numLights;
	float totalSurfaceArea;
};

struct TrianglesData {
	Triangle* triPtr;
	unsigned numTriangles;
};

// TODO: Move image, camera, and curandState pointers into here
struct SettingsData {
	int width;
	int height;
	int samples;
	bool useTexMem;
};

__global__ void debugRenderKernel(Triangle* d_triPtr, int numTriangles,
		Camera* d_camPtr, Vector3Df* d_imgPtr, int width, int height,
		bool useTexMem);
__global__ void setupCurandKernel(curandState *randState);
__global__ void renderKernel(TrianglesData* d_tris, Camera* d_camPtr, Vector3Df* d_imgPtr, LightsData* d_lights, SettingsData* d_settings, curandState *randState);
__global__ void averageSamplesKernel(Vector3Df* d_imgPtr, SettingsData* d_settings);
__device__ float intersectTriangles(Triangle* d_triPtr, int numTriangles, RayHit& hitData, const Ray& ray, bool useTexMem);
__device__ inline Triangle getTriangleFromTexture(unsigned i);


texture_t triangleTexture;

Vector3Df* pathtraceWrapper(Scene& scene, int width, int height, int samples, bool &useTexMemory) {
	int pixels = width * height;
	unsigned numTris = scene.getNumTriangles();
	size_t triangleBytes = sizeof(Triangle) * numTris;
	size_t imageBytes = sizeof(Vector3Df) * width * height;

	// Initialize CUDA memory

	// Triangles -> d_tris
	Triangle* h_triPtr = scene.getTriPtr();
	Triangle* d_triPtr = NULL;
	CUDA_CHECK_RETURN(cudaMalloc((void** )&d_triPtr, triangleBytes));
	CUDA_CHECK_RETURN(cudaMemcpy((void* )d_triPtr, (void* )h_triPtr, triangleBytes, cudaMemcpyHostToDevice));

	TrianglesData* h_tris = (TrianglesData*)malloc(sizeof(TrianglesData) + triangleBytes);
	TrianglesData* d_tris = NULL;
	h_tris->numTriangles = numTris;
	h_tris->triPtr = d_triPtr;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_tris, sizeof(TrianglesData) + triangleBytes));
	CUDA_CHECK_RETURN(cudaMemcpy((void*)d_tris, (void*)h_tris, sizeof(TrianglesData) + triangleBytes, cudaMemcpyHostToDevice));

	// Lights -> d_lights
	Triangle* lightsPtr = scene.getLightsPtr();
	Triangle* d_lightTrianglePtr = NULL;
	size_t lightTrianglesBytes = sizeof(Triangle) * scene.getNumLights();
	CUDA_CHECK_RETURN(cudaMalloc((void** )&d_lightTrianglePtr, lightTrianglesBytes));
	CUDA_CHECK_RETURN(cudaMemcpy((void* )d_lightTrianglePtr, (void *)lightsPtr, lightTrianglesBytes, cudaMemcpyHostToDevice));

	LightsData* h_lights = (LightsData*)malloc(sizeof(LightsData));
	LightsData* d_lights = NULL;
	h_lights->lightsPtr = d_lightTrianglePtr;
	h_lights->numLights = scene.getNumLights();
	h_lights->totalSurfaceArea = scene.getLightsSurfaceArea();
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_lights, sizeof(LightsData) + lightTrianglesBytes));
	CUDA_CHECK_RETURN(cudaMemcpy((void* )d_lights, h_lights, sizeof(LightsData) + lightTrianglesBytes, cudaMemcpyHostToDevice));

	// Setup settings -> d_settings
	SettingsData h_settings;
	SettingsData* d_settings;
	h_settings.width = width;
	h_settings.height = height;
	h_settings.samples = samples;
	h_settings.useTexMem = useTexMemory;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_settings, sizeof(SettingsData)));
	CUDA_CHECK_RETURN(cudaMemcpy((void*)d_settings, &h_settings, sizeof(SettingsData), cudaMemcpyHostToDevice));

	// Image
	Vector3Df* imgDataPtr = new Vector3Df[pixels]();
	Vector3Df* d_imgDataPtr = NULL;
	CUDA_CHECK_RETURN(cudaMalloc((void** )&d_imgDataPtr, imageBytes));
	CUDA_CHECK_RETURN(cudaMemcpy((void* )d_imgDataPtr, (void* )imgDataPtr, imageBytes, cudaMemcpyHostToDevice));

	// Camera
	Camera* camPtr = scene.getCameraPtr();
	Camera* d_camPtr = NULL;
	CUDA_CHECK_RETURN(cudaMalloc((void** )&d_camPtr, sizeof(Camera)));
	CUDA_CHECK_RETURN(cudaMemcpy((void* )d_camPtr, (void* )camPtr, sizeof(Camera), cudaMemcpyHostToDevice));

	// Bind triangles to texture memory -- texture memory doesn't quite work
	cudaArray* d_triDataArray = NULL;
	if (useTexMemory && numTris > TEX_ARRAY_MAX) {
		std::cout << "Not using texture memory because we cannot fit "
				<< numTris << " triangles in 1D cudaArray" << std::endl;
		useTexMemory = false;
	}
	if (useTexMemory) {
		std::cout << "Using texture memory!" << std::endl;
		configureTexture(triangleTexture);
		d_triDataArray = bindTrianglesToTexture(h_triPtr, numTris,
				triangleTexture);
	}

	// Launch kernels
	const unsigned int threadsPerBlock = blockWidth * blockWidth;
	const unsigned int gridBlocks = width / blockWidth * height / blockWidth;
	dim3 block(blockWidth, blockWidth, 1);
	dim3 grid(width / blockWidth, height / blockWidth, 1);

	// Setup cuRand kernel
	curandState* d_curandState;
	CUDA_CHECK_RETURN(cudaMalloc((void** )&d_curandState, threadsPerBlock * gridBlocks * sizeof(curandState)));
	setupCurandKernel<<<grid, block>>>(d_curandState);

	for (int s = 0; s < samples; s++) {
		renderKernel<<<grid, block>>>(d_tris, d_camPtr, d_imgDataPtr, d_lights, d_settings, d_curandState);
	}

	averageSamplesKernel<<<grid, block>>>(d_imgDataPtr, d_settings);

	CUDA_CHECK_RETURN(cudaMemcpy((void* )imgDataPtr, (void* )d_imgDataPtr, imageBytes, cudaMemcpyDeviceToHost));

	free(h_lights);
	free(h_tris);
	cudaFree((void*) d_lightTrianglePtr);
	cudaFree((void*) d_triPtr);
	cudaFree((void*) d_tris);
	cudaFree((void*) d_settings);
	cudaFree((void*) d_imgDataPtr);
	cudaFree((void*) d_curandState);
	if (useTexMemory)
		cudaFreeArray(d_triDataArray);
	return imgDataPtr;
}

__global__ void renderKernel(TrianglesData* d_tris,
							Camera* d_camPtr,
							Vector3Df* d_imgPtr,
							LightsData* d_lights,
							SettingsData* d_settings,
							curandState *randState) {
	int idx = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y)
			+ (threadIdx.y * blockDim.x) + threadIdx.x;
	unsigned int i, j;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;

	Ray ray = d_camPtr->computeCameraRay(i, j, &randState[idx]);
	RayHit hitData, lightHitData;
	Triangle* hitTriPtr;
	Vector3Df hitPt, normal, nextDir, colorAtPixel;
	Vector3Df mask(1.0f, 1.0f, 1.0f);

	// First see if the camera ray hits anything. If not, return black.
	float t = intersectTriangles(d_tris->triPtr, d_tris->numTriangles, hitData, ray, d_settings->useTexMem);
	if (t < MAX_DISTANCE) {
		hitPt = ray.pointAlong(t);
		hitTriPtr = hitData.hitTriPtr;
		normal = hitTriPtr->getNormal(hitData);
	} else {
		d_imgPtr[j * d_settings->width + i] += Vector3Df(0.0f, 0.0f, 0.0f);
		return;
	}

	// Direct lighting: select light at random, test for intersection, add contribution
	// Get a new ray going towards a random point on the selected light
	float randomNumber = curand_uniform(&randState[idx]);
	randomNumber *= (float)d_lights->numLights - 1.0f + 0.9999999f;
	int selectedLightIndex = (int)truncf(randomNumber);
	Triangle selectedLight = d_lights->lightsPtr[selectedLightIndex];

	Vector3Df lightRayDir = selectedLight.getPointOn(&randState[idx]) - hitPt;
	Ray lightRay(hitPt + normal * 0.01f, lightRayDir);
	t = intersectTriangles(d_tris->triPtr, d_tris->numTriangles, lightHitData, lightRay, d_settings->useTexMem);
	if (t > lightRayDir.length())  {
		float distanceFactor = 1.0f;
		float numLightsFactor = 1.0f/(float)d_lights->numLights;
		colorAtPixel = selectedLight._colorEmit * hitData.hitTriPtr->_colorDiffuse * distanceFactor * numLightsFactor;
	}

	for (unsigned bounces = 0; bounces < 4; bounces++) {
		t = intersectTriangles(d_tris->triPtr, d_tris->numTriangles, hitData, ray, d_settings->useTexMem);
		if (t < MAX_DISTANCE) {

			Vector3Df hitPt = ray.pointAlong(t);
			Triangle* hitTriPtr = hitData.hitTriPtr;
			Vector3Df normal = hitTriPtr->getNormal(hitData);

			colorAtPixel += mask * hitTriPtr->_colorEmit;

			if (hitTriPtr->isDiffuse()) {
				float r1 = 2 * M_PI * curand_uniform(&randState[idx]);
				float r2 = curand_uniform(&randState[idx]);
				float r2sq = sqrtf(r2);

				// calculate orthonormal coordinates u, v, w, at hitpt
				Vector3Df w = normal;
				Vector3Df u = normalize(cross( (fabs(w.x) > 0.1f ?
							Vector3Df(0.f, 1.f, 0.f) :
							Vector3Df(1.f, 0.f, 0.f)), w));
				Vector3Df v = cross(w, u);

				// Random point on unit hemisphere @ hit_point and centered at normal
				nextDir = normalize(u * cosf(r1) * r2sq + v * sinf(r1) * r2sq + w * sqrtf(1.f - r2));
				// Division by 1/2 for this PDF weighted by cosine
				mask *= hitTriPtr->_colorDiffuse * dot(nextDir, normal) * 2.f;
				// Shift hitpoint outward by an epsilon
				hitPt += normal * EPSILON;
			}
			ray = Ray(hitPt, nextDir);
		}
	}
	d_imgPtr[j * d_settings->width + i] += colorAtPixel;
}

__global__ void debugRenderKernel(geom::Triangle* d_triPtr, int numTriangles,
		Camera* d_camPtr, Vector3Df* d_imgPtr, int width, int height,
		bool useTexMemory) {
	unsigned int i, j;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;
	curandState d_curandState;
	curand_init(1234, i * j + i, 0, &d_curandState);

	Ray camRay = d_camPtr->computeCameraRay(i, j, &d_curandState);
	RayHit hitData;
	float t = intersectTriangles(d_triPtr, numTriangles, hitData, camRay,
			useTexMemory);
	Vector3Df light(0.0f, 10.0f, 1.0f);
	if (t < MAX_DISTANCE) {
		Vector3Df hitPt = camRay.pointAlong(t);
		Vector3Df lightDir = normalize(light - hitPt);
		Vector3Df normal = hitData.hitTriPtr->getNormal(hitData);
		d_imgPtr[j * width + i] = Vector3Df(
				hitData.hitTriPtr->_colorDiffuse
						* max(dot(lightDir, normal), 0.0f));
	}
}

__global__ void averageSamplesKernel(Vector3Df* d_imgPtr, SettingsData* d_settings) {
	int idx = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y)
			+ (threadIdx.y * blockDim.x) + threadIdx.x;
	d_imgPtr[idx] *= 1.0f / (float) d_settings->samples;
	d_imgPtr[idx].x = fminf(d_imgPtr[idx].x, 1.0f);
	d_imgPtr[idx].y = fminf(d_imgPtr[idx].y, 1.0f);
	d_imgPtr[idx].z = fminf(d_imgPtr[idx].z, 1.0f);
}

__global__ void setupCurandKernel(curandState *randState) {
	int idx = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y)
			+ (threadIdx.y * blockDim.x) + threadIdx.x;
	curand_init(1234, idx, 0, &randState[idx]);
}

__device__ float intersectTriangles(geom::Triangle* d_triPtr,
									int numTriangles,
									RayHit& hitData,
									const Ray& ray,
									bool useTexMemory) {
	float t = MAX_DISTANCE, tprime = MAX_DISTANCE;
	float u, v;
	for (unsigned i = 0; i < numTriangles; i++) {
		Triangle tri;
		if (useTexMemory) {
			Triangle tri = getTriangleFromTexture(i);
			tprime = tri.intersect(ray, u, v);
		} else {
			tprime = d_triPtr[i].intersect(ray, u, v);
		}
		if (tprime < t && tprime > 0.f) {
			t = tprime;
			hitData.hitTriPtr = &d_triPtr[i];
			hitData.u = u;
			hitData.v = v;
		}
	}
	return t;
}

__device__ inline Triangle getTriangleFromTexture(unsigned i) {
	float4 v1, e1, e2, n1, n2, n3, diff, spec, emit;
	v1 = tex1Dfetch(triangleTexture, i * 9);
	e1 = tex1Dfetch(triangleTexture, i * 9 + 1);
	e2 = tex1Dfetch(triangleTexture, i * 9 + 2);
	n1 = tex1Dfetch(triangleTexture, i * 9 + 3);
	n2 = tex1Dfetch(triangleTexture, i * 9 + 4);
	n3 = tex1Dfetch(triangleTexture, i * 9 + 5);
	diff = tex1Dfetch(triangleTexture, i * 9 + 6);
	spec = tex1Dfetch(triangleTexture, i * 9 + 7);
	emit = tex1Dfetch(triangleTexture, i * 9 + 8);
	return Triangle(v1, e1, e2, n1, n2, n3, diff, spec, emit);
}

