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
	int numStreams;
};

__global__ void debugRenderKernel(Triangle* d_triPtr, int numTriangles,
		Camera* d_camPtr, Vector3Df* d_imgPtr, int width, int height,
		bool useTexMem);
__global__ void setupCurandKernel(curandState *randState, int streamOffset);
__global__ void renderKernel(TrianglesData* d_tris, Camera* d_camPtr, Vector3Df* d_imgPtr, LightsData* d_lights, SettingsData* d_settings, curandState *randState, int streamId);
__global__ void averageSamplesKernel(Vector3Df* d_streamImgDataPtr, Vector3Df* d_imgPtr, SettingsData* d_settings);
__device__ float intersectTriangles(Triangle* d_triPtr, int numTriangles, RayHit& hitData, const Ray& ray, bool useTexMem);
__device__ inline Triangle getTriangleFromTexture(unsigned i);


texture_t triangleTexture;

Vector3Df* pathtraceWrapper(Scene& scene, int width, int height, int samples, int numStreams, bool &useTexMemory) {
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

	// Lights -> d_lights
	Triangle* lightsPtr = scene.getLightsPtr();
	Triangle* d_lightTrianglePtr = NULL;
	size_t lightTrianglesBytes = sizeof(Triangle) * scene.getNumLights();
	CUDA_CHECK_RETURN(cudaMalloc((void** )&d_lightTrianglePtr, lightTrianglesBytes));
	CUDA_CHECK_RETURN(cudaMemcpy((void* )d_lightTrianglePtr, (void *)lightsPtr, lightTrianglesBytes, cudaMemcpyHostToDevice));

	LightsData* h_lights = (LightsData*)malloc(sizeof(LightsData) + lightTrianglesBytes);
	LightsData* d_lights = NULL;
	h_lights->lightsPtr = d_lightTrianglePtr;
	h_lights->numLights = scene.getNumLights();
	h_lights->totalSurfaceArea = scene.getLightsSurfaceArea();
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_lights, sizeof(LightsData) + lightTrianglesBytes));
	CUDA_CHECK_RETURN(cudaMemcpy((void* )d_lights, h_lights, sizeof(LightsData) + lightTrianglesBytes, cudaMemcpyHostToDevice));

	// Camera
	Camera* camPtr = scene.getCameraPtr();
	Camera* d_camPtr = NULL;
	CUDA_CHECK_RETURN(cudaMalloc((void** )&d_camPtr, sizeof(Camera)));
	CUDA_CHECK_RETURN(cudaMemcpy((void* )d_camPtr, (void* )camPtr, sizeof(Camera), cudaMemcpyHostToDevice));

	// Setup settings -> d_settings
	SettingsData h_settings;
	SettingsData* d_settings;
	h_settings.width = width;
	h_settings.height = height;
	h_settings.samples = samples;
	h_settings.useTexMem = useTexMemory;
	h_settings.numStreams = numStreams;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_settings, sizeof(SettingsData)));
	CUDA_CHECK_RETURN(cudaMemcpy((void*)d_settings, &h_settings, sizeof(SettingsData), cudaMemcpyHostToDevice));

	// Launch kernels
	const unsigned int threadsPerBlock = blockWidth * blockWidth;
	const unsigned int gridBlocks = width / blockWidth * height / blockWidth;
	dim3 block(blockWidth, blockWidth, 1);
	dim3 grid(width / blockWidth, height / blockWidth, 1);

	cudaStream_t streams[numStreams];

	// Image
	Vector3Df* imgDataPtr = new Vector3Df[pixels]();
	Vector3Df* d_imgDataPtr = NULL;
	Vector3Df* d_streamImgDataPtr;
	CUDA_CHECK_RETURN(cudaMalloc((void** )&d_imgDataPtr, imageBytes));
	CUDA_CHECK_RETURN(cudaMemcpy((void* )d_imgDataPtr, (void* )imgDataPtr, imageBytes, cudaMemcpyHostToDevice));

	// Setup cuRand kernel and data in streams
	curandState* d_curandState;
	int imagePixels = width * height;
	int curandStateSize = threadsPerBlock * gridBlocks;
	size_t curandStateBytes = sizeof(curandState) * curandStateSize;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_streamImgDataPtr, imageBytes * numStreams));
	CUDA_CHECK_RETURN(cudaMalloc((void** )&d_curandState, curandStateBytes * numStreams));
	for (int s = 0; s < numStreams; s++) {
		cudaStreamCreate(&streams[s]);
		curandState* d_curandStatePtr = &d_curandState[s * curandStateSize];
		setupCurandKernel<<<grid, block, 0, streams[s]>>>(d_curandStatePtr, s);
		CUDA_CHECK_RETURN(cudaMemcpy((void* )&d_streamImgDataPtr[s * imagePixels], (void* )imgDataPtr, imageBytes, cudaMemcpyHostToDevice));
	}

	for (int s = 0; s < samples; s++) {
		int streamId = s % numStreams;
		Vector3Df* streamImgData = &d_streamImgDataPtr[streamId * imagePixels];
		curandState* d_curandStatePtr = &d_curandState[curandStateSize * streamId];
		renderKernel<<<grid, block, 0, streams[streamId]>>>(d_tris, d_camPtr, streamImgData, d_lights, d_settings, d_curandStatePtr, streamId);
	}

	// Combine the different streams into a single image
	averageSamplesKernel<<<grid, block>>>(d_streamImgDataPtr, d_imgDataPtr, d_settings);

	CUDA_CHECK_RETURN(cudaMemcpy((void* )imgDataPtr, (void* )d_imgDataPtr, imageBytes, cudaMemcpyDeviceToHost));

	// Clean up the streams
	for (int s = 0; s < numStreams; s++) {
		cudaStreamDestroy(streams[s]);
	}

	// Clean up host memory
	free(h_lights);
	free(h_tris);
	// Clean up device memory
	cudaFree((void*) d_tris);
	cudaFree((void*) d_triPtr);
	cudaFree((void*) d_lights);
	cudaFree((void*) d_lightTrianglePtr);
	cudaFree((void*) d_settings);
	cudaFree((void*) d_curandState);
	cudaFree((void*) d_imgDataPtr);
	cudaFree((void*) d_streamImgDataPtr);
	if (useTexMemory)
		cudaFreeArray(d_triDataArray);
	return imgDataPtr;
}

__global__ void renderKernel(TrianglesData* d_tris,
							Camera* d_camPtr,
							Vector3Df* d_imgPtr,
							LightsData* d_lights,
							SettingsData* d_settings,
							curandState *randState,
							int streamId) {
	int idx = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y)
			+ (threadIdx.y * blockDim.x) + threadIdx.x;
	unsigned int i, j;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;

	Ray ray = d_camPtr->computeCameraRay(i, j, &randState[idx]);
	RayHit hitData, lightHitData;
	Triangle* hitTriPtr;
	Vector3Df hitPt, nextDir, normal, colorAtPixel;
	Vector3Df mask(1.0f, 1.0f, 1.0f);

	// First see if the camera ray hits anything. If not, return black.
	float t = intersectTriangles(d_tris->triPtr, d_tris->numTriangles, hitData, ray, d_settings->useTexMem);
	if (t < MAX_DISTANCE) {
		hitPt = ray.pointAlong(t);
		hitTriPtr = hitData.hitTriPtr;
		normal = hitTriPtr->getNormal(hitData);
		// if we hit a light directly, add its contribution here so as not to double dip in the BSDF calculations below
		if (hitTriPtr->isEmissive()) {
			d_imgPtr[j * d_settings->width + i] += hitTriPtr->_colorEmit;
			return;
		}
	} else {
		d_imgPtr[j * d_settings->width + i] += Vector3Df(0.0f, 0.0f, 0.0f);
		return;
	}


	for (unsigned bounces = 0; bounces < 4; bounces++) {
		// DIFFUSE BSDF:

		// Direct lighting: select light at random, test for intersection, add contribution
		// Get a new ray going towards a random point on the selected light
		float randomNumber = curand_uniform(&randState[idx]);
		randomNumber *= (float)d_lights->numLights - 1.0f + 0.9999999f;
		int selectedLightIndex = (int)truncf(randomNumber);
		Triangle selectedLight = d_lights->lightsPtr[selectedLightIndex];
		Vector3Df lightRayDir = normalize(selectedLight.getRandomPointOn(&randState[idx]) - hitPt);

		Ray lightRay(hitPt + normal * EPSILON, lightRayDir);
		t = intersectTriangles(d_tris->triPtr, d_tris->numTriangles, lightHitData, lightRay, d_settings->useTexMem);
		if (t < MAX_DISTANCE){
			// See if we've hit the light we tested for
			Triangle* lightRayHitPtr = lightHitData.hitTriPtr;
			if (lightRayHitPtr->_triId == selectedLight._triId) {
				float surfaceArea = selectedLight._surfaceArea;
				float distanceSquared = t*t; // scale by factor of 10
				float incidenceAngle = fabs(dot(selectedLight.getNormal(lightHitData), -lightRayDir));
				float weightFactor = surfaceArea/distanceSquared * incidenceAngle;
				colorAtPixel += mask * selectedLight._colorEmit * hitData.hitTriPtr->_colorDiffuse * weightFactor;
			}
		}

		// Now compute indirect lighting
		t = intersectTriangles(d_tris->triPtr, d_tris->numTriangles, hitData, ray, d_settings->useTexMem);
		if (t < MAX_DISTANCE) {

			Vector3Df hitPt = ray.pointAlong(t);
			Triangle* hitTriPtr = hitData.hitTriPtr;
			Vector3Df normal = hitTriPtr->getNormal(hitData);

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

__global__ void averageSamplesKernel(Vector3Df* d_streamImgDataPtr, Vector3Df* d_imgPtr, SettingsData* d_settings) {
	int idx = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y)
			+ (threadIdx.y * blockDim.x) + threadIdx.x;

	Vector3Df pixel(0.0f, 0.0f, 0.0f);
	int pixelsInImage = d_settings->width*d_settings->height;
	for (int s = 0; s < d_settings->numStreams; s++) {
		pixel += d_streamImgDataPtr[idx + s*pixelsInImage];
	}

	pixel *= (1.0f / (float) d_settings->samples);
	d_imgPtr[idx].x = fminf(pixel.x, 1.0f);
	d_imgPtr[idx].y = fminf(pixel.y, 1.0f);
	d_imgPtr[idx].z = fminf(pixel.z, 1.0f);
}

__global__ void setupCurandKernel(curandState *randState, int streamOffset) {
	int idx = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y)
			+ (threadIdx.y * blockDim.x) + threadIdx.x;
	curand_init(1234 + streamOffset, idx, 0, &randState[idx]);
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
