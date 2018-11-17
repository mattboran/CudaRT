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
// Random  number generation with CUDA

using namespace geom;

__global__ void debugRenderKernel(Triangle* d_triPtr, int numTriangles,
		Camera* d_camPtr, Vector3Df* d_imgPtr, int width, int height,
		bool useTexMem);
__global__ void setupCurandKernel(curandState *randState);
__global__ void renderKernel(Triangle* d_triPtr, int numTriangles,
		Camera* d_camPtr, Vector3Df* d_imgPtr, int width, int height,
		bool useTexMem, curandState *randState);
__global__ void averageSamplesKernel(Vector3Df* d_imgPtr, int width, int height,
		unsigned samples);
__device__ float intersectTriangles(Triangle* d_triPtr, int numTriangles,
		RayHit& hitData, const Ray& ray, bool useTexMem);
__device__ inline Triangle getTriangleFromTexture(unsigned i);

texture_t triangleTexture;

Vector3Df* pathtraceWrapper(Scene& scene, int width, int height, int samples,
		bool &useTexMemory) {
	int pixels = width * height;
	unsigned numTris = scene.getNumTriangles();
	size_t triangleBytes = sizeof(Triangle) * numTris;
	size_t imageBytes = sizeof(Vector3Df) * width * height;

	// Initialize CUDA memory
	Triangle* triPtr = scene.getTriPtr();
	Triangle* d_triPtr = NULL;
	Vector3Df* imgDataPtr = new Vector3Df[pixels];
	Vector3Df* d_imgDataPtr = NULL;
	Camera* camPtr = scene.getCameraPtr();
	Camera* d_camPtr = NULL;
	CUDA_CHECK_RETURN(cudaMalloc((void** )&d_triPtr, triangleBytes));
	CUDA_CHECK_RETURN(
			cudaMemcpy((void* )d_triPtr, (void* )triPtr, triangleBytes,
					cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMalloc((void** )&d_imgDataPtr, imageBytes));
	CUDA_CHECK_RETURN(
			cudaMemcpy((void* )d_imgDataPtr, (void* )imgDataPtr, imageBytes,
					cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMalloc((void** )&d_camPtr, sizeof(Camera)));
	CUDA_CHECK_RETURN(
			cudaMemcpy((void* )d_camPtr, (void* )camPtr, sizeof(Camera),
					cudaMemcpyHostToDevice));

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
		d_triDataArray = bindTrianglesToTexture(triPtr, numTris,
				triangleTexture);
	}

	// Launch kernels
	const unsigned int threadsPerBlock = blockWidth * blockWidth;
	const unsigned int gridBlocks = width / blockWidth * height / blockWidth;
	dim3 block(blockWidth, blockWidth, 1);
	dim3 grid(width / blockWidth, height / blockWidth, 1);

	// Setup cuRand
	curandState* d_curandState;
	CUDA_CHECK_RETURN(
			cudaMalloc((void** )&d_curandState,
					threadsPerBlock * gridBlocks * sizeof(curandState)));
	setupCurandKernel<<<grid, block>>>(d_curandState);

	for (int s = 0; s < samples; s++) {
		renderKernel<<<grid, block>>>(d_triPtr, numTris, d_camPtr, d_imgDataPtr,
				width, height, useTexMemory, d_curandState);
	}

	averageSamplesKernel<<<grid, block>>>(d_imgDataPtr, width, height, samples);

	CUDA_CHECK_RETURN(
			cudaMemcpy((void* )imgDataPtr, (void* )d_imgDataPtr, imageBytes,
					cudaMemcpyDeviceToHost));
	cudaFree((void*) d_triPtr);
	cudaFree((void*) d_imgDataPtr);
	cudaFree((void*) d_curandState);
	if (useTexMemory)
		cudaFreeArray(d_triDataArray);
	return imgDataPtr;
}

__global__ void renderKernel(Triangle* d_triPtr, int numTriangles,
		Camera* d_camPtr, Vector3Df* d_imgPtr, int width, int height,
		bool useTexMemory, curandState *randState) {
	int idx = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y)
			+ (threadIdx.y * blockDim.x) + threadIdx.x;
	unsigned int i, j;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;

	Ray ray = d_camPtr->computeCameraRay(i, j, &randState[idx]);
	RayHit hitData;
	Vector3Df colorAtPixel;
	Vector3Df nextDir;
	Vector3Df mask(1.0f, 1.0f, 1.0f);

	for (unsigned bounces = 0; bounces < 6; bounces++) {
		float t = intersectTriangles(d_triPtr, numTriangles, hitData, ray,
				useTexMemory);
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
				Vector3Df u = normalize(
						cross(
								(fabs(w.x) > 0.1f ?
										make_float3(0.f, 1.f, 0.f) :
										make_float3(1.f, 0.f, 0.f)), w));
				Vector3Df v = cross(w, u);

				// Random point on unit hemisphere @ hit_point and centered at normal
				nextDir = normalize(
						u * cosf(r1) * r2sq + v * sinf(r1) * r2sq
								+ w * sqrtf(1.f - r2));
				// Division by 1/2 for this PDF weighted by cosine
				mask *= hitTriPtr->_colorDiffuse * dot(nextDir, normal) * 2.f;
				// Shift hitpoint outward by an epsilon
				hitPt += normal * EPSILON;
			}
			ray = Ray(hitPt, nextDir);
		}
	}
	d_imgPtr[j * width + i] += colorAtPixel;
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

__global__ void averageSamplesKernel(Vector3Df* d_imgPtr, int width, int height,
		unsigned samples) {
	int idx = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y)
			+ (threadIdx.y * blockDim.x) + threadIdx.x;
	d_imgPtr[idx] *= 1.0f / (float) samples;
	d_imgPtr[idx].x = fminf(d_imgPtr[idx].x, 1.0f);
	d_imgPtr[idx].y = fminf(d_imgPtr[idx].y, 1.0f);
	d_imgPtr[idx].z = fminf(d_imgPtr[idx].z, 1.0f);
}

__global__ void setupCurandKernel(curandState *randState) {
	int idx = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y)
			+ (threadIdx.y * blockDim.x) + threadIdx.x;
	curand_init(1234, idx, 0, &randState[idx]);
}

__device__ float intersectTriangles(geom::Triangle* d_triPtr, int numTriangles,
		RayHit& hitData, const Ray& ray, bool useTexMemory) {
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

