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
#include "pathtrace.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "cuda_error_check.h" // includes cuda.h and cuda_runtime_api.h
using namespace geom;

typedef texture<float4, 1, cudaReadModeElementType> texture_t;

__global__ void renderKernel(Triangle* d_triPtr, int numTriangles, Camera* d_camPtr, Vector3Df* d_imgPtr, int width, int height);
__device__ float intersectTriangles(Triangle* d_triPtr, int numTriangles, RayHit& hitData, const Ray& ray);
__host__ void configureTexture(texture_t &triTexture);
__host__ float4* bindTrianglesToTexture(Triangle* triPtr, unsigned numTris, texture_t &triTexture);
__device__ Triangle getTriangleFromTexture(unsigned i);

__device__ static bool* d_useTextureMemory = NULL;

texture_t triangleTexture;

Vector3Df* pathtraceWrapper(Scene& scene, int width, int height, int samples, bool useTexMemory) {
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
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_triPtr, triangleBytes));
	CUDA_CHECK_RETURN(cudaMemcpy((void*)d_triPtr, (void*)triPtr, triangleBytes, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_imgDataPtr, imageBytes));
	CUDA_CHECK_RETURN(cudaMemcpy((void*)d_imgDataPtr, (void*)imgDataPtr, imageBytes, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_camPtr, sizeof(Camera)));
	CUDA_CHECK_RETURN(cudaMemcpy((void*)d_camPtr, (void*)camPtr, sizeof(Camera), cudaMemcpyHostToDevice));

	// Bind triangles to texture memory
	float4* d_triDataPtr = NULL;
	if (useTexMemory) {
		configureTexture(triangleTexture);
		d_triDataPtr = bindTrianglesToTexture(triPtr, numTris, triangleTexture);
	}
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_useTextureMemory, sizeof(bool)));
	CUDA_CHECK_RETURN(cudaMemset((void*)d_useTextureMemory, (int)useTexMemory, sizeof(bool)));

	// Launch kernel
	dim3 block(blockWidth, blockWidth, 1);
	dim3 grid(width/blockWidth, height/blockWidth, 1);

	for (int s = 0; s < samples; s++)
	{
		renderKernel <<<grid, block>>>(d_triPtr, numTris, d_camPtr, d_imgDataPtr, width, height);
	}

	CUDA_CHECK_RETURN(cudaMemcpy((void*)imgDataPtr, (void*)d_imgDataPtr, imageBytes, cudaMemcpyDeviceToHost));
	cudaFree((void*)d_triPtr);
	cudaFree((void*)d_imgDataPtr);
	cudaFree((void*)d_triDataPtr);
	return imgDataPtr;
}

__global__ void renderKernel(geom::Triangle* d_triPtr, int numTriangles, Camera* d_camPtr, Vector3Df* d_imgPtr, int width, int height) {
	unsigned int i, j;
	i = blockIdx.x*blockDim.x + threadIdx.x;
	j = blockIdx.y*blockDim.y + threadIdx.y;

	Ray camRay = d_camPtr->computeCameraRay(i, j);
	RayHit hitData;
	float t = intersectTriangles(d_triPtr, numTriangles, hitData, camRay);
	Vector3Df light(0.0f, 10.0f, 1.0f);
	if (t < MAX_DISTANCE) {
		Vector3Df hitPt = camRay.pointAlong(t);
		Vector3Df lightDir = normalize(light - hitPt);
		Vector3Df normal = hitData.hitTriPtr->getNormal(hitData);
		d_imgPtr[j * width + i] = Vector3Df(hitData.hitTriPtr->_colorDiffuse * max(dot(lightDir, normal), 0.0f));
	}
}

__device__ float intersectTriangles(geom::Triangle* d_triPtr, int numTriangles, RayHit& hitData, const Ray& ray) {
	float t = MAX_DISTANCE, tprime = MAX_DISTANCE;
	float u, v;
	for (unsigned i = 0; i < numTriangles; i++)
	{
		Triangle tri;
		if (d_useTextureMemory) {
			tri = getTriangleFromTexture(i);
		} else {
			tri = d_triPtr[i];
		}
		tprime = d_triPtr[i].intersect(ray, u, v);
		if (tprime < t && tprime > 0.f)
		{
			t = tprime;
			hitData.hitTriPtr = &d_triPtr[i];
			hitData.u = u;
			hitData.v = v;
		}
	}
	return t;
}

void configureTexture(texture_t &triTexture) {
	triTexture.addressMode[0] = cudaAddressModeBorder;
	triTexture.addressMode[1] = cudaAddressModeBorder;
	triTexture.filterMode = cudaFilterModePoint;
	triTexture.normalized = false;
}

// Note, this function returns the new pointer to the triangle data
__host__ float4* bindTrianglesToTexture(Triangle* triPtr, unsigned numTris, texture_t &triTexture) {
	// 3 float3s for verts, 3 float3s for normals, 3 float3s for material
	unsigned numElements = numTris * 9;
	float4* triDataPtr = new float4[numElements];
	// TODO Replace this with cudaArray
	for (unsigned i = 0; i < numTris; i++) {
		triDataPtr[i*9].x = triPtr->_v1.x;
		triDataPtr[i*9].y = triPtr->_v1.y;
		triDataPtr[i*9].z = triPtr->_v1.z;
		triDataPtr[i*9+1].x = triPtr->_e1.x;
		triDataPtr[i*9+1].y = triPtr->_e1.y;
		triDataPtr[i*9+1].z = triPtr->_e1.z;
		triDataPtr[i*9+2].x = triPtr->_e2.x;
		triDataPtr[i*9+2].y = triPtr->_e2.y;
		triDataPtr[i*9+2].z = triPtr->_e2.z;
		triDataPtr[i*9+3].x = triPtr->_n1.x;
		triDataPtr[i*9+3].y = triPtr->_n1.y;
		triDataPtr[i*9+3].z = triPtr->_n1.z;
		triDataPtr[i*9+4].x = triPtr->_n2.x;
		triDataPtr[i*9+4].y = triPtr->_n2.y;
		triDataPtr[i*9+4].z = triPtr->_n2.z;
		triDataPtr[i*9+5].x = triPtr->_n3.x;
		triDataPtr[i*9+5].y = triPtr->_n3.y;
		triDataPtr[i*9+5].z = triPtr->_n3.z;
		triDataPtr[i*9+6].x = triPtr->_colorDiffuse.x;
		triDataPtr[i*9+6].y = triPtr->_colorDiffuse.y;
		triDataPtr[i*9+6].z = triPtr->_colorDiffuse.z;
		triDataPtr[i*9+7].x = triPtr->_colorSpec.x;
		triDataPtr[i*9+7].y = triPtr->_colorSpec.y;
		triDataPtr[i*9+7].z = triPtr->_colorSpec.z;
		triDataPtr[i*9+8].x = triPtr->_colorEmit.x;
		triDataPtr[i*9+8].y = triPtr->_colorEmit.y;
		triDataPtr[i*9+8].z = triPtr->_colorEmit.z;
	}

	float4* d_triDataPtr = NULL;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_triDataPtr, sizeof(float) * numElements));
	CUDA_CHECK_RETURN(cudaMemcpy((void*)d_triDataPtr, triDataPtr, sizeof(float) * numElements, cudaMemcpyHostToDevice));

	size_t offset;
	cudaBindTexture(&offset, triTexture, d_triDataPtr, sizeof(float) * numElements);
	return d_triDataPtr;
}

__device__ Triangle getTriangleFromTexture(unsigned i) {
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
