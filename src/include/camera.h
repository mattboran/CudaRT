
#ifndef CAMERA_H
#define CAMERA_H

#include <cuda_runtime.h>
#include "geometry.h"

struct Sampler;
struct Camera {
	__host__ Camera() {}

	__host__ __device__ Ray computeCameraRay(int i, int j, Sampler* p_sampler) const;

	float3 eye;
	float3 dir;
	float3 up;
	float3 right;
	float fov, aspect;
	float apertureWidth = 0.032f;
	float focusDistance;
	int xpixels, ypixels;
};
#endif
