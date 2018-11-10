#ifndef CAMERA_H
#define CAMERA_H
#include "geometry.cuh"

#include <cuda_runtime_api.h>

class Camera
{
public:
	__host__ Camera() {}
	__host__ Camera(Vector3Df pos, Vector3Df target, Vector3Df upv, Vector3Df rt, float _fov, int x, int y);

	__host__ void rebase();

	__device__ geom::Ray computeCameraRay(int i, int j) const;

private:
	Vector3Df eye;
	Vector3Df dir;
	Vector3Df up;
	Vector3Df right;
	float fov, aspect;
	int xpixels, ypixels;

//	__device__ Ray computePathtraceRay(int i, int j, curandState *randstate) const
//	{
//		// tent filtered and randomly jittered path tracing ray.
//		float r1 = 2 * curand_uniform(randstate);
//		float dx;
//		if (r1 < 1.f){
//			dx = sqrtf(r1) - 1.f;
//		}
//		else{
//			dx = 1.f - sqrtf(2.f - r1);
//		}
//		float r2 = 2 * curand_uniform(randstate);
//		float dy;
//		if (r2 < 1){
//			dy = sqrtf(r2) - 1.f;
//		}
//		else{
//			dy = 1.f - sqrtf(2.f - r2);
//		}
//		float inv_yres = 1.f / (float)ypixels;
//		float normalized_i = (((float)i + dx) / (float)xpixels) - 0.5f;
//		float normalized_j = (((float)j + dy )/ (float)ypixels) - 0.5f;
//
//		float3 direction = dir;
//		direction += ((-1.0 * right) * t_fov * aspect * normalized_i);
//		direction += (up * t_fov * normalized_j);
//		direction = normalize(direction);
//
//		return Ray(eye, direction);
//	}

};

#endif
