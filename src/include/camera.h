#ifndef CAMERA_H
#define CAMERA_H

#include <cuda_runtime.h>
#include "geometry.h"

struct Sampler;
class Camera {
public:
	__host__ Camera() {}
	__host__ Camera(std::string filename, int width, int height);
	__host__ Camera(Vector3Df pos, Vector3Df target, Vector3Df upv, float _fov, int x, int y);

	__host__ void rebase();

	__host__ __device__ Ray computeCameraRay(int i, int j, Sampler* p_sampler) const;
protected:
	Vector3Df eye;
	Vector3Df dir;
	Vector3Df up;
	Vector3Df right;
	float fov, aspect;
	int xpixels, ypixels;
};
#endif
