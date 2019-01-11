#ifndef CAMERA_H
#define CAMERA_H

#include "geometry.h"

#include <curand.h>

struct Sampler;
class Camera
{
public:
	__host__ Camera() {}
	__host__ Camera(Vector3Df pos, Vector3Df target, Vector3Df upv, Vector3Df rt, float _fov, int x, int y);

	__host__ void rebase();

	__host__ __device__ geom::Ray computeCameraRay(int i, int j, Sampler* p_sampler) const;
private:
	Vector3Df eye;
	Vector3Df dir;
	Vector3Df up;
	Vector3Df right;
	float fov, aspect;
	int xpixels, ypixels;
};

#endif
