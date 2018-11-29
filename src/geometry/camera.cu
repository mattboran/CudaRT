#include "camera.cuh"
#include <cuda.h>
#include <cuda_runtime_api.h>


using namespace geom;

__host__ Camera::Camera(Vector3Df pos, Vector3Df target, Vector3Df upv, Vector3Df rt, float _fov, int x, int y) :
		eye(pos), dir(target), up(upv), right(rt), fov(tanf(_fov/2.0f * M_PI/180.0f)), xpixels(x), ypixels(y)
{
	aspect = (float)xpixels / (float)ypixels;
	rebase();
}

// Compute tent filtered ray
__device__ Ray Camera::computeCameraRay(int i, int j, curandState* randState) const
{
	float r1 = 2 * curand_uniform(randState);
	float dx;
	if (r1 < 1.f){
		dx = sqrtf(r1) - 1.f;
	}
	else{
		dx = 1.f - sqrtf(2.f - r1);
	}
	float r2 = 2 * curand_uniform(randState);
	float dy;
	if (r2 < 1){
		dy = sqrtf(r2) - 1.f;
	}
	else{
		dy = 1.f - sqrtf(2.f - r2);
	}

	float normalized_i = 1.0f - (((float)i + dx) / (float)xpixels) - 0.5;
	float normalized_j = 1.0f - (((float)j + dy) / (float)ypixels) - 0.5f;

	Vector3Df direction = dir;
	direction += ((right * -1.0f) * fov * aspect * normalized_i);
	direction += (up * fov * normalized_j);
	direction = normalize(direction);

	return Ray(eye, direction);
}

__host__ Ray Camera::computeTestCameraRay(int i, int j) {
	float normalized_i = 1.0f - (((float)i) / (float)xpixels) - 0.5;
	float normalized_j = 1.0f - (((float)j) / (float)ypixels) - 0.5f;

	Vector3Df direction = dir;
	direction += ((right * -1.0f) * fov * aspect * normalized_i);
	direction += (up * fov * normalized_j);
	direction = normalize(direction);

	return Ray(eye, direction);
}

__host__ void Camera::rebase()
{
	up = normalize(up);
	// todo: see if this is needed or correct
	dir = normalize(dir - eye);
	right = normalize(cross(dir,up));
	up = cross(right, dir);
}
