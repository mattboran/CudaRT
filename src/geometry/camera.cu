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

__device__ Ray Camera::computeCameraRay(int i, int j) const
{
	 float normalized_i = ((float)i / (float)xpixels) - 0.5f;
	float normalized_j = ((float)j / (float)ypixels) - 0.5f;

	Vector3Df direction = dir;
//	float Px = (2.0f * ((i + 0.5) / imageWidth) - 1) * fov * aspect;
//	float Py = (1 - 2 * ((y + 0.5) / imageHeight) * tan(fov / 2 * M_PI / 180);
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
