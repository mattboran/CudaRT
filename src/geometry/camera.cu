#include "camera.h"
#include "renderer.h"

#include <fstream>
#include <iostream>
#include "picojson.h"
#include <stdlib.h>
#include <streambuf>

using namespace std;

__host__ Camera::Camera(Vector3Df pos, Vector3Df target, Vector3Df upv, float _fov, int x, int y) :
		eye(pos), dir(target), up(upv), fov(tanf(_fov/2.0f * M_PI/180.0f)), xpixels(x), ypixels(y)
{
	aspect = (float)xpixels / (float)ypixels;
	up = normalize(up);
	focusDistance = (dir-eye).length();
	dir = normalize(dir - eye);
	right = normalize(cross(dir,up));
}


__host__ __device__ void tentFilter(float &i, float &j, Sampler* p_sampler) {
	float r1, r2;
	r1 = 2.f * p_sampler->getNextFloat();
	r2 = 2.f * p_sampler->getNextFloat();
	if (r1 < 1.f){
		i = sqrtf(r1) - 1.f;
	}
	else{
		i = 1.f - sqrtf(2.f - r1);
	}
	if (r2 < 1){
		j = sqrtf(r2) - 1.f;
	}
	else{
		j = 1.f - sqrtf(2.f - r2);
	}
}

// Compute tent filtered ray
__host__ __device__ Ray Camera::computeCameraRay(int i, int j, Sampler* p_sampler) const {
	float dx, dy;
	tentFilter(dx, dy, p_sampler);

	float normalized_i = (((float)i + dx) / (float)xpixels) - 0.5;
	float normalized_j = 0.5f - (((float)j + dy) / (float)ypixels);

	Vector3Df direction = dir;
	float rightJitter = 1.0f;
	float upJitter = 1.0f;
	Vector3Df origin = eye;
	direction += (right * fov * aspect * normalized_i);
	direction += (up * fov * normalized_j);

	if (apertureWidth > 0.0f) {
		float r1 = p_sampler->getNextFloat() - 0.5f * 2 * M_PI;
		float r2 = p_sampler->getNextFloat() - 0.5f;
		rightJitter = cosf(r1) * r2 * apertureWidth;
		upJitter = sinf(r1)* r2 * apertureWidth;
		origin += ((right * rightJitter) + (up * upJitter));
		Vector3Df focalPoint = eye + direction * focusDistance;
		direction = focalPoint - origin;
	}

	return Ray(origin, normalize(direction));
}
