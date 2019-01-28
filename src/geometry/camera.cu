#include "camera.h"
#include "renderer.h"

#include <fstream>
#include <iostream>
#include "picojson.h"
#include <stdlib.h>
#include <streambuf>

using namespace std;

__host__ Vector3Df vectorFromArray(picojson::array arr) {
	Vector3Df retVal;
	int idx = 0;
	for (picojson::array::iterator it = arr.begin(); it != arr.end(); it++)
	{
		float val = it->get<double>();
		retVal._v[idx] = val;
		idx++;
	}
	return retVal;
}

__host__ Camera::Camera(Vector3Df pos, Vector3Df target, Vector3Df upv, float _fov, int x, int y) :
		eye(pos), dir(target), up(upv), fov(tanf(_fov/2.0f * M_PI/180.0f)), xpixels(x), ypixels(y)
{
	aspect = (float)xpixels / (float)ypixels;
	up = normalize(up);
	focusDistance = (dir-eye).length();
	dir = normalize(dir - eye);
	right = normalize(cross(dir,up));
}

__host__ Camera::Camera(string filename, int width, int height) :
	xpixels(width), ypixels(height) {
	ifstream t(filename);
	string json((istreambuf_iterator<char>(t)),
			istreambuf_iterator<char>());
	picojson::value v;
	string err = picojson::parse(v, json);
	if (!err.empty()) {
		std::cerr << err << std::endl;
	}
	const picojson::value::object& obj = v.get<picojson::object>();
	float f = v.get("fieldOfView").get<double>();
	float focalLength = v.get("focalLength").get<double>();
	float fStop = v.get("fStop").get<double>();
	picojson::array e = v.get("eye").get<picojson::array>();
	picojson::array d = v.get("viewDirection").get<picojson::array>();
	picojson::array u = v.get("upDirection").get<picojson::array>();
	focusDistance = v.get("focusDistance").get<double>();

	fov = tanf(f/2.0f * M_PI/180.0f);
	eye = vectorFromArray(e);
	dir = normalize(vectorFromArray(d));
	up = normalize(vectorFromArray(u));
	right = normalize(cross(dir,up));
	apertureWidth = focalLength/fStop;
	aspect = (float)xpixels / (float)ypixels;
}

__host__ __device__ void tentFilter(float &i, float &j, Sampler* p_sampler) {
	float r1, r2;
	r1 = 2.f * p_sampler->getNextFloat();
	r2 = 2.f * p_sampler->getNextFloat();
	float dx;
	if (r1 < 1.f){
		i = sqrtf(r1) - 1.f;
	}
	else{
		i = 1.f - sqrtf(2.f - r1);
	}
	float dy;
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
	float normalized_j = 1.0f - (((float)j + dy) / (float)ypixels);

	Vector3Df direction = dir;
	float rightJitter = 1.0f;
	float upJitter = 1.0f;
	Vector3Df origin = eye;
	direction += (right * fov * aspect * normalized_i);
	direction += (up * fov * normalized_j);

	if (apertureWidth > 0.0f) {
		float r1 = p_sampler->getNextFloat() * 2 * M_PI;
		float r2 = p_sampler->getNextFloat();
		rightJitter = cosf(r1) * r2 * apertureWidth;
		upJitter = sinf(r1)* r2 * apertureWidth;
		origin += ((right * rightJitter) + (up * upJitter));
		Vector3Df focalPoint = eye + direction * focusDistance;
		direction = focalPoint - origin;
	}

	return Ray(origin, normalize(direction));
}
