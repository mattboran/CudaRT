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

__host__ Camera::Camera(Vector3Df pos, Vector3Df target, Vector3Df upv, Vector3Df rt, float _fov, int x, int y) :
		eye(pos), dir(target), up(upv), right(rt), fov(tanf(_fov/2.0f * M_PI/180.0f)), xpixels(x), ypixels(y)
{
	aspect = (float)xpixels / (float)ypixels;
	rebase();
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
	picojson::array e = v.get("eye").get<picojson::array>();
	picojson::array d = v.get("viewDirection").get<picojson::array>();
	picojson::array u = v.get("upDirection").get<picojson::array>();
	picojson::array rt = v.get("rightDirection").get<picojson::array>();
	fov = tanf(f/2.0f * M_PI/180.0f);
	eye = vectorFromArray(e);
	dir = eye - vectorFromArray(d);
	up = vectorFromArray(u);
	right = vectorFromArray(rt);
	aspect = (float)xpixels / (float)ypixels;
	rebase();
}

// Compute tent filtered ray
__host__ __device__ Ray Camera::computeCameraRay(int i, int j, Sampler* p_sampler) const {
	float r1, r2;
	r1 = 2.f * p_sampler->getNextFloat();
	r2 = 2.f * p_sampler->getNextFloat();
	float dx;
	if (r1 < 1.f){
		dx = sqrtf(r1) - 1.f;
	}
	else{
		dx = 1.f - sqrtf(2.f - r1);
	}
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

__host__ void Camera::rebase()
{
	up = normalize(up);
	// todo: see if this is needed or correct
	dir = normalize(dir - eye);
	right = normalize(cross(dir,up));
	up = cross(right, dir);
}
