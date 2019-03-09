/*
 * json_loader.cpp
 *
 *  Created on: Feb 4, 2019
 *      Author: matt
 */

#include "loaders.h"
#include "linalg.h"

using namespace std;

Vector3Df vector3FromArray(picojson::array arr);

CameraJsonLoader::CameraJsonLoader(string cam) {
	verifyFileExists(cam);
	ifstream c(cam);
	string camJson((istreambuf_iterator<char>(c)),
			istreambuf_iterator<char>());
	string err = picojson::parse(cameraValue, camJson);
	if (!err.empty()) {
		throw runtime_error(err + " loading camera!");
	}
}

Camera CameraJsonLoader::getCamera(pixels_t width, pixels_t height) {
	float f = cameraValue.get("fieldOfView").get<double>();
	float focalLength = cameraValue.get("focalLength").get<double>();
	float fStop = cameraValue.get("fStop").get<double>();
	picojson::array e = cameraValue.get("eye").get<picojson::array>();
	picojson::array d = cameraValue.get("viewDirection").get<picojson::array>();
	picojson::array u = cameraValue.get("upDirection").get<picojson::array>();

	Camera* p_camera = (Camera*)malloc(sizeof(Camera));
	p_camera->xpixels = width;
	p_camera->ypixels = height;
	p_camera->fov = tanf(f * 0.5f * M_PI/180.0f);
	p_camera->eye = vector3FromArray(e);
	Vector3Df dir = vector3FromArray(d);
	p_camera->focusDistance = dir.length();
	p_camera->dir = normalize(dir);
	p_camera->up = normalize(vector3FromArray(u));
	p_camera->right = normalize(cross(p_camera->dir, p_camera->up));
	p_camera->apertureWidth = focalLength/fStop;
	p_camera->aspect = (float)width / (float)height;

	return *p_camera;
}

Vector3Df vector3FromArray(picojson::array arr) {
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
