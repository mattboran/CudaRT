/*
 * json_loader.cpp
 *
 *  Created on: Feb 4, 2019
 *      Author: matt
 */

#include "loaders.h"
#include "linalg.h"

using namespace std;

float3 make_float3(picojson::array arr);

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
	p_camera->eye = make_float3(e);
	float3 dir = make_float3(d);
	p_camera->focusDistance = length(dir);
	p_camera->dir = normalize(dir);
	p_camera->up = normalize(make_float3(u));
	p_camera->right = normalize(cross(p_camera->dir, p_camera->up));
	p_camera->apertureWidth = focalLength/fStop;
	p_camera->aspect = (float)width / (float)height;

	return *p_camera;
}

float3 make_float3(picojson::array arr) {
	float3 retVal;
	int idx = 0;
	float val = 0.0f;
	auto it = arr.begin();
	val = it->get<double>();
	retVal.x = val;
	it++;
	val = it->get<double>();
	retVal.y = val;
	it++;
	val = it->get<double>();
	retVal.z = val;
	return retVal;
}
