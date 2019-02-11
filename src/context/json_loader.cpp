/*
 * json_loader.cpp
 *
 *  Created on: Feb 4, 2019
 *      Author: matt
 */

#include "json_loader.h"
#include "linalg.h"

#include <exception>
#include <fstream>
#include <iostream>
#include <string>

using namespace std;

Vector3Df vector3FromArray(picojson::array arr);

CameraJsonLoader::CameraJsonLoader(std::string cam) : cameraFile(cam){
	ifstream c(cameraFile);
	string camJson((istreambuf_iterator<char>(c)),
			istreambuf_iterator<char>());
	string err = picojson::parse(cameraValue, camJson);
	if (!err.empty()) {
		throw std::runtime_error(err + " loading camera!");
	}
}

Camera CameraJsonLoader::getCamera(int width, int height) {
	float f = cameraValue.get("fieldOfView").get<double>();
	float focalLength = cameraValue.get("focalLength").get<double>();
	float fStop = cameraValue.get("fStop").get<double>();
	picojson::array e = cameraValue.get("eye").get<picojson::array>();
	picojson::array d = cameraValue.get("viewDirection").get<picojson::array>();
	picojson::array u = cameraValue.get("upDirection").get<picojson::array>();

	Camera camera;
	camera.xpixels = width;
	camera.ypixels = height;
	camera.fov = tanf(f * 0.5f * M_PI/180.0f);
	camera.eye = vector3FromArray(e);
	Vector3Df dir = vector3FromArray(d);
	camera.focusDistance = dir.length();
	camera.dir = normalize(dir);
	camera.up = normalize(vector3FromArray(u));
	camera.right = normalize(cross(camera.dir,camera.up));
	camera.apertureWidth = focalLength/fStop;
	camera.aspect = (float)width / (float)height;

	return camera;
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
