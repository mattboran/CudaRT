#ifndef JSON_LOADER_H
#define JSON_LOADER_H

#include "picojson.h"
#include "camera.h"
#include "material.h"

#include <string>

class CameraJsonLoader{
public:
	CameraJsonLoader() {}
	CameraJsonLoader(std::string cam);
	Camera getCamera(int width, int height);
private:
	std::string cameraFile;
	picojson::value cameraValue;
};

#endif
