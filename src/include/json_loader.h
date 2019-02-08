#ifndef JSON_LOADER_H
#define JSON_LOADER_H

#include "picojson.h"
#include "camera.h"
#include "material.h"

#include <string>

class JsonLoader{
public:
	JsonLoader() {}
	JsonLoader(std::string cam, std::string mat);
	Camera getCamera(int width, int height);
private:
	std::string cameraFile;
	std::string materialsFile;
	picojson::value cameraValue;
	picojson::value materialsValue;
};

#endif
