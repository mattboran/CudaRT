#ifndef LOADERS_H
#define LOADERS_H

#include "picojson.h"
#include "camera.h"
#include "material.h"

#include <string>
#include <map>

class CameraJsonLoader {
public:
	CameraJsonLoader() {}
	CameraJsonLoader(std::string cam);
	Camera getCamera(int width, int height);
private:
	std::string cameraFile;
	picojson::value cameraValue;
};

class EnvLoader {
public:
	EnvLoader() {}
	EnvLoader(std::string envPath);
	std::string getMeshesPath();
	std::string getCameraPath();
	std::string getTexturesPath();
private:
	std::map<std::string, std::string> settingsDict;
};

#endif
