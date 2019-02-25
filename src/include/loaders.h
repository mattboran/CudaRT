#ifndef LOADERS_H
#define LOADERS_H

#include "picojson.h"
#include "camera.h"
#include "material.h"

#include <map>
#include <string>
#include <vector>

typedef unsigned int pixels_t;

class CameraJsonLoader {
public:
	CameraJsonLoader() {}
	CameraJsonLoader(std::string cam);
	Camera getCamera(pixels_t width, pixels_t height);
private:
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

// This class handles reading textures from files and copying them to device memory
class TextureStore {
public:
    TextureStore() { }
    Vector3Df* load(std::string filename, int& width, int& height, int& idx);
    void loadAll(std::string* filename, uint numTextures);
    pixels_t* getTextureDimensionsPtr() { return &textureDimensions[0]; }
    Vector3Df** getTextureDataPtr() { return &textureDataPtrs[0]; }
private:
    int currentIdx = 0;
    std::vector<pixels_t> textureDimensions;
    std::vector<Vector3Df*> textureDataPtrs;
};

#endif
