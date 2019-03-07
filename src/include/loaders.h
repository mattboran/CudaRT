#ifndef LOADERS_H
#define LOADERS_H

#include "camera.h"
#include "picojson.h"

#include <exception>
#include <fstream>
#include <map>
#include <string>
#include <vector>

typedef unsigned int pixels_t;

class Loader {
public:
	Loader() {}
protected:
	void verifyFileExists(std::string fileName) {
		if (!fileExists(fileName)) {
			throw std::runtime_error("File " + fileName + " not found.");
		}
	}
private:
	bool fileExists(std::string fileName) {
	    std::ifstream infile(fileName.c_str());
	    return infile.good();
	}
};

class CameraJsonLoader : public Loader {
public:
	CameraJsonLoader() : Loader() {}
	CameraJsonLoader(std::string cam);
	Camera getCamera(pixels_t width, pixels_t height);
private:
	picojson::value cameraValue;
};

class EnvLoader : public Loader{
public:
	EnvLoader() : Loader() {}
	EnvLoader(std::string envPath);
	std::string getMeshesPath();
	std::string getCameraPath();
	std::string getTexturesPath();
private:
	std::map<std::string, std::string> settingsDict;
};

// This class handles reading textures from files and copying them to device memory
class TextureStore : public Loader {
public:
    TextureStore() : Loader() { }
    Vector3Df* load(std::string filename, int& width, int& height, int& idx);
    void loadAll(std::string* filename, uint numTextures);
    Vector3Df** getTextureDataPtr() { return &textureDataPtrs[0]; }
    Vector3Df* getFlattenedTextureDataPtr() { return &flattenedTextureData[0]; }
    pixels_t getTotalPixels() { return textureOffsets.back(); }
    pixels_t* getTextureOffsetsPtr() { return &textureOffsets[0]; }
    pixels_t* getTextureDimensionsPtr() { return &textureDimensions[0]; }
    uint getNumTextures() { return currentIdx; }
private:
    uint currentIdx = 0;
    std::vector<pixels_t> textureDimensions;
    std::vector<Vector3Df*> textureDataPtrs;
    std::vector<Vector3Df> flattenedTextureData;
    std::vector<pixels_t> textureOffsets;
};

#endif
