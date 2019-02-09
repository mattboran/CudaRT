#ifndef TEXTURE_LOADER_H
#define TEXTURE_LOADER_H

#include "linalg.h"

#include <vector>
#include <string>

typedef unsigned int pixels_t;

// This class handles reading textures from files and copying them to device memory
class TextureLoader {
public:
    TextureLoader() { }
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
