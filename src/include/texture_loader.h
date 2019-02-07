#ifndef TEXTURE_LOADER_H
#define TEXTURE_LOADER_H

#include "linalg.h"

#include <string>

// This class handles reading textures from files and copying them to device memory
class TextureLoader {
public:
    TextureLoader() { }
    Vector3Df* load(std::string filename, int& width, int& height, int& idx);
private:
    int currentIdx = 0;
};

#endif
