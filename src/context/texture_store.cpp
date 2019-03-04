#include "loaders.h"

#include <algorithm>
#include <iostream>

#define STBI_NO_JPEG
#define STBI_NO_BMP
#define STBI_NO_PSD
#define STBI_NO_TGA
#define STBI_NO_GIF
#define STBI_NO_PIC
#define STBI_NO_PNM
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using namespace std;

Vector3Df* TextureStore::load(std::string fileName, int& width, int& height, int& idx) {
	verifyFileExists(fileName);
    bool isHdr = stbi_is_hdr(fileName.c_str());
    cout << "Loading " << fileName << ": " << (isHdr ? "is " : "not ") << "hdr." << endl;
    int components = 0;
    float* data = stbi_loadf(fileName.c_str(), &width, &height, &components, 0);
    int pixels = width * height;
    Vector3Df* out = new Vector3Df[pixels]();
    for (int i = 0; i < pixels; i++) {
        // Ignore the last component (alpha) in the case of PNGs
        out[i]._v[0] = data[i * components];
        out[i]._v[1] = data[i * components + 1];
        out[i]._v[2] = data[i * components + 2];
    }
    stbi_image_free(data);
    idx = currentIdx++;
    return out;
}

void TextureStore::loadAll(std::string* fileName, uint numTextures) {
	pixels_t totalTexturePixels = 0;
	textureOffsets.push_back(0);
	for (uint i = 0; i < numTextures; i++) {
		int texW, texH, idx;
		Vector3Df* p_tex = load(fileName[i], texW, texH, idx);
		pixels_t texels = texW * texH;
		textureDimensions.push_back((pixels_t)texW);
		textureDimensions.push_back((pixels_t)texH);
		textureDataPtrs.push_back(p_tex);
		totalTexturePixels += texels;
		textureOffsets.push_back(totalTexturePixels);
	}

	cout << "Loaded " << numTextures << " textures with offsets: " << endl;
	for (pixels_t offset: textureOffsets) {
		cout << offset << endl;
	}

	// Now flatten the data
	flattenedTextureData.reserve(totalTexturePixels);
	for (uint i = 0; i < numTextures; i++) {
		pixels_t start = textureOffsets[i];
		pixels_t end = textureOffsets[i + 1];
		Vector3Df* p_current = textureDataPtrs[i];
		while (start < end) {
			flattenedTextureData.push_back(*p_current);
			start++; p_current++;
		}
	}
}
