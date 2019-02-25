#include "loaders.h"

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

Vector3Df* TextureLoader::load(std::string filename, int& width, int& height, int& idx) {
    bool isHdr = stbi_is_hdr(filename.c_str());
    cout << "Loading " << filename << ": " << (isHdr ? "is " : "not ") << "hdr." << endl;
    int components = 0;
    float* data = stbi_loadf(filename.c_str(), &width, &height, &components, 0);
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

void TextureLoader::loadAll(std::string* filename, uint numTextures) {
	for (unsigned i = 0; i < numTextures; i++) {
		int texW, texH, idx;
		Vector3Df* p_tex = load(filename[i], texW, texH, idx);
		textureDimensions.push_back((pixels_t)texW);
		textureDimensions.push_back((pixels_t)texH);
		textureDataPtrs.push_back(p_tex);
	}
}
