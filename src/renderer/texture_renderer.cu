#include "renderer.h"
#include <iostream>

__host__ TextureRenderer::TextureRenderer(Vector3Df* p_texture, int _width, int _height) :
    Renderer(NULL, _width, _height, 1) {
    h_texture = p_texture;
}

__host__ void TextureRenderer::renderOneSamplePerPixel(uchar4* p_img) {
    samplesRendered++;
    for (unsigned x = 0; x < width; x++) {
        for (unsigned y = 0; y < height; y++) {
            int idx = y * width + x;
            // Vector3Df color(1, 0, 0);
            p_img[idx] = vector3ToUchar4(h_texture[idx]);
        }
    }
}

__host__ void TextureRenderer::copyImageBytes(uchar4* p_img) {
    int pixels = width * height;
    size_t imgBytes = sizeof(uchar4) * pixels;
    memcpy(h_imgPtr, p_img, imgBytes);
    for (unsigned i = 0; i < pixels; i++) {
        gammaCorrectPixel(h_imgPtr[i]);
    }
}
