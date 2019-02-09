#include "renderer.h"
#include <iostream>

__host__ TextureRenderer::TextureRenderer(Vector3Df* p_texture, pixels_t texWidth, pixels_t texHeight, pixels_t _width, pixels_t _height) :
    Renderer(NULL, _width, _height, 1) {
    h_texture = p_texture;
    h_dimensions = new pixels_t[2];
    h_dimensions[0] = texWidth;
    h_dimensions[1] = texHeight;
    allocateTextures(&h_dimensions[0], 1);
    loadTextures(&h_texture, h_dimensions, 1);
}

__host__ void TextureRenderer::renderOneSamplePerPixel(uchar4* p_img) {
    samplesRendered++;
    for (uint x = 0; x < width; x++) {
        for (uint y = 0; y < height; y++) {
            int idx = y * width + x;
            float u = (float)x / (float)width;
            float v = (float)y / (float)height;
             Vector3Df color = sampleTexture(0, u, v);
            p_img[idx] = vector3ToUchar4(color);
        }
    }
}

__host__ void TextureRenderer::copyImageBytes(uchar4* p_img) {
    pixels_t pixels = width * height;
    size_t imgBytes = sizeof(uchar4) * pixels;
    memcpy(h_imgPtr, p_img, imgBytes);
    for (unsigned i = 0; i < pixels; i++) {
        gammaCorrectPixel(h_imgPtr[i]);
    }
}
