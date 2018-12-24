/*
 * launcher.cpp
 *
 *  Created on: Dec 23, 2018
 *      Author: matt
 */

#include "launcher.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stbi_save_image.h"

void BaseLauncher::render() {
	int samples = p_renderer->getSamples();
	for (int s = 0; s < 1; s++) {
		p_renderer->renderOneSamplePerPixel();
	}
	p_renderer->copyImageBytes();
}

void BaseLauncher::saveToImage() {
	int width = p_renderer->getWidth();
	int height = p_renderer->getHeight();
	uchar4* p_imgData = p_renderer->h_imgPtr;
	const unsigned comp = 4;
	const unsigned strideBytes = p_renderer->getWidth() * 4;

	stbi_write_png(outFilename, width, height, comp, p_imgData, strideBytes);
}
