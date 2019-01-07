/*
 * launcher.cpp
 *
 *  Created on: Dec 23, 2018
 *      Author: matt
 */

#include "launcher.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stbi_save_image.h"

void Launcher::saveToImage() {
	int width = p_renderer->getWidth();
	int height = p_renderer->getHeight();
	uchar4* p_imgData = p_renderer->h_imgPtr;
	const unsigned comp = 4;
	const unsigned strideBytes = p_renderer->getWidth() * 4;

	stbi_write_png(outFilename, width, height, comp, p_imgData, strideBytes);
}
