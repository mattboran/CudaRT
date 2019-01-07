/*
 * terminal_launcher.cpp
 *
 *  Created on: Dec 23, 2018
 *      Author: matt
 */

#include "launcher.h"

void TerminalLauncher::render() {
	int samples = p_renderer->getSamples();
	uchar4* p_img = p_renderer->getImgBytesPointer();
	for (int s = 0; s < samples; s++) {
		p_renderer->renderOneSamplePerPixel(p_img);
	}
	p_renderer->copyImageBytes();
}
