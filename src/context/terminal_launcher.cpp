/*
 * terminal_launcher.cpp
 *
 *  Created on: Dec 23, 2018
 *      Author: matt
 */

#include "launcher.h"

void TerminalLauncher::render() {
	int samples = p_renderer->getSamples();
	for (int s = 0; s < 1; s++) {
		p_renderer->renderOneSamplePerPixel();
	}
	p_renderer->copyImageBytes();
}
