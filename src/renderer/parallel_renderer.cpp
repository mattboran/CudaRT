/*
 * parallel_renderer.cpp
 *
 *  Created on: Dec 22, 2018
 *      Author: matt
 */

#include "renderer.h"

ParallelRenderer::~ParallelRenderer() {
	free(h_settingsData);
	free(h_lightsData);
	free(h_trianglesData);
}

void ParallelRenderer::renderOneSample() {

}


