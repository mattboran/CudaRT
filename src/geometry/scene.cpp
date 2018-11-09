// Implementation for Scene functions. This file is responsible for setting up the scene for rendering

#include "scene.h"
#include <iostream>

using namespace scene;

Scene::Scene(std::string filename) {
	meshLoader = objl::Loader();
	std::cout << "Loading single .obj as scene from " << filename << std::endl;
	if (!meshLoader.LoadFile(filename)) {
		std::cerr << "Failed to load mesh for " << filename << std::endl;
	}
}

Scene::Scene(std::vector<std::string>& filenames) {
	std::cerr << "Multiple .objs not implemented yet!" << std::endl;
}
