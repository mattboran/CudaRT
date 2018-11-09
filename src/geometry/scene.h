// Prototypes for model loader - takes obj_load Loader objects and converts them into a format CUDA can use
#pragma once
#include "obj_load.h"
#include <vector>
#include <string>

namespace scene
{
	class Scene {
	public:
		Scene() { }

		~Scene() { }

		Scene(std::string filename);

		Scene(std::vector<std::string>& filenames);
	private:
		objl::Loader meshLoader;

	};
}

