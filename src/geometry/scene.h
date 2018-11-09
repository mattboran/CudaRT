// Prototypes for model loader - takes obj_load Loader objects and converts them into a format CUDA can use
#ifndef SCENE_H
#define SCENE_H
#include "obj_load.h"
#include "geometry.h"
#include <vector>
#include <string>

namespace scene
{
	class Scene {
	public:
		Scene() { }
		~Scene();
		Scene(std::string filename);
		Scene(std::string filename, std::vector<std::string> emissiveMeshes, std::vector<Vector3Df> emissionValues);
		Scene(std::vector<std::string>& filenames);

		// Get methods
		int getNumMeshes();
		objl::Mesh getMesh(int i);
		int getNumTriangles();
		geom::Triangle* getTriPtr();
	private:
		objl::Loader meshLoader;
		geom::Triangle* trianglesPtr = NULL;
		int numLights = 0;

		geom::Triangle* loadTriangles(std::vector<std::string> emissiveMeshes, std::vector<Vector3Df> emissionValues);

	};
}

#endif
