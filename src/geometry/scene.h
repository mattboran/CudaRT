// Prototypes for model loader - takes obj_load Loader objects and converts them into a format CUDA can use
#ifndef SCENE_H
#define SCENE_H
#include "geometry.h"
#include "obj_load.h"
#include <string>
#include <vector>

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
		int getNumTriangles();
		geom::Triangle* getTriPtr();
		objl::Mesh getMesh(int i);
	private:
		int numLights = 0;
		geom::Triangle* trianglesPtr = NULL;
		objl::Loader meshLoader;

		geom::Triangle* loadTriangles(std::vector<std::string> emissiveMeshes, std::vector<Vector3Df> emissionValues);

	};
}

#endif
