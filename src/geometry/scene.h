// Prototypes for model loader - takes obj_load Loader objects and converts them into a format CUDA can use
#ifndef SCENE_H
#define SCENE_H
#include "camera.h"
#include "geometry.h"
#include "obj_load.h"
#include <string>
#include <vector>


class Scene {
public:
	~Scene();
	Scene(std::string filename);
	Scene(std::string filename, std::vector<std::string> emissiveMeshes, std::vector<Vector3Df> emissionValues);
	Scene(std::vector<std::string>& filenames);

	// Get methods
	int getNumMeshes();
	int getNumTriangles();
	geom::Triangle* getTriPtr();
	objl::Mesh getMesh(int i);
	Camera getCamera();

	// Set methods
	void setCamera(const Camera& cam);

private:
	int numLights = 0;
	geom::Triangle* trianglesPtr = NULL;
	objl::Loader meshLoader;
	Camera camera;
	Vector3Df sceneMin;
	Vector3Df sceneMax;

	geom::Triangle* loadTriangles(std::vector<std::string> emissiveMeshes, std::vector<Vector3Df> emissionValues);
};


#endif
