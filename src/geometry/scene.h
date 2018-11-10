// Prototypes for model loader - takes obj_load Loader objects and converts them into a format CUDA can use
#ifndef SCENE_H
#define SCENE_H
#include "camera.cuh"
#include "geometry.cuh"
#include "obj_load.h"
#include <string>
#include <vector>


class Scene {
public:
	~Scene();
	Scene(std::string filename);
	Scene(std::vector<std::string>& filenames);

	// Get methods
	int getNumMeshes();
	int getNumTriangles();
	geom::Triangle* getTriPtr();
	objl::Mesh getMesh(int i);
	Camera* getCameraPtr();

	// Set methods
	void setCamera(const Camera& cam);

private:
	geom::Triangle* trianglesPtr = NULL;
	objl::Loader meshLoader;
	Camera camera;
	Vector3Df sceneMin;
	Vector3Df sceneMax;

	geom::Triangle* loadTriangles();
};


#endif
