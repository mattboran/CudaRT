// Implementation for Scene functions. This file is responsible for setting up the scene for rendering

//#include "obj_load.h"
#include "scene.h"
#include <iostream>

using namespace scene;

// Constructors
Scene::Scene(std::string filename) {
	meshLoader = objl::Loader();
	std::cout << "Loading single .obj as scene from " << filename << std::endl;
	if (!meshLoader.LoadFile(filename)) {
		std::cerr << "Failed to load mesh for " << filename << std::endl;
	}
	trianglesPtr = loadTriangles();
}

Scene::Scene(std::vector<std::string>& filenames) {
	std::cerr << "Multiple .objs not implemented yet!" << std::endl;
}

Scene::~Scene() {
	free(trianglesPtr);
}

// Get methods
int Scene::getNumMeshes() {
	return meshLoader.LoadedMeshes.size();
}

objl::Mesh Scene::getMesh(int i) {
	return meshLoader.LoadedMeshes[i];
}

int Scene::getNumTriangles() {
	return meshLoader.LoadedVertices.size() / 3;
}

Triangle* Scene::getTriPtr() {
	return trianglesPtr;
}

Triangle* Scene::loadTriangles() {
	size_t trianglesSize = getNumTriangles()*sizeof(Triangle);
	Triangle* triPtr = (Triangle*)malloc(trianglesSize);
	Triangle* currentTriPtr = triPtr;

	std::vector<objl::Vertex> vertices = meshLoader.LoadedVertices;
	std::vector<unsigned> indices = meshLoader.LoadedIndices;
	for (int i = 0; i < getNumTriangles(); i++) {
		currentTriPtr->_v1 = Vector3Df(vertices[indices[i*3]].Position);
		currentTriPtr->_v2 = Vector3Df(vertices[indices[i*3 + 1]].Position);
		currentTriPtr->_v2 = Vector3Df(vertices[indices[i*3 + 2]].Position);
		currentTriPtr++;
	}

	return triPtr;
}
