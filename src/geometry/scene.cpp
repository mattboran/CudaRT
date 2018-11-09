// Implementation for Scene functions. This file is responsible for setting up the scene for rendering

//#include "obj_load.h"
#include "scene.h"
#include <iostream>

using namespace scene;
using namespace geom;
using std::vector;
// Constructors
Scene::Scene(std::string filename) : Scene(filename, vector<std::string>(), vector<Vector3Df>()) { }

Scene::Scene(std::string filename, vector<std::string> emissiveMeshes, vector<Vector3Df> emissionValues) {
	meshLoader = objl::Loader();
	std::cout << "Loading single .obj as scene from " << filename << std::endl;
	if (!meshLoader.LoadFile(filename)) {
		std::cerr << "Failed to load mesh for " << filename << std::endl;
	}
	trianglesPtr = loadTriangles(emissiveMeshes, emissionValues);
}

Scene::Scene(vector<std::string>& filenames) {
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

Triangle* Scene::loadTriangles(std::vector<std::string> emissiveMeshes, std::vector<Vector3Df> emissionValues) {
	size_t trianglesSize = getNumTriangles()*sizeof(Triangle);
	Triangle* triPtr = (Triangle*)malloc(trianglesSize);
	Triangle* currentTriPtr = triPtr;

	numLights = std::max(emissiveMeshes.size(), emissionValues.size());

	vector<objl::Mesh> meshes = meshLoader.LoadedMeshes;
	for (auto const& mesh: meshes) {
		vector<objl::Vertex> vertices = mesh.Vertices;
		vector<unsigned> indices = mesh.Indices;
		objl::Material material = mesh.MeshMaterial;

		for (int i = 0; i < vertices.size()/3; i++) {
			objl::Vertex v1 = vertices[indices[i*3]];
			objl::Vertex v2 = vertices[indices[i*3 + 1]];
			objl::Vertex v3 = vertices[indices[i*3 + 2]];
			currentTriPtr->_v1 = Vector3Df(v1.Position);
			currentTriPtr->_v2 = Vector3Df(v2.Position);
			currentTriPtr->_v3 = Vector3Df(v3.Position);

			// Materials
			currentTriPtr->_colorDiffuse = Vector3Df(material.Kd);
			currentTriPtr->_colorSpec = Vector3Df(material.Ks);
			currentTriPtr->_colorEmit = Vector3Df(0.0f, 0.0f, 0.0f);
			for (int i = 0; i < numLights; i++) {
				if (emissiveMeshes[i] == mesh.MeshName) {
					currentTriPtr->_colorEmit = Vector3Df(emissionValues[i]);
				}
			}
			currentTriPtr++;
		}
	}

	return triPtr;
}
