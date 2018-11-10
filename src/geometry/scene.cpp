// Implementation for Scene functions. This file is responsible for setting up the scene for rendering

#include "scene.h"
#include <iostream>
#include <limits>
#include <math.h>
using namespace geom;
using std::vector;


// Constructors
Scene::Scene(std::string filename) {
	meshLoader = objl::Loader();
	std::cout << "Loading single .obj as scene from " << filename << std::endl;
	if (!meshLoader.LoadFile(filename)) {
		std::cerr << "Failed to load mesh for " << filename << std::endl;
	}
	trianglesPtr = loadTriangles();
}

Scene::Scene(vector<std::string>& filenames) {
	std::cerr << "Multiple .objs not implemented yet!" << std::endl;
}

Scene::~Scene() {
	delete trianglesPtr;
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

Camera* Scene::getCameraPtr() {
	return &camera;
}

void Scene::setCamera(const Camera& cam) {
	camera = Camera(cam);
}

Triangle* Scene::loadTriangles() {
	Triangle* triPtr = new Triangle[getNumTriangles()];
	Triangle* currentTriPtr = triPtr;

	// Also create bounding box for the whole scene
	sceneMax = Vector3Df(FLT_MIN, FLT_MIN, FLT_MIN);
	sceneMin = Vector3Df(FLT_MAX, FLT_MAX, FLT_MAX);
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
			Vector3Df faceNormal = normalize(Vector3Df(v1.Normal) + Vector3Df(v2.Normal) + Vector3Df(v3.Normal));
			currentTriPtr->_normal = Vector3Df(faceNormal);

			sceneMax = max4(currentTriPtr->_v1, currentTriPtr->_v2, currentTriPtr->_v3, sceneMax);
			sceneMin = min4(currentTriPtr->_v1, currentTriPtr->_v2, currentTriPtr->_v3, sceneMin);
			// Materials
			currentTriPtr->_colorDiffuse = Vector3Df(material.Kd);
			currentTriPtr->_colorSpec = Vector3Df(material.Ks);
			currentTriPtr->_colorEmit = Vector3Df(material.Ka);

			currentTriPtr++;
		}
	}
	return triPtr;
}

