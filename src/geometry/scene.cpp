// Implementation for Scene functions. This file is responsible for setting up the scene for rendering

#include "scene.h"
#include <iostream>
#include <limits>
#include <math.h>
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

Camera Scene::getCamera() {
	return camera;
}

void Scene::setCamera(const Camera& cam) {
	camera = Camera(cam);
}

Triangle* Scene::loadTriangles(std::vector<std::string> emissiveMeshes, std::vector<Vector3Df> emissionValues) {
	Triangle* triPtr = new Triangle[getNumTriangles()];
	Triangle* currentTriPtr = triPtr;

	numLights = std::max(emissiveMeshes.size(), emissionValues.size());

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
			Vector3Df faceNormal = (v1.Normal + v2.Normal + v3.Normal) / 3.0f;
			currentTriPtr->_normal = Vector3Df(faceNormal);

			sceneMax = max4(currentTriPtr->_v1, currentTriPtr->_v2, currentTriPtr->_v3, sceneMax);
			sceneMin = min4(currentTriPtr->_v1, currentTriPtr->_v2, currentTriPtr->_v3, sceneMin);
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

