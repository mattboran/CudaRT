// Implementation for Scene functions. This file is responsible for setting up the scene for rendering

#include "scene.h"
#include "bvh.h"

#include <algorithm>
#include <iostream>
#include <cfloat>
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
	vertexIndices = &meshLoader.LoadedIndices[0];
	vertexPtr = &meshLoader.LoadedVertices[0];

	// Create BVH and CFBVH
	triIndexBVHPtr = new unsigned[getNumTriangles()];
	CreateBoundingVolumeHeirarchy(this);
}

Scene::Scene(vector<std::string>& filenames) {
	std::cerr << "Multiple .objs not implemented yet!" << std::endl;
}

Scene::~Scene() {
	// TODO: Find out why there's a memory corruption or double free when we delete
	// the other pointers in this class
//	delete sceneCFBVH;
//	delete triIndexBVHPtr;
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

int Scene::getNumLights() {
	return lightsList.size();
}

unsigned Scene::getNumVertices() {
	return meshLoader.LoadedVertices.size();
}

float Scene::getLightsSurfaceArea() {
	float surfaceArea;
	for (auto light: lightsList) {
		surfaceArea += light._surfaceArea;
	}
	return surfaceArea;
}
Triangle* Scene::getTriPtr() {
	return trianglesPtr;
}

Triangle* Scene::getLightsPtr(){
	return &lightsList[0];
}

objl::Vertex* Scene::getVertexPtr() {
	return vertexPtr;
}

unsigned* Scene::getVertexIndicesPtr(){
	return vertexIndices;
}

BVHNode* Scene::getSceneBVHPtr() {
	return sceneBVH;
}

CacheFriendlyBVHNode* Scene::getSceneCFBVHPtr() {
	return sceneCFBVH;
}

unsigned *Scene::getTriIndexBVHPtr() {
	return triIndexBVHPtr;
}

unsigned Scene::getNumBVHNodes() {
	return numBVHNodes;
}

Camera* Scene::getCameraPtr() {
	return &camera;
}


void Scene::setCamera(const Camera& cam) {
	camera = Camera(cam);
}

void Scene::setBVHPtr(BVHNode *bvhPtr) {
	sceneBVH = bvhPtr;
}

void Scene::setCacheFriendlyVBHPtr(CacheFriendlyBVHNode* bvhPtr) {
	sceneCFBVH = bvhPtr;
}

void Scene::setNumBVHNodes(unsigned i) {
	numBVHNodes = i;
}

void Scene::allocateCFBVHNodeArray(unsigned nodes) {
	sceneCFBVH = new CacheFriendlyBVHNode[nodes];
}

Triangle* Scene::loadTriangles() {
	Triangle* triPtr = (Triangle*)malloc(sizeof(Triangle) * getNumTriangles());
	Triangle* currentTriPtr = triPtr;

	// Min and max for creating bounding boxes.
	const Vector3Df vectorMin = Vector3Df(-FLT_MAX, -FLT_MAX, -FLT_MAX);
	const Vector3Df vectorMax = Vector3Df(FLT_MAX, FLT_MAX, FLT_MAX);
	vector<objl::Mesh> meshes = meshLoader.LoadedMeshes;
	unsigned triId = 0;
	for (auto const& mesh: meshes) {
		vector<objl::Vertex> vertices = mesh.Vertices;
		vector<unsigned> indices = mesh.Indices;
		objl::Material material = mesh.MeshMaterial;

		for (int i = 0; i < vertices.size()/3; i++) {
			currentTriPtr->_id1 = indices[i*3];
			currentTriPtr->_id2 = indices[i*3 + 1];
			currentTriPtr->_id3 = indices[i*3 + 2];
			objl::Vertex v1 = vertices[indices[i*3]];
			objl::Vertex v2 = vertices[indices[i*3 + 1]];
			objl::Vertex v3 = vertices[indices[i*3 + 2]];
			Vector3Df _v1(v1.Position);
			Vector3Df _v2(v2.Position);
			Vector3Df _v3(v3.Position);
			currentTriPtr->_v1 = _v1;
			currentTriPtr->_n1 = Vector3Df(v1.Normal);
			currentTriPtr->_n2 = Vector3Df(v2.Normal);
			currentTriPtr->_n3 = Vector3Df(v3.Normal);
			currentTriPtr->_e1 = _v2 - _v1;
			currentTriPtr->_e2 = _v3 - _v1;

			// Materials
			currentTriPtr->_colorDiffuse = Vector3Df(material.Kd);
			currentTriPtr->_colorSpec = Vector3Df(material.Ks);
			currentTriPtr->_colorEmit = Vector3Df(material.Ka) * 2.0f;

			currentTriPtr->_surfaceArea = cross(currentTriPtr->_e1, currentTriPtr->_e2).length()/2.0f;
			currentTriPtr->_triId = triId++;

			if (currentTriPtr->_colorEmit.lengthsq() > 0.0f) {
				lightsList.push_back(*currentTriPtr);
			}

			// currentTriPtr->_center = Vector3Df((_v1.x + _v2.x + _v3.x) / 3.0f,
			// 		(_v1.y + _v2.y + _v3.y) / 3.0f,
			// 		(_v1.z + _v2.z + _v3.z) / 3.0f);
			currentTriPtr->_bottom = min4(_v1, _v2, _v3, vectorMax);
			currentTriPtr->_top = max4(_v1, _v2, _v3, vectorMin);
			currentTriPtr->_center = (currentTriPtr->_bottom + currentTriPtr->_top) * 0.5f;

			currentTriPtr++;
		}
	}
	std::sort(lightsList.begin(), lightsList.end(),
			[](const Triangle &a, const Triangle &b) -> bool {
		return a._surfaceArea > b._surfaceArea;
	});
	return triPtr;
}
