// Prototypes for model loader - takes obj_load Loader objects and converts them into a format CUDA can use
#ifndef SCENE_H
#define SCENE_H

#include "camera.h"
#include "material.h"

#include "obj_load.h"

#include <algorithm>
#include <string>
#include <vector>

#include "bvh.h"

struct LinearBVHNode;

class Scene {
public:
	~Scene() { };
	Scene(std::string filename);

	// Get methods
	int getNumMeshes() { return meshLoader.LoadedMeshes.size(); }
	int getNumTriangles() { return getNumVertices() / 3; }
	int getNumLights() { return lightsList.size(); }
	unsigned getNumVertices() { return meshLoader.LoadedVertices.size(); }
	unsigned int getNumBvhNodes() { return numBvhNodes; }
	float getLightsSurfaceArea();
	Triangle* getTriPtr() { return p_triangles; }
	Triangle* getLightsPtr() { return &lightsList[0]; }
	objl::Vertex* getVertexPtr() { return p_vertices; }
	unsigned* getVertexIndicesPtr() { return vertexIndices; }
	LinearBVHNode* getBvhPtr() { return p_bvh; }

	objl::Mesh getMesh(int i) { return meshLoader.LoadedMeshes[i]; }
	Camera* getCameraPtr() { return p_camera; }
	Material* getMaterialsPtr() { return &materials[0]; }

	// Set methods
	void setCameraPtr(Camera* p) { p_camera = p; }

	void allocateBvhArray(const unsigned int n) { p_bvh = new LinearBVHNode[n](); numBvhNodes = n; }

private:
	// Geometry - todo: phase these out if possible
	Triangle* p_triangles = NULL;
	std::vector<Triangle> lightsList;
	objl::Loader meshLoader;
	Camera* p_camera;
	unsigned* vertexIndices;
	objl::Vertex* p_vertices;
	LinearBVHNode* p_bvh = NULL;
	unsigned int numBvhNodes;

	std::vector<Material> materials;

	Triangle* loadTriangles();
};


#endif
