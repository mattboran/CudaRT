// Prototypes for model loader - takes obj_load Loader objects and converts them into a format CUDA can use
#ifndef SCENE_H
#define SCENE_H

#include "camera.h"

#include "obj_load.h"

#include <string>
#include <vector>


struct BVHNode;
struct BVHBuildNode;
struct CacheFriendlyBVHNode;

class Scene {
public:
	~Scene() {};
	Scene(std::string filename);

	// Get methods
	int getNumMeshes() { return meshLoader.LoadedMeshes.size(); }
	int getNumTriangles() { return getNumVertices() / 3; }
	int getNumLights() { return lightsList.size(); }
	unsigned getNumVertices() { return meshLoader.LoadedVertices.size(); }
	unsigned getNumBvhNodes() { return numBvhNodes; }
	float getLightsSurfaceArea();
	Triangle* getTriPtr() { return p_triangles; }
	Triangle* getLightsPtr() { return &lightsList[0]; }
	objl::Vertex* getVertexPtr() { return p_vertices; }
	unsigned* getVertexIndicesPtr() { return vertexIndices; }
	BVHBuildNode* getBvhPtr() { return p_bvh; }
	unsigned int getNumBVHNodes() { return numBvhNodes; }

	objl::Mesh getMesh(int i) { return meshLoader.LoadedMeshes[i]; }
	Camera* getCameraPtr() { return &camera; }

	// Set methods
	void setCamera(const Camera& cam) { camera = Camera(cam); }
	void setBvhPtr(BVHBuildNode* p) { p_bvh = p; }
	void setNumBvhNodes(unsigned int n) { numBvhNodes = n; }

private:
	// Geometry - todo: phase these out if possible
	Triangle* p_triangles = NULL;
	std::vector<Triangle> lightsList;
	objl::Loader meshLoader;
	Camera camera;
	unsigned *vertexIndices;
	objl::Vertex *p_vertices;
	BVHBuildNode* p_bvh;
	unsigned int numBvhNodes;

	Triangle* loadTriangles();
};


#endif
