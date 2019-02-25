// Prototypes for model loader - takes obj_load Loader objects and converts them into a format CUDA can use
#ifndef SCENE_H
#define SCENE_H

#include "bvh.h"
#include "camera.h"
#include "loaders.h"
#include "material.h"

#include "obj_load.h"

#include <algorithm>
#include <string>
#include <vector>


struct LinearBVHNode;

class Scene {
public:
	~Scene() { };
	Scene();
	Scene(std::string filename);

	// Get methods
	int getNumMeshes() { return meshLoader.LoadedMeshes.size(); }
	int getNumTriangles() { return getNumVertices() / 3; }
	int getNumLights() { return lightsList.size(); }
	uint getNumVertices() { return meshLoader.LoadedVertices.size(); }
	uint getNumBvhNodes() { return numBvhNodes; }
	uint getNumMaterials() { return numMaterials; }
	float getLightsSurfaceArea();
	Triangle* getTriPtr() { return p_triangles; }
	Triangle* getLightsPtr() { return &lightsList[0]; }
	objl::Vertex* getVertexPtr() { return p_vertices; }
	uint* getVertexIndicesPtr() { return vertexIndices; }
	LinearBVHNode* getBvhPtr() { return p_bvh; }

	objl::Mesh getMesh(int i) { return meshLoader.LoadedMeshes[i]; }
	Camera* getCameraPtr() { return p_camera; }
	Material* getMaterialsPtr() { return p_materials; }

	// Load functions
	Camera* loadCamera(std::string cameraPath);
	Triangle* loadTriangles(std::string objPath);
	void loadTextures(std::string texturesPath);

	// Set methods
	void setCameraPtr(Camera* p) { p_camera = p; }

	// For bvh construction
	void allocateBvhArray(const uint n) { p_bvh = new LinearBVHNode[n](); numBvhNodes = n; }

private:
	// Geometry - todo: phase these out if possible
	objl::Loader meshLoader;
	uint* vertexIndices;
	objl::Vertex* p_vertices;
	Triangle* p_triangles;
	std::vector<Triangle> lightsList;
	Camera* p_camera;

	LinearBVHNode* p_bvh;
	uint numBvhNodes;

	Material* p_materials;
	uint numMaterials;
	TextureStore textureStore;
	std::vector<std::string> textureFiles;

	void getTextureFilesFromMaterials();
	void loadTextures(std::string texturesPath);
	Triangle* loadTriangles();
};


#endif
