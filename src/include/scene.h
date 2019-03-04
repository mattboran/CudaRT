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
	~Scene() { }
	Scene() { }

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
	uint* getVertexIndicesPtr() { return p_vertexIndices; }
	LinearBVHNode* getBvhPtr() { return p_bvh; }

	objl::Mesh getMesh(int i) { return meshLoader.LoadedMeshes[i]; }
	Camera* getCameraPtr() { return p_camera; }
	Material* getMaterialsPtr() { return p_materials; }

	uint getNumTextures() { return p_textureStore->getNumTextures(); }
	pixels_t getTotalTexturePixels() { return p_textureStore->getTotalPixels(); }
	Vector3Df* getTexturePtr(uint i) { return p_textureStore->getTextureDataPtr()[i]; }
	Vector3Df* getTexturePtr() { return p_textureStore->getFlattenedTextureDataPtr(); }
	pixels_t* getTextureDimensionsPtr() { return p_textureStore->getTextureDimensionsPtr(); }
	pixels_t* getTextureOffsetsPtr() { return p_textureStore->getTextureOffsetsPtr(); }

	// Load functions
	void loadObj(std::string objPath);
	void loadCamera(std::string cameraPath, pixels_t width, pixels_t height);
	void loadTriangles();
	void loadTextures(std::string texturesPath);
	void constructBvh();

	// For bvh construction
	void allocateBvhArray(const uint n) { p_bvh = new LinearBVHNode[n](); numBvhNodes = n; }

private:
	// Geometry - todo: phase these out if possible
	objl::Loader meshLoader;
	uint* p_vertexIndices;
	objl::Vertex* p_vertices;
	Triangle* p_triangles;
	std::vector<Triangle> lightsList;
	Camera* p_camera;

	LinearBVHNode* p_bvh;
	uint numBvhNodes;

	Material* p_materials;
	uint numMaterials;
	TextureStore* p_textureStore = NULL;
	std::vector<std::string> textureFiles;
};


#endif
