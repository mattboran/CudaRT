// Prototypes for model loader - takes obj_load Loader objects and converts them into a format CUDA can use
#ifndef SCENE_H
#define SCENE_H

#include "bvh.h"
#include "camera.cuh"
#include "geometry.cuh"
#include "obj_load.h"

#include <string>
#include <vector>

struct BVHNode;
struct CacheFriendlyBVHNode;

class Scene {
public:
	~Scene();
	Scene(std::string filename);
	Scene(std::vector<std::string>& filenames);

	// Get methods
	int getNumMeshes();
	int getNumTriangles();
	int getNumLights();
	unsigned getNumVertices();
	float getLightsSurfaceArea();
	geom::Triangle* getTriPtr();
	geom::Triangle* getLightsPtr();
	objl::Vertex* getVertexPtr();
	unsigned* getVertexIndicesPtr();

	BVHNode* getSceneBVHPtr();
	CacheFriendlyBVHNode* getSceneCFBVHPtr();
	unsigned getNumBVHNodes();
	unsigned getNumCacheFriendlyBVHNodes();

	objl::Mesh getMesh(int i);
	Camera* getCameraPtr();

	// Set methods
	void setCamera(const Camera& cam);
	void setBVHPtr(BVHNode* bvhPtr);
	void setCacheFriendlyVBHPtr(CacheFriendlyBVHNode* bvhPtr);
	void setNumBVHNodes(unsigned i);
	void setNumCacheFriendlyBVHNodes(unsigned i);
private:
	// global variables
	BVHNode* sceneBVH;
	CacheFriendlyBVHNode* sceneCFBVH;
	unsigned numBVHNodes;
	unsigned numCacheFriendlyBVHNodes;
	geom::Triangle* trianglesPtr = NULL;
	std::vector<geom::Triangle> lightsList;
	objl::Loader meshLoader;
	Camera camera;
	unsigned *vertexIndices;
	objl::Vertex *vertexPtr;

	geom::Triangle* loadTriangles();
};


#endif
