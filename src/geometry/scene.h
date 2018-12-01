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
	std::vector<CacheFriendlyBVHNode> cfBVHNodeVector;
	unsigned *getBVHIndexPtr();
	unsigned getNumBVHNodes();

	objl::Mesh getMesh(int i);
	Camera* getCameraPtr();

	// Set methods
	void setCamera(const Camera& cam);
	void setBVHPtr(BVHNode* bvhPtr);
	void setCacheFriendlyVBHPtr(CacheFriendlyBVHNode* bvhPtr);
	void setNumBVHNodes(unsigned i);
	void allocateCFBVHNodeArray(unsigned nodes);
	void allocateBVHNodeIndexArray(unsigned nodes);
private:
	// BVH Variables
	unsigned numBVHNodes;
	BVHNode* sceneBVH;
	// Corresponds with the sceneCFBVH: an index into trianglesPtr
	// Indices into sceneCFBVHPtr
	unsigned* bvhIndexPtr;

	// Geometry - todo: phase these out if possible
	geom::Triangle* trianglesPtr = NULL;
	std::vector<geom::Triangle> lightsList;
	objl::Loader meshLoader;
	Camera camera;
	unsigned *vertexIndices;
	objl::Vertex *vertexPtr;

	geom::Triangle* loadTriangles();
};


#endif
