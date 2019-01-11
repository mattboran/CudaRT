// Prototypes for model loader - takes obj_load Loader objects and converts them into a format CUDA can use
#ifndef SCENE_H
#define SCENE_H

#include "obj_load.h"

#include <string>
#include <vector>

#include "camera.h"

struct BVHNode;
struct CacheFriendlyBVHNode;

class Scene {
public:
	~Scene() {};
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

	unsigned getNumBVHNodes();

	objl::Mesh getMesh(int i);
	Camera* getCameraPtr();

	// Set methods
	void setCamera(const Camera& cam);
private:
	// Geometry - todo: phase these out if possible
	geom::Triangle* p_triangles = NULL;
	std::vector<geom::Triangle> lightsList;
	objl::Loader meshLoader;
	Camera camera;
	unsigned *vertexIndices;
	objl::Vertex *vertexPtr;

	geom::Triangle* loadTriangles();
};


#endif
