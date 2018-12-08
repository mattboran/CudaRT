#include "camera.cuh"
#include "sequential.h"
#include <algorithm>
#include <cstdlib>
#include <vector>
#include <cfloat>

using namespace std;
using namespace geom;

typedef vector<Triangle> TriVec;

struct BBox {
	int boxId;
	Vector3Df _bottom;
	Vector3Df _top;
	Vector3Df _color;
	int leftId, rightId;
	bool isLeaf;
	TriVec tris;
};

bool hitsBox(const Ray& ray, BVHNode* bbox) {
	float t0 = -FLT_MAX, t1 = FLT_MAX;
	//axes

	float invRayDir = 1.f/ray.dir.x;
	float tNear = (bbox->_bottom.x - ray.origin.x) * invRayDir;
	float tFar = (bbox->_top.x - ray.origin.x) * invRayDir;
	if (tNear > tFar) {
		float tmp = tNear;
		tNear = tFar;
		tFar = tmp;
	}
	t0 = tNear > t0 ? tNear : t0;
	t1 = tFar < t1 ? tFar : t1;
	if (t0 > t1) return false;

	invRayDir = 1.f/ray.dir.y;
	tNear = (bbox->_bottom.y - ray.origin.y) * invRayDir;
	tFar = (bbox->_top.y - ray.origin.y) * invRayDir;
	if (tNear > tFar) {
		float tmp = tNear;
		tNear = tFar;
		tFar = tmp;
	}
	t0 = tNear > t0 ? tNear : t0;
	t1 = tFar < t1 ? tFar : t1;
	if (t0 > t1) return false;

	invRayDir = 1.f/ray.dir.z;
	tNear = (bbox->_bottom.z - ray.origin.z) * invRayDir;
	tFar = (bbox->_top.z - ray.origin.z) * invRayDir;
	if (tNear > tFar) {
		float tmp = tNear;
		tNear = tFar;
		tFar = tmp;
	}
	t0 = tNear > t0 ? tNear : t0;
	t1 = tFar < t1 ? tFar : t1;
	if (t0 > t1) return false;

	return true;
}

bool recursiveIntersectBVH(BVHNode* bvh,
				  const Ray& ray,
				  Vector3Df *imgPtr) {

	if (!(bvh->IsLeaf())) {   // INNER NODE
		if (hitsBox(ray, bvh)) {
			BVHInner *p = dynamic_cast<BVHInner*>(bvh);
			if (!recursiveIntersectBVH(p->_right, ray, imgPtr)) {
				recursiveIntersectBVH(p->_left, ray, imgPtr);
			}
		}
	}
	else { // LEAF NODE
		BVHLeaf *p = dynamic_cast<BVHLeaf*>(bvh);
		float u, v;
		for (auto tri: p->_triangles) {
			Triangle triangle = *tri;
			if (triangle.intersect(ray, u, v) < FLT_MAX) {
				*imgPtr = triangle._colorDiffuse;
				return true;
			}
		}
	}
	return false;
}


Vector3Df* sequentialRenderWrapper(Scene& scene, int width, int height, int samples, int numStreams, bool &useTexMemory, int argc, char** argv) {

	Vector3Df* img = new Vector3Df[width*height];
	srand(0);
	Camera* camera = scene.getCameraPtr();
	BVHNode* bboxPtr = scene.getSceneBVHPtr();
	Triangle* triangles = scene.getTriPtr();
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++){
			int idx = width*i + j;
			Ray ray = camera->computeSequentialCameraRay(j, i);
			recursiveIntersectBVH(bboxPtr, ray, &img[idx]);
		}
	}
	return img;
}

void testRender(BBox* bboxPtr, Camera* camPtr, Vector3Df* imgPtr, int width, int height) {

}
