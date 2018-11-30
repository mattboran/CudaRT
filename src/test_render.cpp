#include "camera.cuh"
#include "test_render.h"
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

static int boxId;
vector<BBox> bboxes;
vector<int> bvhIndices;

std::ostream& operator << (std::ostream& o, const Vector3Df &v) {
	o << "x: " << v.x << "\ty: " << v.y <<  "\tz: " << v.z << std::endl;
	return o;
}

std::ostream& operator << (std::ostream& o, const geom::Triangle *v) {
	o << "Triangle with ID: " << v->_triId << std::endl;
	return o;
}

std::ostream& operator << (std::ostream& o, const BBox &b) {
	o << "BBox ID: " << b.boxId;
	if (b.isLeaf) {
		o << ". Is a leaf and has tris:" << endl;
		for (auto tri: b.tris) {
			o << "Triangle with ID: " << tri._triId << std::endl;
		}
	} else {
		o << ". Is an inner node with children:" << endl;
		o << "Left: " << b.leftId << " and Right: " << b.rightId << endl;
	}
	return o;
}

int AddBoxes(BVHNode *root)
{
	BBox bbox;
	bbox.boxId = boxId++;
	bbox._bottom = root->_bottom;
	bbox._top = root->_top;
	bbox._color = Vector3Df((float)(rand() % 255)/255.0f,
							(float)(rand() % 255)/255.0f,
							(float)(rand() % 255)/255.0f);
	if (!root->IsLeaf()) {
		BVHInner *p = dynamic_cast<BVHInner*>(root);
		bbox.isLeaf = false;
		bbox.leftId = AddBoxes(p->_right);
		bbox.rightId = AddBoxes(p->_left);
		bboxes.push_back(bbox);
	}
	else
	{
		BVHLeaf *p = dynamic_cast<BVHLeaf*>(root);
		bbox.isLeaf = true;
		TriVec tris;
		for (auto tri: p->_triangles) {
			Triangle triangle = *tri;
			tris.push_back(triangle);
		}
		bbox.tris = tris;
		bboxes.push_back(bbox);
		return bbox.boxId;
	}
	return bbox.boxId;
}
int hitsBox(const Ray& ray, BBox* bbox) {
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
	if (t0 > t1) return -1;

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
	if (t0 > t1) return -1;

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
	if (t0 > t1) return -1;

	return bbox->boxId;
}
void AddBVHIndices() {
	int i = 0;
	while (i < bboxes.size()) {
		for (int j = 0; j < bboxes.size(); j++) {
			if (bboxes[j].boxId == i) {
				bvhIndices.push_back(j);
				i++;
				break;
			}
		}

	}
}
float testIntersectBVH(BBox* bvh,
				  const Ray& ray,
				  Vector3Df *imgPtr) {
	int stack[BVH_STACK_SIZE];
	int stackIdx = 0;
	stack[stackIdx++] = 0;
	// while the stack is not empty
	while (stackIdx) {
		// pop a BVH node from the stack
		int boxIdx = bvhIndices[stack[--stackIdx]];
		BBox* pCurrent = &bvh[boxIdx];

		if (!(pCurrent->isLeaf)) {   // INNER NODE
			// if ray intersects inner node, push indices of left and right child nodes on the stack
			if (hitsBox(ray, pCurrent) >= 0) {
				stack[stackIdx++] = pCurrent->rightId;
				stack[stackIdx++] = pCurrent->leftId;
				// return if stack size is exceeded
				if (stackIdx>BVH_STACK_SIZE) {
					return FLT_MAX;
				}
			}
		}
		else { // LEAF NODE
			float u, v;
			for (Triangle tri: pCurrent->tris) {
				if (tri.intersect(ray, u, v) < FLT_MAX) {
					*imgPtr = tri._colorDiffuse;
				}
			}
		}
	}
	return FLT_MAX;
}
Vector3Df* testRenderWrapper(Scene& scene, int width, int height, int samples, int numStreams, bool &useTexMemory, int argc, char** argv) {

	Vector3Df* img = new Vector3Df[width*height];
	srand(0);
	AddBoxes(scene.getSceneBVHPtr());
	AddBVHIndices();
	float u = -1.0f, v = -1.0f;
	Camera* camera = scene.getCameraPtr();
	BBox* bboxPtr = &bboxes[bvhIndices[0]];
	Triangle* triangles = scene.getTriPtr();
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++){
			int idx = width*i + j;
			Ray ray = camera->computeTestCameraRay(j, i);
			testIntersectBVH(&bboxes[0], ray, &img[idx]);
		}
	}
	return img;
}

void testRender(BBox* bboxPtr, Camera* camPtr, Vector3Df* imgPtr, int width, int height) {

}
