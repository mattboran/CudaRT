/*
 * bvh.h
 *
 *  Created on: Jan 7, 2019
 *      Author: matt
 */

#ifndef BVH_H_
#define BVH_H_
#include "linalg.h"


class Scene;

void constructBVH(Scene* p_scene);

struct LinearBVHNode {
	Vector3Df min;
	Vector3Df max;
	union {
		int trianglesOffset;
		int secondChildOffset;
	};
	int16_t numTriangles = 0;
	uint8_t axis;
	int8_t pad = 0;
};


//struct BVHNode {
//	BVHNode(Vector3Df &a, Vector3Df &b) : min(a), max(b) {}
//	Vector3Df min;
//	Vector3Df max;
//	virtual bool isLeaf() = 0;
//	virtual ~BVHNode();
//};
//
//struct BVHInner : BVHNode {
//	BVHNode* p_left;
//	BVHNode* p_right;
//	virtual bool isLeaf() { return false; }
//};
//
//struct BVHLeaf : BVHNode {
//	std::vector<Triangle> v_triangles;
//	virtual bool isLeaf() { return true; }
//};

#endif /* BVH_H_ */
