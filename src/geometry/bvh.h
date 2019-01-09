/*
 * bvh.h
 *
 *  Created on: Jan 7, 2019
 *      Author: matt
 */

#ifndef BVH_H_
#define BVH_H_

#include "geometry.h"
#include "scene.h"

#include <vector>

void constructBVH(Scene* p_scene);

struct BVHNode {
	BVHNode(Vector3Df &a, Vector3Df &b) : min(a), max(b) {}
	Vector3Df min;
	Vector3Df max;
	virtual bool isLeaf() = 0;
	virtual ~BVHNode();
};

struct BVHInner : BVHNode {
	BVHNode* p_left;
	BVHNode* p_right;
	virtual bool isLeaf() { return false; }
};

struct BVHLeaf : BVHNode {
	std::vector<geom::Triangle*> v_triangles;
	virtual bool isLeaf() { return true; }
};

#endif /* BVH_H_ */
