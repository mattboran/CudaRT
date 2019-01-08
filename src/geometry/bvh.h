/*
 * bvh.h
 *
 *  Created on: Jan 7, 2019
 *      Author: matt
 */

#ifndef BVH_H_
#define BVH_H_

#include "geometry.h"

#include <vector>

using std::vector;

struct BVHNode {
	BVHNode(Vector3Df &a, Vector3Df &b) : min(a), max(b) {}
	Vector3Df min;
	Vector3Df max;
	virtual bool isLeaf() = 0;
};

struct BVHInner : BVHNode {
	BVHNode* p_left;
	BVHNode* p_right;
	virtual bool isLeaf() { return false; }
};

struct BVHLeaf : BVHNode {
	std::vector<geom::Triangle*> _triangles;
	virtual bool IsLeaf() { return true; }
};

#endif /* BVH_H_ */
