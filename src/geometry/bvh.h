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

struct BVHBuildNode {
    Vector3Df min;
    Vector3Df max;
    BVHBuildNode *children[2];
    uint splitAxis, firstTriOffset, numTriangles;
    BVHBuildNode() { children[0] = children[1] = NULL; }
	~BVHBuildNode() { if(children[0]) delete children[0]; if (children[1]) delete children[1]; }
    void initLeaf(uint first, uint n, const Vector3Df& _min, const Vector3Df& _max);
    void initInner(uint axis, BVHBuildNode* c0, BVHBuildNode* c1);
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
