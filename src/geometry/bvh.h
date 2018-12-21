#ifndef _BVH_H_
#define _BVH_H_

#include <list>
#include "linalg.h"
#include "scene.h"
#include "geometry.h"

#define BVH_STACK_SIZE 32

class Scene;

struct BVHNode {
	unsigned boxId;
	Vector3Df _bottom;
	Vector3Df _top;
	virtual bool IsLeaf() = 0; // pure virtual
};

struct BVHInner : BVHNode {
	BVHNode *_left;
	BVHNode *_right;
	virtual bool IsLeaf() { return false; }
};

struct BVHLeaf : BVHNode {
	std::list<const geom::Triangle*> _triangles;
	virtual bool IsLeaf() { return true; }
};

struct CacheFriendlyBVHNode {
	// bounding box
	Vector3Df _bottom;
	Vector3Df _top;
	// parameters for leafnodes and innernodes occupy same space (union) to save memory
	// top bit discriminates between leafnode and innernode
	// no pointers, but indices (int): faster

	union {
		// inner node - stores indexes to array of CacheFriendlyBVHNode
		struct {
			unsigned _idxLeft;
			unsigned _idxRight;
		} inner;
		// leaf node: stores triangle count and starting index in triangle list
		struct {
			unsigned _count; // Top-most bit set, leafnode if set, innernode otherwise
			unsigned _startIndexInTriIndexList;
		} leaf;
	} u;
};

// The ugly, cache-friendly form of the BVH: 32 bytes
void CreateCFBVH(); // CacheFriendlyBVH

// The single-point entrance to the BVH - call only this
void CreateBoundingVolumeHeirarchy(Scene* scene);

int CountBoxes(BVHNode *root, bool label);

unsigned CountTriangles(BVHNode *root);

void CountDepth(BVHNode *root, int depth, int& maxDepth);


#endif
