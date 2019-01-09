/*
    This code was inspired by chapter 4.4 of PBR by Matt Pharr and Greg Humphreys.

    Great book, instrumental in writing this whole thing but especially this section
*/

#include "bvh.h"

#include <algorithm>
#include <iostream>

using namespace geom;
using namespace std;

typedef unsigned int uint;

struct TriangleBBox {
	int triId;
	Vector3Df min;
	Vector3Df max;
	Vector3Df center;
    TriangleBBox(int _triId, Vector3Df &_min, Vector3Df &_max)
    :
    triId(_triId), min(_min), max(_max) {
        center = (_min + _max) * 0.5f;
    }
};

struct BVHBuildNode {
    Vector3Df min;
    Vector3Df max;
    BVHBuildNode *children[2];
    uint splitAxis, firstTriOffset, numTriangles;
    BVHBuildNode() { children[0] = children[1] = NULL; }
    void initLeaf(uint first, uint n, const Vector3Df& _min, const Vector3Df& _max);
    void initInner(uint axis, BVHBuildNode* c0, BVHBuildNode* c1);
};

int maximumExtent(const Vector3Df& min, const Vector3Df& max);
Vector3Df min2(Vector3Df& a, Vector3Df& b);
Vector3Df max2(Vector3Df& a, Vector3Df& b);
void createTriangleBboxes(Scene* p_scene);
void cleanupBboxes();
void constructBVH(Scene* p_scene);
BVHBuildNode* recursiveBuild(Triangle* p_triangles, vector<TriangleBBox>& trianglesInfo,
     uint start, uint end, uint* totalNodes, vector<Triangle> orderedTriangles);

vector<TriangleBBox> trianglesInfo;
vector<Triangle> orderedTriangles;

ostream& operator<< (ostream &out, const Vector3Df &v) {
    out << "("<<v.x<<", "<<v.y<<", "<<v.z<<")";
    return out;
}

ostream& operator<< (ostream &out, const TriangleBBox &b){
    out << "BBox #" << b.triId << ": Min: " << b.min << " Max: " << b.max << " Center: " << b.center << endl;
    return out;
}

void constructBVH(Scene* p_scene) {
    createTriangleBboxes(p_scene);
    uint totalNodes = 0;
    BVHBuildNode* p_root = recursiveBuild(p_scene->getTriPtr(), trianglesInfo, 0, trianglesInfo.size(), &totalNodes, orderedTriangles);
    cout << "Total Nodes: " << totalNodes << endl << "Created BVH using mid point heuristic " << endl;
}

void createTriangleBboxes(Scene* p_scene)  {
    objl::Vertex* p_vertices = p_scene->getVertexPtr();
    unsigned int* p_indices = p_scene->getVertexIndicesPtr();
    int numTriangles = p_scene->getNumTriangles();
    trianglesInfo.reserve(numTriangles);
    orderedTriangles.reserve(numTriangles);
    for (int i = 0; i < numTriangles; i++) {
        objl::Vertex v1 = p_vertices[p_indices[i*3]];
        objl::Vertex v2 = p_vertices[p_indices[i*3 + 1]];
        objl::Vertex v3 = p_vertices[p_indices[i*3 + 2]];
        Vector3Df _v1(v1.Position);
        Vector3Df _v2(v2.Position);
        Vector3Df _v3(v3.Position);
        Vector3Df _min = min3(_v1, _v2, _v3);
        Vector3Df _max = max3(_v1, _v2, _v3);
        trianglesInfo.push_back(TriangleBBox(i, _min, _max));
    }
}

void BVHBuildNode::initLeaf(uint first, uint n, const Vector3Df& _min, const Vector3Df& _max) {
    firstTriOffset = first;
    numTriangles = n;
    min = _min;
    max = _max;
}

void BVHBuildNode::initInner(uint axis, BVHBuildNode* c0, BVHBuildNode* c1) {
    children[0] = c0;
    children[1] = c1;
    max = max2(c0->max, c1->max);
    min = min2(c0->min, c1->min);
    splitAxis = axis;
    numTriangles = 0;
}

BVHBuildNode* recursiveBuild(Triangle* p_triangles, vector<TriangleBBox>& trianglesInfo, uint start, uint end, uint* totalNodes, vector<Triangle> orderedTriangles) {
    (*totalNodes)++;
    stopper++;
    BVHBuildNode* node = new BVHBuildNode;
    Vector3Df workingMin(FLT_MAX, FLT_MAX, FLT_MAX);
    Vector3Df workingMax(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    for (uint i = start; i < end; i++) {
        workingMin = min2(workingMin, trianglesInfo[i].min);
        workingMax = max2(workingMax, trianglesInfo[i].max);
    }
    uint numTriangles = end - start;
    if (numTriangles == 1) {
        uint firstTriOffset = orderedTriangles.size();
        for (uint i = start; i < end; ++i) {
            uint triId = trianglesInfo[i].triId;
            orderedTriangles.push_back(p_triangles[triId]);
        }
        node->initLeaf(firstTriOffset, numTriangles, workingMin, workingMax);
    }
    else  {
        Vector3Df centroidMin, centroidMax;
        for (uint i = start; i < end; ++i) {
            centroidMin = min2(centroidMin, trianglesInfo[i].center);
            centroidMax = max2(centroidMax, trianglesInfo[i].center);
        }
        int dim = maximumExtent(centroidMin, centroidMax);
        uint mid = (start + end) / 2;
        if (centroidMax._v[dim] == centroidMin._v[dim]) {
            uint firstTriOffset = orderedTriangles.size();
            for (uint i = start; i < end; ++i) {
                uint triId = trianglesInfo[i].triId;
                orderedTriangles.push_back(p_triangles[triId]);
            }
            node->initLeaf(firstTriOffset, numTriangles, workingMin, workingMax);
            return node;
        }
        // Partition primitives based on SAH
        // Temporary partition based on midpoint
		while (true) {
			float pmid = .5f * (centroidMin._v[dim] + centroidMax._v[dim]);
			TriangleBBox* p_mid = std::partition(&trianglesInfo[start], &trianglesInfo[end-1] + 1,
				[dim, pmid](const TriangleBBox& a) -> bool {
					return a.center._v[dim] < pmid;
				});
			mid = p_mid - &trianglesInfo[0];
			if (mid != start && mid != end) {
				break;
			}
			mid = (start + end) / 2;
			std::nth_element(&trianglesInfo[start], &trianglesInfo[mid], &trianglesInfo[end-1] + 1,
				[dim](const TriangleBBox& a, const TriangleBBox& b) {
					return a.center._v[dim] < b.center._v[dim];
				});
			break;
		}
        node->initInner(dim,
                        recursiveBuild(p_triangles, trianglesInfo, mid, end, totalNodes, orderedTriangles),
						recursiveBuild(p_triangles, trianglesInfo, start, mid, totalNodes, orderedTriangles));
    }
    return node;
}

int maximumExtent(const Vector3Df& min, const Vector3Df& max) {
    Vector3Df diag = max - min;
    int retVal = 0;
    if (diag.x > diag.y && diag.x > diag.z)
        return 0;
    if (diag.y > diag.z)
        return 1;
    return 2;
}

Vector3Df min2(Vector3Df& a, Vector3Df& b) {
    float x = a.x < b.x ? a.x : b.x;
    float y = a.y < b.y ? a.y : b.y;
    float z = a.z < b.z ? a.z : b.z;
    return Vector3Df(x, y, z);
}

Vector3Df max2(Vector3Df& a, Vector3Df& b) {
    float x = a.x > b.x ? a.x : b.x;
    float y = a.y > b.y ? a.y : b.y;
    float z = a.z > b.z ? a.z : b.z;
    return Vector3Df(x, y, z);
}
