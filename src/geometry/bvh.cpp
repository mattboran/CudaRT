/*
    This code was inspired by chapter 4.4 of PBR by Matt Pharr and Greg Humphreys.

    Great book, instrumental in writing this whole thing but especially this section
*/

#include "bvh.h"

#include <algorithm>
#include <iostream>

// #define USE_MIDPOINT
#define USE_SAH

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

// Vector math functions that are used in making BVH - possibly put these in geometry.h/cu
Vector3Df offset(const Vector3Df& min, const Vector3Df& max, const Vector3Df& p);
float surfaceArea(const Vector3Df& min, const Vector3Df& max);
int maximumExtent(const Vector3Df& min, const Vector3Df& max);
Vector3Df min2(const Vector3Df& a, const Vector3Df& b);
Vector3Df max2(const Vector3Df& a, const Vector3Df& b);

// Local functions
void createTriangleBboxes(Scene* p_scene);
void constructBVH(Scene* p_scene);
BVHBuildNode* recursiveBuild(Triangle* p_triangles, vector<TriangleBBox>& trianglesInfo,
     uint start, uint end, uint* totalNodes, vector<Triangle> orderedTriangles);

vector<TriangleBBox> trianglesInfo;
vector<Triangle> orderedTriangles;
const int maxTrisInNode = 3;

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
    p_scene->setBvhPtr(recursiveBuild(p_scene->getTriPtr(), trianglesInfo, 0, trianglesInfo.size(), &totalNodes, orderedTriangles));
    cout << "Total BVH Nodes: " << totalNodes << endl;// << "Created BVH using mid point heuristic " << endl
    p_scene->setNumBvhNodes(totalNodes);
    // Copy triangles to scene
    Triangle* p_dest = p_scene->getTriPtr();
    std::copy(orderedTriangles.begin(), orderedTriangles.end(), p_dest);
    // Set BVH variables in scene
//    p_scene->setBvhPtr(p_root);
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
    BVHBuildNode* node = new BVHBuildNode;
    Vector3Df workingMin(FLT_MAX, FLT_MAX, FLT_MAX);
    Vector3Df workingMax(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    for (uint i = start; i < end; i++) {
        workingMin = min2(workingMin, trianglesInfo[i].min);
        workingMax = max2(workingMax, trianglesInfo[i].max);
    }
    uint numTriangles = end - start;
    if (numTriangles <= maxTrisInNode) {
        uint firstTriOffset = orderedTriangles.size();
        for (uint i = start; i < end; i++) {
            uint triId = trianglesInfo[i].triId;
            orderedTriangles.push_back(p_triangles[triId]);
        }
        node->initLeaf(firstTriOffset, numTriangles, workingMin, workingMax);
    }
    else  {
        Vector3Df centroidMin, centroidMax;
        for (uint i = start; i < end; i++) {
            centroidMin = min2(centroidMin, trianglesInfo[i].center);
            centroidMax = max2(centroidMax, trianglesInfo[i].center);
        }
        int dim = maximumExtent(centroidMin, centroidMax);
        uint mid = (start + end) / 2;
        if (centroidMax._v[dim] == centroidMin._v[dim]) {
            uint firstTriOffset = orderedTriangles.size();
            for (uint i = start; i < end; i++) {
                uint triId = trianglesInfo[i].triId;
                orderedTriangles.push_back(p_triangles[triId]);
            }
            node->initLeaf(firstTriOffset, numTriangles, workingMin, workingMax);
            return node;
        }
        // Partition primitives based on SAH
#ifdef USE_MIDPOINT
		// Partition based on centroid midpoint
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
			// And then fall through to partition primitives into equally sized subsets if that failed
			mid = (start + end) / 2;
			std::nth_element(&trianglesInfo[start], &trianglesInfo[mid], &trianglesInfo[end-1] + 1,
				[dim](const TriangleBBox& a, const TriangleBBox& b) {
					return a.center._v[dim] < b.center._v[dim];
				});
			break;
		}
#elif defined(USE_SAH)
		if (numTriangles <= 4) {
			// Partition primitives into equally sized subsets
			mid = (start + end) / 2;
			std::nth_element(&trianglesInfo[start], &trianglesInfo[mid], &trianglesInfo[end-1] + 1,
				[dim](const TriangleBBox& a, const TriangleBBox& b) {
					return a.center._v[dim] < b.center._v[dim];
				});
		} else {
			// Allocate BucketInfo for SAH partition buckets
			constexpr int numBuckets = 8;
			struct BucketInfo {
				int count = 0;
				Vector3Df minBound, maxBound;
			};
			BucketInfo buckets[numBuckets];
			// Initialize BucketInfo for SAH partition buckets by determining the bucket
			// that its centroid lies in and updating the bucket's bounds to include the Triangle bounds
			for (int i = start; i < end; i++) {
				int b = numBuckets * offset(centroidMin, centroidMax, trianglesInfo[i].center)._v[dim];
				if (b == numBuckets) { b = numBuckets - 1; }
				buckets[b].count++;
				buckets[b].minBound = min2(buckets[b].minBound, trianglesInfo[i].min);
				buckets[b].maxBound = max2(buckets[b].maxBound, trianglesInfo[i].max);
			}
			// Compute costs of splitting after each bucket
			float cost[numBuckets-1];
			for (int i = 0; i < numBuckets-1; i++) {
				Vector3Df b0min,b0max, b1min, b1max;
				int count0 = 0, count1 = 0;
				for (int j = 0; j <= i; j++) {
					b0min = min2(b0min, buckets[j].minBound);
					b0max = max2(b0max, buckets[j].maxBound);
					count0 += buckets[j].count;
				}
				for (int j = i+1; j < numBuckets; j++) {
					b1min = min2(b1min, buckets[j].minBound);
					b1max = max2(b1max, buckets[j].maxBound);
					count1 += buckets[j].count;
				}
				cost[i] = 0.125f+ (count0 * surfaceArea(b0min, b0max) +
						   count1 * surfaceArea(b1min, b1max)) / surfaceArea(workingMin, workingMax);
			}
			// Find bucket to split after that minimizes SAH
			float* p_minCost = std::min_element(&cost[0], &cost[numBuckets-1]);
			float minCost = *p_minCost;
			int minCostSplitBucket = p_minCost - cost;
			// Either create leaf or split triangles at selected SAH bucket
			float leafCost = numTriangles;
			if (numTriangles > maxTrisInNode || minCost < leafCost) {
				while (true) {
					TriangleBBox *p_mid = std::partition(&trianglesInfo[start], &trianglesInfo[end-1]+1,
						//[numBuckets, dim, minCostSplitBucket, centroidMin, centroidMax]
						[=](const TriangleBBox &t) {
							int b = numBuckets * offset(centroidMin, centroidMax, t.center)._v[dim];
							if (b == numBuckets) b = numBuckets - 1;
							return b <= minCostSplitBucket;
						});
					mid = p_mid - &trianglesInfo[0];
					if (mid != start && mid != end) {
						break;
					}
					// And then fall through to partition primitives into equally sized subsets if that failed
					mid = (start + end) / 2;
					std::nth_element(&trianglesInfo[start], &trianglesInfo[mid], &trianglesInfo[end-1] + 1,
						[dim](const TriangleBBox& a, const TriangleBBox& b) {
							return a.center._v[dim] < b.center._v[dim];
						});
					break;
				}
			} else {
				// Create leaf
				uint firstTriOffset = orderedTriangles.size();
	            for (uint i = start; i < end; i++) {
	                uint triId = trianglesInfo[i].triId;
	                orderedTriangles.push_back(p_triangles[triId]);
	            }
	            node->initLeaf(firstTriOffset, numTriangles, workingMin, workingMax);
			}
		}
#endif
        node->initInner(dim,
                        recursiveBuild(p_triangles, trianglesInfo, mid, end, totalNodes, orderedTriangles),
						recursiveBuild(p_triangles, trianglesInfo, start, mid, totalNodes, orderedTriangles));
    }
    return node;
}

// TODO: Consider moving this into geometry.h
Vector3Df offset(const Vector3Df& min, const Vector3Df& max, const Vector3Df& p) {
	Vector3Df retVal = p - min;
	if (max.x > min.x) retVal.x /= (max.x - min.x);
	if (max.y > min.y) retVal.y /= (max.y - min.y);
	if (max.z > min.z) retVal.z /= (max.z - min.z);
	return retVal;
}

float surfaceArea(const Vector3Df& min, const Vector3Df& max) {
	Vector3Df diag = max - min;
	return diag.x * diag.y * diag.z;
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

Vector3Df min2(const Vector3Df& a, const Vector3Df& b) {
    float x = a.x < b.x ? a.x : b.x;
    float y = a.y < b.y ? a.y : b.y;
    float z = a.z < b.z ? a.z : b.z;
    return Vector3Df(x, y, z);
}

Vector3Df max2(const Vector3Df& a, const Vector3Df& b) {
    float x = a.x > b.x ? a.x : b.x;
    float y = a.y > b.y ? a.y : b.y;
    float z = a.z > b.z ? a.z : b.z;
    return Vector3Df(x, y, z);
}
