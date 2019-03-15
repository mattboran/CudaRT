/*
    This code was inspired by chapter 4.4 of PBR by Matt Pharr and Greg Humphreys.

    Great book, instrumental in writing this whole thing but especially this section
*/

#include "bvh.h"
#include "scene.h"

#include <algorithm>
#include <iostream>

// #define USE_MIDPOINT
#define USE_SAH

using namespace std;

struct TriangleBBox {
	int triId;
	float3 min;
	float3 max;
	float3 center;
    TriangleBBox(int _triId, float3 &_min, float3 &_max)
    :
    triId(_triId), min(_min), max(_max) {
        center = (_min + _max) * 0.5f;
    }
};

struct BVHBuildNode {
    float3 min;
    float3 max;
    BVHBuildNode* children[2];
    uint splitAxis=0, firstTriOffset=0, numTriangles = 0;
    BVHBuildNode() { children[0] = children[1] = NULL; }
	~BVHBuildNode() { if(children[0]) delete children[0]; if (children[1]) delete children[1]; }
    void initLeaf(uint first, uint n, const float3& _min, const float3& _max);
    void initInner(uint axis, BVHBuildNode* c0, BVHBuildNode* c1);
};


// Vector math functions that are used in making BVH - possibly put these in geometry.h/cu
float3 offset(const float3& min, const float3& max, const float3& p);
float surfaceArea(const float3& min, const float3& max);
int maximumExtent(const float3& min, const float3& max);
float3 min2(const float3& a, const float3& b);
float3 max2(const float3& a, const float3& b);


// Local functions
void createTriangleBboxes(Scene* p_scene);
void constructBVH(Scene* p_scene);
BVHBuildNode* recursiveBuild(Triangle* p_triangles, vector<TriangleBBox>& trianglesInfo,
     uint start, uint end, uint* totalNodes, vector<Triangle>& orderedTriangles);
int flattenBVHTree(LinearBVHNode* const p_linearNodes, BVHBuildNode* p_node, int* offset);


static vector<TriangleBBox> trianglesInfo;
static vector<Triangle> orderedTriangles;
static const int maxTrisInNode = 3;


void constructBVH(Scene* p_scene) {
    createTriangleBboxes(p_scene);
    uint totalNodes = 0;
    BVHBuildNode* p_bvh = recursiveBuild(p_scene->getTriPtr(), trianglesInfo, 0, trianglesInfo.size(), &totalNodes, orderedTriangles);
    cout << "Total BVH Nodes: " << totalNodes << endl;
    p_scene->allocateBvhArray(totalNodes);
    LinearBVHNode* p_linearBvh = p_scene->getBvhPtr();
    int offset = 0;
    flattenBVHTree(p_linearBvh, p_bvh, &offset);
    Triangle* p_dest = p_scene->getTriPtr();
    std::copy(orderedTriangles.begin(), orderedTriangles.end(), p_dest);

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
        float3 _v1(v1.Position);
        float3 _v2(v2.Position);
        float3 _v3(v3.Position);
        float3 _min = min3(_v1, _v2, _v3);
        float3 _max = max3(_v1, _v2, _v3);
        trianglesInfo.push_back(TriangleBBox(i, _min, _max));
    }
}

void BVHBuildNode::initLeaf(uint first, uint n, const float3& _min, const float3& _max) {
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

BVHBuildNode* recursiveBuild(Triangle* p_triangles, vector<TriangleBBox>& trianglesInfo, uint start, uint end, uint* totalNodes, vector<Triangle>& orderedTriangles) {
    (*totalNodes)++;
    BVHBuildNode* p_node = new BVHBuildNode;
    float3 workingMin = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    float3 workingMax = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
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
        p_node->initLeaf(firstTriOffset, numTriangles, workingMin, workingMax);
    }
    else  {
        float3 centroidMin, centroidMax;
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
            p_node->initLeaf(firstTriOffset, numTriangles, workingMin, workingMax);
            return p_node;
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
				float3 minBound, maxBound;
			};
			BucketInfo buckets[numBuckets];
			// Initialize BucketInfo for SAH partition buckets by determining the bucket
			// that its centroid lies in and updating the bucket's bounds to include the Triangle bounds
			for (uint i = start; i < end; i++) {
				int b = numBuckets * offset(centroidMin, centroidMax, trianglesInfo[i].center)._v[dim];
				if (b == numBuckets) { b = numBuckets - 1; }
				buckets[b].count++;
				buckets[b].minBound = min2(buckets[b].minBound, trianglesInfo[i].min);
				buckets[b].maxBound = max2(buckets[b].maxBound, trianglesInfo[i].max);
			}
			// Compute costs of splitting after each bucket
			float cost[numBuckets-1];
			for (int i = 0; i < numBuckets-1; i++) {
				float3 b0min,b0max, b1min, b1max;
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
				cost[i] = 1.f + (count0 * surfaceArea(b0min, b0max) +
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
						[numBuckets, dim, minCostSplitBucket, centroidMin, centroidMax]
						(const TriangleBBox &t) {
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
	            p_node->initLeaf(firstTriOffset, numTriangles, workingMin, workingMax);
			}
		}
#endif
        p_node->initInner(dim,
                        recursiveBuild(p_triangles, trianglesInfo, mid, end, totalNodes, orderedTriangles),
						recursiveBuild(p_triangles, trianglesInfo, start, mid, totalNodes, orderedTriangles));
    }
    return p_node;
}

int flattenBVHTree(LinearBVHNode* const p_linearNodes, BVHBuildNode* p_node, int* offset) {
	LinearBVHNode* p_linearNode = &p_linearNodes[*offset];
	p_linearNode->min = p_node->min;
	p_linearNode->max = p_node->max;
	int myOffset = (*offset)++;
	if (p_node->numTriangles > 0) {
		p_linearNode->trianglesOffset = p_node->firstTriOffset;
		p_linearNode->numTriangles = p_node->numTriangles;
		p_linearNode->axis = p_node->splitAxis;
	} else {
		flattenBVHTree(p_linearNodes, p_node->children[0], offset);
		p_linearNode->secondChildOffset = flattenBVHTree(p_linearNodes, p_node->children[1], offset);
	}
	return myOffset;
}

// TODO: Consider moving this into geometry.h
float3 offset(const float3& min, const float3& max, const float3& p) {
	float3 retVal = p - min;
	if (max.x > min.x) retVal.x /= (max.x - min.x);
	if (max.y > min.y) retVal.y /= (max.y - min.y);
	if (max.z > min.z) retVal.z /= (max.z - min.z);
	return retVal;
}

float surfaceArea(const float3& min, const float3& max) {
	float3 diag = max - min;
	return diag.x * diag.y * diag.z;
}

int maximumExtent(const float3& min, const float3& max) {
    float3 diag = max - min;
    if (diag.x > diag.y && diag.x > diag.z)
        return 0;
    if (diag.y > diag.z)
        return 1;
    return 2;
}

float3 min2(const float3& a, const float3& b) {
    float x = a.x < b.x ? a.x : b.x;
    float y = a.y < b.y ? a.y : b.y;
    float z = a.z < b.z ? a.z : b.z;
    return float3(x, y, z);
}

float3 max2(const float3& a, const float3& b) {
    float x = a.x > b.x ? a.x : b.x;
    float y = a.y > b.y ? a.y : b.y;
    float z = a.z > b.z ? a.z : b.z;
    return float3(x, y, z);
}
