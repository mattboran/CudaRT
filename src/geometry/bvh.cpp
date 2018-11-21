#include <algorithm>
#include <vector>
#include <cfloat>
#include <string>
#include <assert.h>
#include <stdio.h>
#include <ctime>

#include "bvh.h"
#include "geometry.cuh"
// TODO: Add this to -I
#include "../pathtrace.h"

using namespace std;

// report progress during BVH construction
#define PROGRESS_REPORT
#ifdef PROGRESS_REPORT
#define REPORT(x) x
#define REPORTPRM(x) x,
#else
#define REPORT(x)
#define REPORTPRM(x)
#endif
unsigned g_reportCounter = 0;

BVHNode *g_sceneBVHPtr;

struct BBoxTmp {
	// Bottom point (ie minx,miny,minz)
	Vector3Df _bottom;
	// Top point (ie maxx,maxy,maxz)
	Vector3Df _top;
	// Center point, ie 0.5*(top-bottom)
	Vector3Df _center; // = bbox centroid
	// Triangle
	const Triangle *_pTri;  // triangle list
	BBoxTmp()
		:
		_bottom(FLT_MAX, FLT_MAX, FLT_MAX),
		_top(-FLT_MAX, -FLT_MAX, -FLT_MAX),
		_pTri(NULL)
	{}
};
