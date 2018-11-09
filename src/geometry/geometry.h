#ifndef __GEOMETRY_H_
#define __GEOMETRY_H_

#include "linalg.h"

struct Vertex : public Vector3Df
{
	// normal vector of this vertex
	Vector3Df _normal;
	// ambient occlusion of this vertex (pre-calculated in e.g. MeshLab)
	float _ambientOcclusionCoeff;

	Vertex(float x, float y, float z, float nx, float ny, float nz, float amb = 60.f)
		:
		Vector3Df(x, y, z), _normal(Vector3Df(nx, ny, nz)), _ambientOcclusionCoeff(amb)
	{
		// assert |nx,ny,nz| = 1
	}
};

struct Triangle {
	// indexes in vertices array
	unsigned _idx1;
	unsigned _idx2;
	unsigned _idx3;
	// RGB Color Vector3Df
	Vector3Df _colorf;
	// Center point
	Vector3Df _center;
	// triangle normal
	Vector3Df _normal;
	// ignore back-face culling flag
	bool _twoSided;
	// Raytracing intersection pre-computed cache:
	float _d, _d1, _d2, _d3;
	Vector3Df _e1, _e2, _e3;
	// bounding box
	Vector3Df _bottom;
	Vector3Df _top;
};



#endif
