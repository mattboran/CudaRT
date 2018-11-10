#ifndef __GEOMETRY_H_
#define __GEOMETRY_H_

#include "linalg.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <limits>

#define FLT_MAX std::numeric_limits<float>::max()
#define MAX_DISTANCE 100000000.0f
#define FLT_MIN std::numeric_limits<float>::min()
#define EPSILON 0.00001f


Vector3Df max4(const Vector3Df& a, const Vector3Df& b, const Vector3Df& c, const Vector3Df& d);
Vector3Df min4(const Vector3Df& a, const Vector3Df& b, const Vector3Df& c, const Vector3Df& d);

namespace geom {

	struct Vertex : public Vector3Df {
		// normal vector of this vertex
		Vector3Df _normal;

		Vertex(float x, float y, float z, float nx, float ny, float nz)
			:
			Vector3Df(x, y, z), _normal(Vector3Df(nx, ny, nz))
		{ }
	};

	struct Ray {
		Vector3Df origin;
		Vector3Df dir;

		__device__ Ray(Vector3Df o, Vector3Df d) : origin(o), dir(normalize(d)) { }
		__device__ Vector3Df pointAlong(float t) { return Vector3Df(origin + dir*t); }
	};

	struct RayHit;

	struct Triangle {
		unsigned meshId;
		// RGB Color Vector3Df
		Vector3Df _colorDiffuse;
		Vector3Df _colorSpec;
		Vector3Df _colorEmit;
		// triangle normal
		Vector3Df _normal;
		// Unoptimized triangles for moller-trombore
		Vector3Df _v1, _v2, _v3;
		Vector3Df _e1, _e2;
		Vector3Df _n1, _n2, _n3;

		__device__ float intersect(const Ray &r, float &_u, float &_v) const;
		__device__ Vector3Df getNormal(const  RayHit& rh) const;
		// TODO: Implement these properties
		// Center point
//		Vector3Df _center;
		// ignore back-face culling flag
//		bool _twoSided;

		// Raytracing intersection pre-computed cache:
//		float _d, _d1, _d2, _d3;
//		Vector3Df _e1, _e2, _e3;
		// bounding box
//		Vector3Df _bottom;
//		Vector3Df _top;
	} __attribute__ ((aligned (64))) ;

	struct RayHit {
		Triangle* hitTriPtr;
		float u, v;
	};
}
#endif
