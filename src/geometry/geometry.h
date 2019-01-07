#ifndef __GEOMETRY_H_
#define __GEOMETRY_H_

#include "linalg.h"
#include <cfloat>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <curand_kernel.h>
// #include <limits>
#include <iostream>

#define EPSILON 0.00001f

Vector3Df min3(const Vector3Df& a, const Vector3Df& b, const Vector3Df& c);
Vector3Df max3(const Vector3Df& a, const Vector3Df& b, const Vector3Df& c);

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
		float tMin = EPSILON;
		float tMax = FLT_MAX;

		__host__ __device__ Ray(Vector3Df o, Vector3Df d) : origin(o), dir(normalize(d)) { }
		__device__ Vector3Df pointAlong(float t) { return Vector3Df(origin + dir*t); }
	};

	struct RayHit;

	struct Triangle {
		// Triangle ID
		unsigned _triId;
		// RGB Color Vector3Df
		Vector3Df _colorDiffuse;
		Vector3Df _colorSpec;
		Vector3Df _colorEmit;

		// Vertex indices will be used for intersection soon (see github issues)
		unsigned _id1, _id2, _id3;
		// Unoptimized triangles for moller-trombore
		Vector3Df _v1;
		Vector3Df _e1, _e2;
		Vector3Df _n1, _n2, _n3;
		float _surfaceArea = 0.0f;
		__device__ __host__ Triangle() {}
		__device__ __host__ Triangle(const Triangle &t);
		__host__ __device__ float intersect(const Ray &r, float &_u, float &_v) const;
		__host__ __device__ Vector3Df getNormal(const  RayHit& rh) const;
		__host__ Vector3Df getRandomPointOn() const;
		__device__ Vector3Df getRandomPointOn(curandState *randState) const;
		__host__ __device__ bool isEmissive() const;
		__host__ __device__ bool isDiffuse() const;
		// TODO: Implement these properties
		// Center point
		Vector3Df _center;
		Vector3Df _bottom;
		Vector3Df _top;

		// Raytracing intersection pre-computed cache:
//		float _d, _d1, _d2, _d3;
//		Vector3Df _e1, _e2, _e3;
	} __attribute__ ((aligned (128))) ;

	struct RayHit {
		Triangle* p_hitTriangle = NULL;
		float u, v, t;
	}__attribute__((aligned (32)));

	struct SurfaceInteraction {
		Triangle* p_hitTriangle = NULL;
		Vector3Df position;
		Vector3Df normal;
		Vector3Df inputDirection;
		Vector3Df outputDirection;
		float pdf, u, v, t;
	};
}
#endif
