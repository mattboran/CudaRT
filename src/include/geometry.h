#ifndef __GEOMETRY_H_
#define __GEOMETRY_H_

#include "linalg.h"

#include <cfloat>
#include <cuda.h>

#define EPSILON 0.00001f
#define EPSILON_2 0.000005f

struct Sampler;

Vector3Df min3(const Vector3Df& a, const Vector3Df& b, const Vector3Df& c);
Vector3Df max3(const Vector3Df& a, const Vector3Df& b, const Vector3Df& c);

struct Ray {
	Vector3Df origin;
	Vector3Df dir;
	float tMin = EPSILON;
	float tMax = FLT_MAX;

	__host__ __device__ Ray(Vector3Df o, Vector3Df d) : origin(o), dir(normalize(d)) { }
};


struct Triangle {
	// Triangle ID
	unsigned _triId;
	unsigned _materialId;

	// Vertex indices will be used for intersection soon (see github issues)
	unsigned _id1, _id2, _id3;
	// Unoptimized triangles for moller-trombore
	Vector3Df _v1;
	Vector3Df _e1, _e2;
	Vector3Df _n1, _n2, _n3;
	Vector2Df _uv1, _uv2, _uv3;
	float _surfaceArea = 0.0f;
	__device__ __host__ Triangle() {}
	__device__ __host__ Triangle(const Triangle &t);
	__host__ __device__ float intersect(const Ray &r, float &o_u, float &o_v) const;
	__host__ __device__ Vector3Df getNormal(const float u, const float v) const;
	__host__ __device__ Vector3Df getRandomPointOn(Sampler* p_sampler) const;
	// Raytracing intersection pre-computed cache:
//		float _d, _d1, _d2, _d3;
//		Vector3Df _e1, _e2, _e3;
} __attribute__ ((aligned (128))) ;

struct SurfaceInteraction {
	Triangle* p_hitTriangle = NULL;
	Vector3Df position;
	Vector3Df normal;
	Vector3Df inputDirection;
	Vector3Df outputDirection;
	float pdf, u, v, t;
};

#endif
