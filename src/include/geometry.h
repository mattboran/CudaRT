#ifndef __GEOMETRY_H_
#define __GEOMETRY_H_

#include "linalg.h"

#include <cfloat>
#include <cuda.h>

#define EPSILON 0.00001f
#define EPSILON_2 0.000005f

struct Sampler;

float3 min3(const float3& a, const float3& b, const float3& c);
float3 max3(const float3& a, const float3& b, const float3& c);

struct Ray {
	float3 origin;
	float3 dir;
	float tMin = EPSILON;
	float tMax = FLT_MAX;

	__host__ __device__ Ray(float3 o, float3 d) {
		origin = o;
		dir = d;
	}
};


struct Triangle {
	// Triangle ID
	unsigned _triId;
	unsigned _materialId;

	// Vertex indices will be used for intersection soon (see github issues)
//	unsigned _id1, _id2, _id3;
	// Unoptimized triangles for moller-trombore
	float3 _v1;
	float3 _e1, _e2;
	float3 _n1, _n2, _n3;
	float2 _uv1, _uv2, _uv3;
	float _surfaceArea = 0.0f;
	__device__ __host__ Triangle() {}
	__device__ __host__ Triangle(const Triangle &t);
	__host__ __device__ float intersect(const Ray &r, float &o_u, float &o_v) const;
	__host__ __device__ float3 getNormal(const float u, const float v) const;
	__host__ __device__ float3 getRandomPointOn(Sampler* p_sampler) const;
	// Raytracing intersection pre-computed cache:
//		float _d, _d1, _d2, _d3;
//		Vector3Df _e1, _e2, _e3;
} __attribute__ ((aligned (128))) ;

struct SurfaceInteraction {
	float3 position;
	float3 normal;
	float3 inputDirection;
	float3 outputDirection;
	float pdf = 1.0f;
	float u, v, t;
	uint hitTriIdx;
};

#endif
