#include "geometry.h"
#include "renderer.h"
#include <cfloat>
#include <math.h>

using std::max;
using std::min;

float3 min3(const float3& a, const float3& b,
		const float3& c) {
	float x = min(a.x, min(b.x, c.x));
	float y = min(a.y, min(b.y, c.y));
	float z = min(a.z, min(b.z, c.z));
	return make_float3(x, y, z);
}

float3 max3(const float3& a, const float3& b,
		const float3& c) {
	float x = max(a.x, max(b.x, c.x));
	float y = max(a.y, max(b.y, c.y));
	float z = max(a.z, max(b.z, c.z));
	return make_float3(x, y, z);
}

__host__ __device__ Triangle::Triangle(const Triangle &t) :
		_surfaceArea(t._surfaceArea), _triId(t._triId), _materialId(t._materialId) {
	_v1 = t._v1;
	_e1 = t._e1;
	_e2 = t._e2;
	_n1 = t._n1;
	_n2 = t._n2;
	_n3 = t._n3;
	_uv1 = t._uv1;
	_uv2 = t._uv2;
	_uv3 = t._uv3;
}

__device__ float Triangle::intersect(const Ray& r, float &o_u, float &o_v) const {
	float3 P, Q, T;
	P = cross(r.dir, _e2);
	float det = dot(_e1, P);

	if (fabsf(det) < EPSILON) {
		return FLT_MAX;
	}
	float inv_det = 1.f / det;
	T = r.origin - _v1;
	float u = dot(T, P) * inv_det;
	if (u < 0.f || u > 1.f) {
		return FLT_MAX;
	}

	Q = cross(T, _e1);
	float v = dot(r.dir, Q) * inv_det;
	if (v < 0.f || u + v > 1.f) {
		return FLT_MAX;
	}
	float t = dot(_e2, Q) * inv_det;
	if (t > EPSILON) {
		o_u = u;
		o_v = v;
		return t;
	}
	return FLT_MAX;
}

__host__ __device__ float3 Triangle::getNormal(const float u, const float v) const {
	// Face normal:
	float w = 1.f - u - v;
	return normalize(_n1*w + _n2*u + _n3*v);
}

__host__ __device__ float3 Triangle::getRandomPointOn(Sampler* p_sampler) const {
	float u = p_sampler->getNextFloat();
	float v = p_sampler->getNextFloat();
	if (u + v >= 1.0f) {
		u = 1.0f - u;
		v = 1.0f - v;
	}
	return _v1 + _e1 * u + _e2 * v;
}
