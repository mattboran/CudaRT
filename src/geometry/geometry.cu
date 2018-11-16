#include "geometry.cuh"
#include <math.h>

using namespace geom;
using std::max;
using std::min;

extern Vector3Df max4(const Vector3Df& a, const Vector3Df& b, const Vector3Df& c, const Vector3Df& d) {
	float x = max(a.x, max(b.x, max(c.x, d.x)));
	float y = max(a.y, max(b.y, max(c.y, d.y)));
	float z = max(a.z, max(b.z, max(c.z, d.z)));
	return Vector3Df(x,y,z);
}

extern Vector3Df min4(const Vector3Df& a, const Vector3Df& b, const Vector3Df& c, const Vector3Df& d) {
	float x = min(a.x, min(b.x, min(c.x, d.x)));
	float y = min(a.y, min(b.y, min(c.y, d.y)));
	float z = min(a.z, min(b.z, min(c.z, d.z)));
	return Vector3Df(x,y,z);
}

__device__ Triangle::Triangle(const float* data, unsigned i) {
	_v1 = Vector3Df(data[i*12], data[i*12 + 1], data[i*12+2]);
	_e1 = Vector3Df(data[i*12 + 3], data[i*12 + 4], data[i*12 + 5]);
	_e1 = Vector3Df(data[i*12 + 6], data[i*12 + 7], data[i*12 + 8]);
	_n1 = Vector3Df(data[i*12 + 9], data[i*12 + 10], data[i*12 + 11]);
	_n2 = Vector3Df(data[i*12 + 12], data[i*12 + 13], data[i*12 + 14]);
	_n3 = Vector3Df(data[i*12 + 15], data[i*12 + 16], data[i*12 + 17]);
	_colorDiffuse = Vector3Df(data[i*12 + 18], data[i*12 + 19], data[i*12 + 20]);
	_colorSpec = Vector3Df(data[i*12 + 21], data[i*12 + 22], data[i*12 + 23]);
	_colorEmit = Vector3Df(data[i*12 + 24], data[i*12 + 25], data[i*12 + 26]);
}

__device__ Triangle::Triangle(float3 v1, float3 e1, float3 e2, float3 n1, float3 n2, float3 n3, float3 diff, float3 spec, float3 emit) :
	_v1(v1), _e1(e1), _e2(e2), _n1(n1), _n2(n2), _n3(n3), _colorDiffuse(diff), _colorSpec(spec), _colorEmit(emit)
		{ }

__device__ Triangle::Triangle(float4 v1, float4 e1, float4 e2, float4 n1, float4 n2, float4 n3, float4 diff, float4 spec, float4 emit) :
	_v1(v1), _e1(e1), _e2(e2), _n1(n1), _n2(n2), _n3(n3), _colorDiffuse(diff), _colorSpec(spec), _colorEmit(emit)
		{ }

__device__ float Triangle::intersect(const Ray& r, float &_u, float &_v) const {
	Vector3Df P, Q, T;
	P = cross(r.dir, _e2);
	float det = dot(_e1, P);

	if(fabsf(det) < EPSILON)
	{
		return MAX_DISTANCE;
	}
	float inv_det = 1.f / det;
	T = r.origin - _v1;
	float u = dot(T, P) * inv_det;
	if ( u < 0.f || u > 1.f)
	{
		return MAX_DISTANCE;
	}

	Q = cross(T, _e1);
	float v = dot(r.dir, Q) * inv_det;
	if ( v < 0.f || u + v > 1.f)
	{
		return MAX_DISTANCE;
	}
	float t = dot(_e2, Q) * inv_det;
	if ( t > EPSILON)
	{
		_u = u;
		_v = v;
		return t;
	}
	return MAX_DISTANCE;
}

__device__ bool equals(const Vector3Df &v1, const Vector3Df &v2, const Vector3Df &v3) {
	if (v1.x != v2.x && v2.x != v3.x) {
		return false;
	}
	if (v1.y != v2.y && v2.y != v3.z) {
		return false;
	}
	if (v1.z != v2.z && v2.y != v3.z) {
		return false;
	}
	return true;
}
__device__ Vector3Df Triangle::getNormal(const RayHit& rh) const {
	// Face normal:  Vector3Df(normalize(_n1 + _n2 + _n3));
	float w = 1 - rh.u - rh.v;
	float u = rh.u;
	float v = rh.v;
	return Vector3Df(normalize(_n1 *w + _n2 *u+ _n3*v));
}
