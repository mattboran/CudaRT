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

__device__ __host__ Triangle::Triangle(const Triangle &t) :
	_v1(t._v1), _e1(t._e1), _e2(t._e2), _n1(t._n1), _n2(t._n2), _n3(t._n3), _colorDiffuse(t._colorDiffuse), _colorSpec(t._colorSpec), _colorEmit(t._colorEmit), _surfaceArea(t._surfaceArea){}

// NOTE: Do not use these functions as they calculate surface area on the fly and this is not great. Ever more reason texture memory use sucks
__device__ Triangle::Triangle(float3 v1, float3 e1, float3 e2, float3 n1, float3 n2, float3 n3, float3 diff, float3 spec, float3 emit) :
	_v1(v1), _e1(e1), _e2(e2), _n1(n1), _n2(n2), _n3(n3), _colorDiffuse(diff), _colorSpec(spec), _colorEmit(emit)
{
	_surfaceArea = cross(_e1, _e2).length()/2.0f;
}

__device__ Triangle::Triangle(float4 v1, float4 e1, float4 e2, float4 n1, float4 n2, float4 n3, float4 diff, float4 spec, float4 emit) :
	_v1(v1), _e1(e1), _e2(e2), _n1(n1), _n2(n2), _n3(n3), _colorDiffuse(diff), _colorSpec(spec), _colorEmit(emit)
{
	_surfaceArea = cross(_e1, _e2).length()/2.0f;
}

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

__device__ Vector3Df Triangle::getNormal(const RayHit& rh) const {
	// Face normal:  Vector3Df(normalize(_n1 + _n2 + _n3));
	float w = 1 - rh.u - rh.v;
	float u = rh.u;
	float v = rh.v;
	return Vector3Df(normalize(_n1 *w + _n2 *u+ _n3*v));
}

__device__ Vector3Df Triangle::getPointOn(curandState *randState) const {
	float u = curand_uniform(randState);
	float v = curand_uniform(randState);
	if (u + v >= 1.0f) {
		u = 1.0f - u;
		v = 1.0f - v;
	}
	return Vector3Df(_v1 + (_v1-_v2) * v + (_v1-_v3) * u);
}

__device__ bool Triangle::isEmissive() const {
	return _colorEmit.lengthsq() > 0.0f;
}
__device__ bool Triangle::isSpecular() const {
	return _colorSpec.lengthsq() > 0.0f;
}
__device__ bool Triangle::isDiffuse() const {
	return _colorDiffuse.lengthsq() > 0.0f;
}
