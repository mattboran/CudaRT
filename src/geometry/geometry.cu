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
