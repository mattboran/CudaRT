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

__device__ float Triangle::intersect(const Ray& r, RayHit& rh) const {
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
		rh.u = u;
		rh.v = v;
		return t;
	}
	return MAX_DISTANCE;
}

__device__ Vector3Df Triangle::getNormal(const RayHit& rh) const {
	return Vector3Df(normalize(_n1 + _n2 + _n3));
}
