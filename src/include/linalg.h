#ifndef __LINEAR_ALGEBRA_H_
#define __LINEAR_ALGEBRA_H_

#include <cuda_runtime.h> // for __host__  __device__
#include <math.h>
#include "obj_load.h"

struct Vector2Df
{
	union {
		struct { float x, y; };
		float _v[2];
	};

	__host__ Vector2Df(const objl::Vector2& v): x(v.X), y(v.Y) {}

	__host__ __device__ Vector2Df(float _x = 0, float _y = 0) : x(_x), y(_y) {}
	__host__ __device__ Vector2Df(const Vector2Df& v) : x(v.x), y(v.y) {}
	inline __host__ __device__ Vector2Df operator*(float a) const{ return Vector2Df(x*a, y*a); }
	inline __host__ __device__ Vector2Df operator/(float a) const{ return Vector2Df(x/a, y/a); }
	inline __host__ __device__ Vector2Df operator*(const Vector2Df& v) const{ return Vector2Df(x * v.x, y * v.y); }
	inline __host__ __device__ Vector2Df operator+(const Vector2Df& v) const{ return Vector2Df(x + v.x, y + v.y); }
	inline __host__ __device__ Vector2Df operator-(const Vector2Df& v) const{ return Vector2Df(x - v.x, y - v.y); }
	inline __host__ __device__ Vector2Df operator-() const { Vector2Df v; v.x = -x; v.y = -y; return v; }
};

struct Vector3Df
{
	union {
		struct { float x, y, z; };
		float _v[3];
	};

	// For interop with obj_load.h
	__host__ Vector3Df(const objl::Vector3& v) : x(v.X), y(v.Y), z(v.Z) {}

	__host__ __device__ Vector3Df(float _x = 0, float _y = 0, float _z = 0) : x(_x), y(_y), z(_z) {}
	__host__ __device__ Vector3Df(const Vector3Df& v) : x(v.x), y(v.y), z(v.z) {}
	__host__ __device__ Vector3Df(const float3& v) : x(v.x), y(v.y), z(v.z) {}
	__host__ __device__ Vector3Df(const float4& v) : x(v.x), y(v.y), z(v.z) {}
	inline __host__ __device__ float length(){ return sqrtf(x*x + y*y + z*z); }
	// sometimes we dont need the sqrt, we are just comparing one length with another
	inline __host__ __device__ float lengthsq(){ return x*x + y*y + z*z; }
	inline __host__ __device__ float lengthsq() const { return x*x + y*y + z*z; }
	inline __host__ __device__ void normalize(){ float norm = sqrtf(x*x + y*y + z*z); x /= norm; y /= norm; z /= norm; }
	inline __host__ __device__ Vector3Df& operator+=(const Vector3Df& v){ x += v.x; y += v.y; z += v.z; return *this; }
	inline __host__ __device__ Vector3Df& operator-=(const Vector3Df& v){ x -= v.x; y -= v.y; z -= v.z; return *this; }
	inline __host__ __device__ Vector3Df& operator*=(const float& a){ x *= a; y *= a; z *= a; return *this; }
	inline __host__ __device__ Vector3Df& operator*=(const Vector3Df& v){ x *= v.x; y *= v.y; z *= v.z; return *this; }
	inline __host__ __device__ Vector3Df operator*(float a) const{ return Vector3Df(x*a, y*a, z*a); }
	inline __host__ __device__ Vector3Df operator/(float a) const{ return Vector3Df(x/a, y/a, z/a); }
	inline __host__ __device__ Vector3Df operator*(const Vector3Df& v) const{ return Vector3Df(x * v.x, y * v.y, z * v.z); }
	inline __host__ __device__ Vector3Df operator+(const Vector3Df& v) const{ return Vector3Df(x + v.x, y + v.y, z + v.z); }
	inline __host__ __device__ Vector3Df operator-(const Vector3Df& v) const{ return Vector3Df(x - v.x, y - v.y, z - v.z); }
	inline __host__ __device__ Vector3Df operator-() const { Vector3Df v; v.x = -x; v.y = -y; v.z = -z; return v; }
	inline __host__ __device__ Vector3Df& operator/=(const float& a){ x /= a; y /= a; z /= a; return *this; }
	inline __host__ __device__ bool operator!=(const Vector3Df& v){ return x != v.x || y != v.y || z != v.z; }
	inline __host__ __device__ bool operator!=(const Vector3Df& v) const{ return x != v.x || y != v.y || z != v.z; }
};

inline __host__ __device__ float4 make_float4(const Vector3Df& v) { return make_float4(v.x, v.y, v.z, 0.0f); }
inline __host__ __device__ float3 make_float3(const Vector3Df& v) { return make_float3(v.x, v.y, v.z); }

inline __host__ __device__ Vector3Df normalize(const Vector3Df& v) { float norm = sqrtf(v.x * v.x + v.y*v.y + v.z*v.z); return Vector3Df(v.x/norm, v.y/norm, v.z/norm);}
inline __host__ __device__ Vector3Df min3(const Vector3Df& v1, const Vector3Df& v2){ return Vector3Df(v1.x < v2.x ? v1.x : v2.x, v1.y < v2.y ? v1.y : v2.y, v1.z < v2.z ? v1.z : v2.z); }
inline __host__ __device__ Vector3Df max3(const Vector3Df& v1, const Vector3Df& v2){ return Vector3Df(v1.x > v2.x ? v1.x : v2.x, v1.y > v2.y ? v1.y : v2.y, v1.z > v2.z ? v1.z : v2.z); }
inline __host__ __device__ float maxComponent(const Vector3Df& v) { float m = v.x > v.y ? v.x : v.y; return m > v.z ? m : v.z; }
inline __host__ __device__ Vector3Df cross(const Vector3Df& v1, const Vector3Df& v2){ return Vector3Df(v1.y*v2.z - v1.z*v2.y, v1.z*v2.x - v1.x*v2.z, v1.x*v2.y - v1.y*v2.x); }
inline __host__ __device__ float dot(const Vector3Df& v1, const Vector3Df& v2){ return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z; }
inline __host__ __device__ float dot(const Vector3Df& v1, const float4& v2){ return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z; }
inline __host__ __device__ float dot(const float4& v1, const Vector3Df& v2){ return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z; }
inline __host__ __device__ float distancesq(const Vector3Df& v1, const Vector3Df& v2){ return (v1.x - v2.x)*(v1.x - v2.x) + (v1.y - v2.y)*(v1.y - v2.y) + (v1.z - v2.z)*(v1.z - v2.z); }
inline __host__ __device__ float distance(const Vector3Df& v1, const Vector3Df& v2){ return sqrtf((v1.x - v2.x)*(v1.x - v2.x) + (v1.y - v2.y)*(v1.y - v2.y) + (v1.z - v2.z)*(v1.z - v2.z)); }
inline __host__ __device__ float clamp(const float& f, const float& mn, const float& mx) { if(f<mn) return mn; if(f>mx) return mx; return f;}
#endif
