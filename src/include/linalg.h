#ifndef __LINEAR_ALGEBRA_H_
#define __LINEAR_ALGEBRA_H_

#include <cuda_runtime.h> // for __host__  __device__
#include <math.h>

inline __host__ __device__ float2 make_float2(const float2& v) { return make_float2(v.x, v.y); }
inline __host__ __device__ float3 make_float3(const float4& v) { return make_float3(v.x, v.y, v.z); }
inline __host__ __device__ float4 make_float4(const float3& v) { return make_float4(v.x, v.y, v.z, 0.0f); }

inline __host__ __device__ bool operator==(const float3& a, const float3& b) { return a.x == b.x && a.y == b.y && a.z == b.z; }
inline __host__ __device__ bool operator!=(const float3& a, const float3& b) { return a.x != b.x && a.y != b.y && a.z != b.z; }
inline __host__ __device__ float3 operator+(const float3& a, const float3& b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
inline __host__ __device__ float3 operator-(const float3& a, const float3& b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
inline __host__ __device__ float3 operator*(const float3& a, const float3& b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
inline __host__ __device__ float3 operator/(const float3& a, const float3& b) { return make_float3(a.x / b.x, a.y / b.y, a.z / b.z); }
inline __host__ __device__ float3 operator*(const float3& a, const float& b) { return make_float3(a.x * b, a.y * b, a.z * b); }
inline __host__ __device__ float3 operator/(const float3& a, const float& b) { return make_float3(a.x / b, a.y / b, a.z / b); }
inline __host__ __device__ float3 operator*(const float& b, const float3& a) { return make_float3(a.x * b, a.y * b, a.z * b); }
inline __host__ __device__ float3 operator/(const float& b, const float3& a) { return make_float3(a.x / b, a.y / b, a.z / b); }

inline __host__ __device__ float2 operator+(const float2& a, const float2& b) { return make_float2(a.x + b.x, a.y + b.y); }
inline __host__ __device__ float2 operator-(const float2& a, const float2& b) { return make_float2(a.x - b.x, a.y - b.y); }
inline __host__ __device__ float2 operator*(const float2& a, const float2& b) { return make_float2(a.x * b.x, a.y * b.y); }
inline __host__ __device__ float2 operator/(const float2& a, const float2& b) { return make_float2(a.x / b.x, a.y / b.y); }
inline __host__ __device__ float2 operator*(const float2& a, const float& b) { return make_float2(a.x * b, a.y * b); }
inline __host__ __device__ float2 operator/(const float2& a, const float& b) { return make_float2(a.x / b, a.y / b); }
inline __host__ __device__ float2 operator*(const float& b, const float2& a) { return make_float2(a.x * b, a.y * b); }
inline __host__ __device__ float2 operator/(const float& b, const float2& a) { return make_float2(a.x / b, a.y / b); }

inline __host__ __device__ float3 normalize(const float3& v) { float norm = sqrtf(v.x * v.x + v.y*v.y + v.z*v.z); return make_float3(v.x/norm, v.y/norm, v.z/norm);}
inline __host__ __device__ float3 min3(const float3& v1, const float3& v2){ return make_float3(v1.x < v2.x ? v1.x : v2.x, v1.y < v2.y ? v1.y : v2.y, v1.z < v2.z ? v1.z : v2.z); }
inline __host__ __device__ float3 max3(const float3& v1, const float3& v2){ return make_float3(v1.x > v2.x ? v1.x : v2.x, v1.y > v2.y ? v1.y : v2.y, v1.z > v2.z ? v1.z : v2.z); }
inline __host__ __device__ float maxComponent(const float3& v) { float m = v.x > v.y ? v.x : v.y; return m > v.z ? m : v.z; }
inline __host__ __device__ float3 cross(const float3& v1, const float3& v2){ return make_float3(v1.y*v2.z - v1.z*v2.y, v1.z*v2.x - v1.x*v2.z, v1.x*v2.y - v1.y*v2.x); }
inline __host__ __device__ float dot(const float3& v1, const float3& v2){ return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z; }
inline __host__ __device__ float distancesq(const float3& v1, const float3& v2){ return (v1.x - v2.x)*(v1.x - v2.x) + (v1.y - v2.y)*(v1.y - v2.y) + (v1.z - v2.z)*(v1.z - v2.z); }
inline __host__ __device__ float distance(const float3& v1, const float3& v2){ return sqrtf((v1.x - v2.x)*(v1.x - v2.x) + (v1.y - v2.y)*(v1.y - v2.y) + (v1.z - v2.z)*(v1.z - v2.z)); }
inline __host__ __device__ float lengthsq(const float3& v){ return v.x*v.x + v.y*v.y + v.z*v.z; }
inline __host__ __device__ float length(const float3& v) { return sqrtf(v.x*v.x + v.y*v.y + v.z*v.z); }
inline __host__ __device__ float clamp(const float&v, const float& min, const float& max) { if (v < min) return min; if (v > max) return max; return v;}
#endif
