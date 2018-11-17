#ifndef CUDA_TEXTURES_CUH
#define CUDA_TEXTURES_CUH

#include "geometry.cuh"

#include <cuda.h>
#include <cuda_runtime_api.h>

#define TEX_ARRAY_MAX 8192
typedef texture<float4, 1, cudaReadModeElementType> texture_t;

void configureTexture(texture_t &triTexture);
__host__ cudaArray* bindTrianglesToTexture(geom::Triangle* triPtr, unsigned numTris, texture_t &triTexture);
__device__ geom::Triangle getTriangleFromTexture(unsigned i) ;

#endif
