#include "cuda_textures.cuh"
#include "cuda_error_check.h"
#include <cuda.h>

using namespace geom;

extern void configureTexture(texture_t &triTexture) {
	triTexture.addressMode[0] = cudaAddressModeBorder;
	triTexture.addressMode[1] = cudaAddressModeBorder;
	triTexture.filterMode = cudaFilterModePoint;
	triTexture.normalized = false;
}

// Note, this function returns a pointer to the new triangle data
__host__ cudaArray* bindTrianglesToTexture(Triangle* triPtr, unsigned numTris, texture_t &triTexture) {
	// 3 float3s for verts, 3 float3s for normals, 3 float3s for material
	unsigned numElements = numTris * 9;
	float4* triDataPtr = new float4[numElements];
	// TODO Replace this with cudaArray
	for (unsigned i = 0; i < numTris; i++) {
		triDataPtr[i*9].x = triPtr->_v1.x;
		triDataPtr[i*9].y = triPtr->_v1.y;
		triDataPtr[i*9].z = triPtr->_v1.z;
		triDataPtr[i*9+1].x = triPtr->_e1.x;
		triDataPtr[i*9+1].y = triPtr->_e1.y;
		triDataPtr[i*9+1].z = triPtr->_e1.z;
		triDataPtr[i*9+2].x = triPtr->_e2.x;
		triDataPtr[i*9+2].y = triPtr->_e2.y;
		triDataPtr[i*9+2].z = triPtr->_e2.z;
		triDataPtr[i*9+3].x = triPtr->_n1.x;
		triDataPtr[i*9+3].y = triPtr->_n1.y;
		triDataPtr[i*9+3].z = triPtr->_n1.z;
		triDataPtr[i*9+4].x = triPtr->_n2.x;
		triDataPtr[i*9+4].y = triPtr->_n2.y;
		triDataPtr[i*9+4].z = triPtr->_n2.z;
		triDataPtr[i*9+5].x = triPtr->_n3.x;
		triDataPtr[i*9+5].y = triPtr->_n3.y;
		triDataPtr[i*9+5].z = triPtr->_n3.z;
		triDataPtr[i*9+6].x = triPtr->_colorDiffuse.x;
		triDataPtr[i*9+6].y = triPtr->_colorDiffuse.y;
		triDataPtr[i*9+6].z = triPtr->_colorDiffuse.z;
		triDataPtr[i*9+7].x = triPtr->_colorSpec.x;
		triDataPtr[i*9+7].y = triPtr->_colorSpec.y;
		triDataPtr[i*9+7].z = triPtr->_colorSpec.z;
		triDataPtr[i*9+8].x = triPtr->_colorEmit.x;
		triDataPtr[i*9+8].y = triPtr->_colorEmit.y;
		triDataPtr[i*9+8].z = triPtr->_colorEmit.z;
	}

	cudaArray* cuArray;
	cudaChannelFormatDesc channelDesc =
			cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat );
	CUDA_CHECK_RETURN(cudaMallocArray(&cuArray, &channelDesc, numElements));
	CUDA_CHECK_RETURN(cudaMemcpyToArray(cuArray, 0, 0, triDataPtr, sizeof(float4)*numElements, cudaMemcpyHostToDevice));

	CUDA_CHECK_RETURN(cudaBindTextureToArray(triTexture, cuArray, channelDesc));
	return cuArray;
}
