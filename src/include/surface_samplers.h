/*
 * surface_samplers.h
 *
 *  Created on: Mar 18, 2019
 *      Author: matt
 */

#include "renderer.h"

#ifndef SURFACE_SAMPLERS_H_
#define SURFACE_SAMPLERS_H_

struct Fresnel {
	float probReflection;
	float probTransmission;
};

__host__ __device__ Fresnel getFresnelReflectance(SurfaceInteraction* p_interaction, const float ior, float3& transmittedDir);
__host__ __device__ float3 sampleDiffuseBSDF(SurfaceInteraction* p_interaction,
												Triangle* p_hitTriangle,
												const float3& diffuseColor,
												dataPtr_t p_textureContainer,
												Sampler* p_sampler);
__host__ __device__ float3 sampleSpecularBSDF(SurfaceInteraction* p_interaction,
												 const float3& specularColor);
__host__ __device__ float3 sampleGlossyReflectorBSDF(SurfaceInteraction* p_interaction,
											const float exp,
											const float3& specularColor,
											Sampler* p_sampler);
__host__ __device__ float3 sampleTexture(dataPtr_t p_textureContainer,  float u, float v) ;


#endif /* SURFACE_SAMPLERS_H_ */
