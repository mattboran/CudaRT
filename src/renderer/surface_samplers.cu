/*
 * surface_samplers.cu
 *
 *  Created on: Mar 18, 2019
 *      Author: matt
 */

#include "surface_samplers.h"

__host__ __device__ float3 randomPointOnHemisphere(const float e, Sampler* p_sampler);
__host__ __device__ float3 reflect(const float3& incedent, const float3& normal);

__host__ __device__ float3 randomPointOnHemisphere(const float e, Sampler* p_sampler) {
	float phi = 2.0f * M_PI * p_sampler->getNextFloat();
	float cosPhi = cosf(phi);
	float sinPhi = sinf(phi);
	float theta = 1.0f - p_sampler->getNextFloat();
	float cosTheta = powf((theta), (1.f/(e + 1.f)));
	float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);
	float pu = sinTheta * cosPhi;
	float pv = sinTheta * sinPhi;
	float pw = cosTheta;
	return make_float3(pu, pv, pw);
}

__host__ __device__ float3 sampleDiffuseBSDF(SurfaceInteraction* p_interaction,
												Triangle* p_hitTriangle,
												const float3& diffuseColor,
												dataPtr_t p_textureContainer,
												Sampler* p_sampler) {
   float r1 = 2.0f * M_PI * p_sampler->getNextFloat();
   float r2 = p_sampler->getNextFloat();
   float r2sq = sqrtf(r2);
   // calculate orthonormal coordinates u, v, w, at hitpt
   float3 w = p_interaction->normal;
   float3 u = normalize(cross( (fabs(w.x) > 0.1f ?
			   make_float3(0.f, 1.f, 0.f) :
			   make_float3(1.f, 0.f, 0.f)), w));
   float3 v = cross(w, u);
   p_interaction->inputDirection = normalize(u * cosf(r1) * r2sq + v * sinf(r1) * r2sq + w * sqrtf(1.f - r2));
   p_interaction->pdf = 0.5f;

   float3 kd = diffuseColor;
   if (p_textureContainer != NULL) {
	   float u = p_interaction->u;
	   float v = p_interaction->v;
	   float w = 1.f - u - v;
	   float2 uv = p_hitTriangle->_uv1 * w + p_hitTriangle->_uv2 * u + p_hitTriangle->_uv3 * v;
	   kd = sampleTexture(p_textureContainer, uv.x, uv.y);
   }
   return kd;
}

__host__ __device__ float3 sampleSpecularBSDF(SurfaceInteraction* p_interaction, const float3& specularColor) {
	p_interaction->inputDirection = reflect(p_interaction->outputDirection,  p_interaction->normal);
	p_interaction->pdf = 1.0f;
	return specularColor;
}

__host__ __device__ float3 sampleGlossyReflectorBSDF(SurfaceInteraction* p_interaction, const float exp, const float3& specularColor, Sampler* p_sampler) {
	float3 w = reflect(p_interaction->outputDirection,  p_interaction->normal);
	float3 u = normalize(cross( (fabs(w.x) > 0.1f ?
			   make_float3(0.f, 1.f, 0.f) :
			   make_float3(1.f, 0.f, 0.f)), w));
	float3 v = cross(w, u);
	float3 sp = randomPointOnHemisphere(exp, p_sampler);
	float3 wi = sp.x * u + sp.y * v + sp.z * w;
	if (dot(p_interaction->normal, wi) < 0.0) // reflected ray is below surface
		wi = -sp.x * u - sp.y * v + sp.z * w;
	float phong_lobe = powf(dot(w,  wi), exp);
	p_interaction->inputDirection = wi;
	p_interaction->pdf = phong_lobe * (dot(w, wi));
	return specularColor * phong_lobe * 0.9f;
}

__host__ __device__ float3 reflect(const float3& incedent, const float3& normal) {
	return incedent - normal * dot(incedent, normal) * 2.f;
}

__host__ __device__ float3 estimateDirectLighting(Triangle* p_light,
													 uint lightIdx,
													 SceneData* p_sceneData,
													 const float3& lightColor,
													 const float lightsSurfaceArea,
													 const SurfaceInteraction &interaction,
													 Sampler* p_sampler) {
	if (interaction.hitTriIdx == lightIdx) {
		return make_float3(0.0f, 0.0f, 0.0f);
	}
	float3 directLighting = make_float3(0.0f, 0.0f, 0.0f);
	float3 rayOrigin = interaction.position + interaction.normal * EPSILON;
	Ray ray(rayOrigin,  normalize(p_light->getRandomPointOn(p_sampler) - interaction.position));
	SurfaceInteraction lightInteraction;
	// Sample the light
#ifdef __CUDA_ARCH__
	dataPtr_t p_bvh = (dataPtr_t)p_sceneData->p_cudaTexObjects;
#else
	dataPtr_t p_bvh = (dataPtr_t)p_sceneData->p_bvh;
#endif
	bool intersectsLight = intersectBVH(p_bvh,
			p_sceneData->p_triangles, lightInteraction, ray);
	if (intersectsLight && lightInteraction.hitTriIdx == lightIdx) {
		float surfaceArea = p_light->_surfaceArea;
		float distanceSquared = ray.tMax * ray.tMax;
		// For directional lights also consider light direction
		float cosTheta = fabs(dot(p_light->getNormal(lightInteraction.u, lightInteraction.v), ray.dir * -1.0f));
		float bsdfCosTheta = fabs(dot(interaction.normal, ray.dir));
		float weightFactor = surfaceArea/(distanceSquared * lightsSurfaceArea)* cosTheta;
		directLighting = lightColor * weightFactor * bsdfCosTheta;
	}
	return directLighting;
}

__host__ __device__ Fresnel getFresnelReflectance(SurfaceInteraction* p_interaction, const float ior, float3& transmittedDir) {
	Fresnel fresnel;
	float3 incedent = p_interaction->outputDirection;
	float3 normal = p_interaction->normal;
	float cosi = dot(incedent, normal);
	float etai = 1, etat = ior;
	float3 n = normal;
	if (cosi < 0.0f) {
		cosi = -cosi;
	} else {
		float temp = etai;
		etai = etat;
		etat = temp;
		n = normal * -1.0f;
	}
	float eta = etai / etat;
	float k = 1 - eta * eta * (1 - cosi * cosi);
	if (k < 0) {
		fresnel.probReflection = 1.0f;
		fresnel.probTransmission = 0.0f;
	} else {
		transmittedDir = incedent * eta + n * (eta * cosi - sqrtf(k));
		float R0 = (etai - etat) * (etai - etat) / (etat + etai) * (etat + etai);
		float c = 1.f - dot(transmittedDir, normal);
		float Re = R0 + (1.f - R0) * c * c * c * c * c;
		fresnel.probReflection = Re;
		fresnel.probTransmission = 1.f - Re;
	}
	return fresnel;
}

__host__ __device__ float3 sampleTexture(dataPtr_t p_textureContainer,  float u, float v) {
#ifdef __CUDA_ARCH__
	float4 texValue = tex2D<float4>(*(cudaTextureObject_t*)p_textureContainer, u, v);
	return make_float3(texValue);
#else
	TextureContainer* p_texContainer = (TextureContainer*)p_textureContainer;
	pixels_t* p_texDimensions = p_texContainer->p_textureDimensions;
	pixels_t width = p_texDimensions[0];
	pixels_t height = p_texDimensions[1];
	float pixelCoordU = u * (float)width;
	float pixelCoordV = v * (float)height;
	pixels_t i = truncf(pixelCoordU);
	pixels_t j = truncf(pixelCoordV);

	float floorPixelCoordU = floorf(pixelCoordU);
	float floorPixelCoordV = floorf(pixelCoordV);
	float ceilPixelCoordU = ceilf(pixelCoordU);
	float ceilPixelCoordV = ceilf(pixelCoordV);

	// Bilinear interpolation
	float3* p_texture = p_texContainer->p_textureData;
	float3 valA1 = p_texture[j * width + i + 1] * (pixelCoordU - floorPixelCoordU);
	float3 valA2 = p_texture[j * width + i] * (ceilPixelCoordU - pixelCoordU);
	float3 valB1 = p_texture[(j + 1) * width + i + 1] * (pixelCoordU - floorPixelCoordU);
	float3 valB2 = p_texture[(j + 1) * width + i] * (ceilPixelCoordU - pixelCoordU);

	float3 valC1 = (valA1 + valA2) * (ceilPixelCoordV - pixelCoordV);
	float3 valC2 = (valB1 + valB2) * (pixelCoordV - floorPixelCoordV);
	return valC1 + valC2;
#endif
}

