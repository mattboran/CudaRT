/*
 * renderer.cpp

 *
 *  Created on: Dec 23, 2018
 *      Author: matt
 *
 *  This file contains functions that both the sequential
 *  and parallel renderers use
 */

#include "renderer.h"

#include <float.h>
#include <math.h>
#include <iostream>

#define USE_BVH
#define USE_SKYBOX
#define UNBIASED

#ifdef __CUDA_ARCH__
#define TEXTURE_CONTAINER_FACTORY_ARGUMENTS cudaTextureObject_t* p_texObject
#define TEXTURE_CONTAINER_FACTOR_PARAMETERS(p_sceneData) p_sceneData->p_cudaTexObjects
__constant__ uint c_maxBounces = 6;
#else
#define TEXTURE_CONTAINER_FACTORY_ARGUMENTS Vector3Df* p_texData, \
											pixels_t* p_texDimensions, \
											pixels_t* p_texOffsets
#define TEXTURE_CONTAINER_FACTOR_PARAMETERS(p_sceneData) p_sceneData->p_textureData, \
														 p_sceneData->p_textureDimensions, \
														 p_sceneData->p_textureOffsets
static const uint c_maxBounces = 6;
static pixels_t c_width;
static pixels_t c_height;
static uint c_samples;
#endif

typedef void* dataPtr_t;

struct Fresnel {
	float probReflection;
	float probTransmission;
};

__host__ __device__ bool intersectTriangles(Triangle* p_triangles, int numTriangles, SurfaceInteraction &interaction, Ray& ray);
__host__ __device__ bool rayIntersectsBox(Ray& ray, const Vector3Df& min, const Vector3Df& max);
__host__ __device__ Vector3Df sampleDiffuseBSDF(SurfaceInteraction* p_interaction,
												Triangle* p_hitTriangle,
												const Vector3Df& diffuseColor,
												dataPtr_t p_textureContainer,
												Sampler* p_sampler);
__host__ __device__ Vector3Df sampleSpecularBSDF(SurfaceInteraction* p_interaction,
												 const Vector3Df& specularColor);
__host__ __device__ Vector3Df estimateDirectLighting(Triangle* p_light,
													 SceneData* p_sceneData,
													 const Vector3Df& lightColor,
													 const SurfaceInteraction &interaction,
													 Sampler* p_sampler);
__host__ __device__ bool intersectBVH(dataPtr_t p_bvhData, Triangle* p_triangles, SurfaceInteraction &interaction, Ray& ray);
__host__ __device__ Vector3Df reflect(const Vector3Df& incedent, const Vector3Df& normal);
__host__ __device__ Fresnel getFresnelReflectance(const SurfaceInteraction& interaction, const float ior, Vector3Df& transmittedDir);
__host__ __device__ dataPtr_t textureContainerFactory(int i, TEXTURE_CONTAINER_FACTORY_ARGUMENTS);

#ifdef USE_SKYBOX
#ifdef __CUDA_ARCH__
__constant__ float skybox[] = { 0.0f, 0.025f, 0.05f };
#else
static const float skybox[] = { 0.0f, 0.025f, 0.05f };
#endif
#endif

__host__ __device__ float Sampler::getNextFloat() {
	#ifdef __CUDA_ARCH__
		return curand_uniform(p_curandState);
	#else
		return rand() / (float)RAND_MAX;
	#endif
}

__host__ Renderer::Renderer(Scene* _scenePtr, pixels_t _width, pixels_t _height, uint _samples) :
	p_scene(_scenePtr), width(_width), height(_height), samples(_samples)
{
	h_imgPtr = new uchar4[width*height]();
	samplesRendered = 0;
}

__host__ void Renderer::createSceneData(SceneData* p_sceneData,
										Triangle* p_triangles,
										LinearBVHNode* p_bvh,
										Vector3Df* p_textureData,
										pixels_t* p_textureDimensions,
										pixels_t* p_textureOffsets) {
	// Note: cudaTextureObjects are assigned in copyMemoryToCuda for ParallelRenderer
	p_sceneData->p_triangles = p_triangles;
#ifndef __CUDA_ARCH__
	p_sceneData->p_bvh = p_bvh;
	p_sceneData->p_textureData = p_textureData;
	p_sceneData->p_textureDimensions = p_textureDimensions;
	p_sceneData->p_textureOffsets = p_textureOffsets;
	p_sceneData->numBVHNodes = p_scene->getNumBvhNodes();
	p_sceneData->numTextures = p_scene->getNumTextures();
#endif
}

__host__ void Renderer::createLightsData(LightsData* p_lightsData, Triangle* p_triangles) {
	p_lightsData->lightsPtr = p_triangles;
	p_lightsData->numLights = p_scene->getNumLights();
	p_lightsData->totalSurfaceArea = p_scene->getLightsSurfaceArea();
}

__host__ __device__ Vector3Df sampleTexture(dataPtr_t p_textureContainer,  float u, float v) {
#ifdef __CUDA_ARCH__
	float4 texValue = tex2D<float4>(*(cudaTextureObject_t*)p_textureContainer, u, v);
	return Vector3Df(texValue);
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
	Vector3Df* p_texture = p_texContainer->p_textureData;
	Vector3Df valA1 = p_texture[j * width + i + 1] * (pixelCoordU - floorPixelCoordU);
	Vector3Df valA2 = p_texture[j * width + i] * (ceilPixelCoordU - pixelCoordU);
	Vector3Df valB1 = p_texture[(j + 1) * width + i + 1] * (pixelCoordU - floorPixelCoordU);
	Vector3Df valB2 = p_texture[(j + 1) * width + i] * (ceilPixelCoordU - pixelCoordU);

	Vector3Df valC1 = (valA1 + valA2) * (ceilPixelCoordV - pixelCoordV);
	Vector3Df valC2 = (valB1 + valB2) * (pixelCoordV - floorPixelCoordV);
	return valC1 + valC2;
#endif
}

__host__ __device__ Vector3Df samplePixel(int x, int y,
										  Camera camera,
										  SceneData* p_sceneData,
										  LightsData *p_lightsData,
										  Sampler* p_sampler,
                                          float3* p_matFloats,
                                          int2* p_matIndices) {
	Ray ray = camera.computeCameraRay(x, y, p_sampler);

    Vector3Df color = Vector3Df(0.f, 0.f, 0.f);
    Vector3Df mask = Vector3Df(1.f, 1.f, 1.f);
    SurfaceInteraction interaction = SurfaceInteraction();
    Triangle* p_triangles = p_sceneData->p_triangles;
    Triangle* p_hitTriangle = NULL;
    refl_t currentBsdf = DIFFUSE;
    refl_t previousBsdf = DIFFUSE;
	uint materialId;
#ifdef __CUDA_ARCH__
    dataPtr_t p_bvh = (dataPtr_t)p_sceneData->p_cudaTexObjects;
#else
    dataPtr_t p_bvh = (dataPtr_t)p_sceneData->p_bvh;
#endif
    for (unsigned bounces = 0; bounces < c_maxBounces; bounces++) {
    	if (!intersectBVH(p_bvh, p_triangles, interaction, ray))
        {
#ifdef USE_SKYBOX
    		color += mask * Vector3Df(skybox[0], skybox[1], skybox[2]);
#endif
    		break;
    	}

        p_hitTriangle = interaction.p_hitTriangle;
#ifdef SHOW_NORMALS
        return p_hitTriangle->getNormal(interaction.u, interaction.v);
#endif
		materialId = p_hitTriangle->_materialId;
        if (bounces == 0 || previousBsdf == SPECULAR || previousBsdf == REFRACTIVE) {
			color += mask * Vector3Df(p_matFloats[materialId*MATERIALS_FLOAT_COMPONENTS + KA_OFFSET]);
        }

        interaction.normal = p_hitTriangle->getNormal(interaction.u, interaction.v);
        interaction.position = ray.origin + ray.dir * ray.tMax;
        interaction.outputDirection = normalize(ray.dir);

        // SHADING CALCULATIONS
        currentBsdf = (refl_t)p_matIndices[materialId].x;

        // DIFFUSE AND SPECULAR BSDF
        if (currentBsdf == DIFFSPEC) {
			// use Russian roulette to decide whether to evaluate diffuse or specular BSDF
        	float p = p_matFloats[materialId*MATERIALS_FLOAT_COMPONENTS + AUX_OFFSET].z;
        	if (p_sampler->getNextFloat() < p) {
        		currentBsdf = DIFFUSE;
				mask *= 1.0f / p;
        	} else {
        		currentBsdf = SPECULAR;
        		mask *= 1.0f / (1.0f - p);
        	}
        }

        if (currentBsdf == REFRACTIVE) {
			Vector3Df reflectedDir, transmittedDir;
			float ni = p_matFloats[materialId*MATERIALS_FLOAT_COMPONENTS + AUX_OFFSET].y;
			Fresnel fresnel = getFresnelReflectance(interaction, ni, transmittedDir);
			if (fresnel.probReflection == 1.0f) {
				currentBsdf = SPECULAR;
			}
			else {
				// TODO: What's this magic number?
				float unknownMagicNumber = 0.25f;
				float P = unknownMagicNumber + .5f * fresnel.probReflection;
				float RP = fresnel.probReflection /P;
				float TP = fresnel.probTransmission / (1.f - P);
				// Reason dictates that this should be Re
				if (p_sampler->getNextFloat() > unknownMagicNumber) {
					interaction.inputDirection = transmittedDir;
					mask *= Vector3Df(p_matFloats[materialId*MATERIALS_FLOAT_COMPONENTS + KS_OFFSET]) * TP;
				} else {
					mask *= RP;
					currentBsdf = SPECULAR;
				}
			}
        }

        // DIFFUSE BSDF
        if (currentBsdf == DIFFUSE) {
        	dataPtr_t p_texContainer = textureContainerFactory(p_matIndices[materialId].y,
        												   	   TEXTURE_CONTAINER_FACTOR_PARAMETERS(p_sceneData));
			Vector3Df diffuseColor = Vector3Df(p_matFloats[materialId*MATERIALS_FLOAT_COMPONENTS + KD_OFFSET]);
        	Vector3Df diffuseSample = sampleDiffuseBSDF(&interaction, p_hitTriangle, diffuseColor, p_texContainer, p_sampler);
			mask = mask * diffuseSample / interaction.pdf;
#ifndef __CUDA_ARCH__
        	delete (TextureContainer*)p_texContainer;
#endif

			float randomNumber = p_sampler->getNextFloat() * ((float)p_lightsData->numLights - .00001f);
			int selectedLightIdx = truncf(randomNumber);
			Triangle* p_light = &p_lightsData->lightsPtr[selectedLightIdx];
			Vector3Df lightColor = Vector3Df(p_matFloats[p_light->_materialId*MATERIALS_FLOAT_COMPONENTS + KA_OFFSET]);
			Vector3Df directLighting = estimateDirectLighting(p_light, p_sceneData, lightColor, interaction, p_sampler);

#ifndef UNBIASED
			directLighting.x = clamp(directLighting.x, 0.0f, 1.0f);
			directLighting.y = clamp(directLighting.y, 0.0f, 1.0f);
			directLighting.z = clamp(directLighting.z, 0.0f, 1.0f);
#endif
			color += mask * directLighting;
		}

        // PURE SPECULAR BSDF
        if (currentBsdf == SPECULAR) {
			Vector3Df specularColor = Vector3Df(p_matFloats[materialId*MATERIALS_FLOAT_COMPONENTS + KS_OFFSET]);
        	Vector3Df perfectSpecularSample = sampleSpecularBSDF(&interaction, specularColor);
			mask = mask * perfectSpecularSample / interaction.pdf;
        }

        previousBsdf = currentBsdf;

        ray.origin = interaction.position + interaction.inputDirection * EPSILON;
        ray.dir = interaction.inputDirection;
        ray.tMin = EPSILON;
        ray.tMax = FLT_MAX;

        // Russian Roulette
        if (bounces >= 3) {
            float p = maxComponent(mask);
            if (p_sampler->getNextFloat() > p) {
                break;
            }
            mask *= 1.0f / p;
        }
    }
    return color;
}

__host__ __device__ bool intersectTriangles(Triangle* p_triangles, int numTriangles, SurfaceInteraction &interaction, Ray& ray) {
	const float tInitial = ray.tMax;
	float t;
	float u, v;
	Triangle* p_current = p_triangles;
	for(int i = 0; i < numTriangles; i++) {
		t = p_current->intersect(ray, u, v);
		if (t < ray.tMax && t > ray.tMin) {
			ray.tMax = t;
			interaction.p_hitTriangle = p_current;
			interaction.u = u;
			interaction.v = v;
		}
		p_current++;
	}
	return ray.tMax < tInitial;
}

// Check optimization on p 284 and 128-129
__host__ __device__ bool intersectBVH(dataPtr_t p_bvhData, Triangle* p_triangles, SurfaceInteraction &interaction, Ray& ray) {

	bool hit = false;
	Vector3Df invDir = Vector3Df(1/ray.dir.x, 1/ray.dir.y, 1/ray.dir.z);
	int dirIsNeg[3] = { invDir.x < 0, invDir.y < 0, invDir.z < 0 };
	int toVisitOffset = 0, currentNodeIndex = 0;
	int stack[16];
	while(true) {
#ifdef __CUDA_ARCH__
		cudaTextureObject_t* p_textureObject = (cudaTextureObject_t*)p_bvhData;
		Vector3Df minBound = Vector3Df(tex1Dfetch<float4>(p_textureObject[BVH_BOUNDS_OFFSET], 2*currentNodeIndex));
		Vector3Df maxBound = Vector3Df(tex1Dfetch<float4>(p_textureObject[BVH_BOUNDS_OFFSET], 2*currentNodeIndex + 1));
		int2 bvhOffsets = tex1Dfetch<int2>(p_textureObject[BVH_INDEX_OFFSET], currentNodeIndex);
		int32_t offset = bvhOffsets.x;
		int16_t numTriangles = (int16_t)((bvhOffsets.y & 0xFFFF0000) >> 16);
		int16_t axis = (int16_t)(bvhOffsets.y & 0xFFFF);
#else
		const LinearBVHNode* p_bvh = (LinearBVHNode*)p_bvhData;
		const LinearBVHNode* p_node = &p_bvh[currentNodeIndex];
		Vector3Df minBound = p_node->min;
		Vector3Df maxBound = p_node->max;
		int32_t offset = p_node->secondChildOffset;
		int16_t numTriangles = p_node->numTriangles;
		int16_t axis = p_node->axis;
#endif
		if (rayIntersectsBox(ray, minBound, maxBound)) {
			if (numTriangles > 0) {
				if (intersectTriangles(&p_triangles[offset], numTriangles, interaction, ray)) {
					hit = true;
				}
				if (toVisitOffset == 0) break;
				currentNodeIndex = stack[--toVisitOffset];
			} else {
				if (dirIsNeg[axis]) {
					stack[toVisitOffset++] = currentNodeIndex + 1;
					currentNodeIndex = offset;
				} else {
					stack[toVisitOffset++] = offset;
					currentNodeIndex = currentNodeIndex + 1;
				}
			}
		} else {
			if (toVisitOffset == 0) break;
			currentNodeIndex = stack[--toVisitOffset];
		}
	}
	return hit;
}

// Note: potential optimization here on page 128-129 (bounds.intersectP())
__host__ __device__ bool rayIntersectsBox(Ray& ray, const Vector3Df& min, const Vector3Df& max) {
	float t0 = -FLT_MAX, t1 = FLT_MAX;

	float invRayDir = 1.f/ray.dir.x;
	float tNear = (min.x - ray.origin.x) * invRayDir;
	float tFar = (max.x - ray.origin.x) * invRayDir;
	if (tNear > tFar) {
		float tmp = tNear;
		tNear = tFar;
		tFar = tmp;
	}
	t0 = tNear > t0 ? tNear : t0;
	t1 = tFar < t1 ? tFar : t1;
	if (t0 > t1) return false;

	invRayDir = 1.f/ray.dir.y;
	tNear = (min.y - ray.origin.y) * invRayDir;
	tFar = (max.y - ray.origin.y) * invRayDir;
	if (tNear > tFar) {
		float tmp = tNear;
		tNear = tFar;
		tFar = tmp;
	}
	t0 = tNear > t0 ? tNear : t0;
	t1 = tFar < t1 ? tFar : t1;
	if (t0 > t1) return false;

	invRayDir = 1.f/ray.dir.z;
	tNear = (min.z - ray.origin.z) * invRayDir;
	tFar = (max.z - ray.origin.z) * invRayDir;
	if (tNear > tFar) {
		float tmp = tNear;
		tNear = tFar;
		tFar = tmp;
	}
	t0 = tNear > t0 ? tNear : t0;
	t1 = tFar < t1 ? tFar : t1;
	if (t0 > t1) return false;

	return true;
}

__host__ __device__ Vector3Df sampleDiffuseBSDF(SurfaceInteraction* p_interaction,
												Triangle* p_hitTriangle,
												const Vector3Df& diffuseColor,
												dataPtr_t p_textureContainer,
												Sampler* p_sampler) {
   float r1 = 2 * M_PI * p_sampler->getNextFloat();
   float r2 = p_sampler->getNextFloat();
   float r2sq = sqrtf(r2);
   // calculate orthonormal coordinates u, v, w, at hitpt
   Vector3Df w = p_interaction->normal;
   Vector3Df u = normalize(cross( (fabs(w.x) > 0.1f ?
			   Vector3Df(0.f, 1.f, 0.f) :
			   Vector3Df(1.f, 0.f, 0.f)), w));
   Vector3Df v = cross(w, u);
   p_interaction->inputDirection = normalize(u * cosf(r1) * r2sq + v * sinf(r1) * r2sq + w * sqrtf(1.f - r2));
   p_interaction->pdf = 0.5f;
   float cosineWeight = dot(p_interaction->inputDirection, p_interaction->normal);

   Vector3Df kd = diffuseColor;
   if (p_textureContainer != NULL) {
	   float u = p_interaction->u;
	   float v = p_interaction->v;
	   float w = 1.f - u - v;
	   Vector2Df uv = p_hitTriangle->_uv1 * w + p_hitTriangle->_uv2 * u + p_hitTriangle->_uv3 * v;
	   kd = sampleTexture(p_textureContainer, uv.x, uv.y);
   }
   return kd * cosineWeight;
}

__host__ __device__ Vector3Df sampleSpecularBSDF(SurfaceInteraction* p_interaction, const Vector3Df& specularColor) {
	p_interaction->inputDirection = reflect(p_interaction->outputDirection,  p_interaction->normal);
	p_interaction->pdf = 1.0f;
	return specularColor;
}

__host__ __device__ Vector3Df reflect(const Vector3Df& incedent, const Vector3Df& normal) {
	return incedent - normal * dot(incedent, normal) * 2.f;
}

__host__ __device__ Vector3Df estimateDirectLighting(Triangle* p_light, SceneData* p_sceneData, const Vector3Df& lightColor, const SurfaceInteraction &interaction, Sampler* p_sampler) {
	if (interaction.p_hitTriangle->_triId == p_light->_triId) {
		return Vector3Df(0.0f, 0.0f, 0.0f);
	}
	Vector3Df directLighting(0.0f, 0.0f, 0.0f);
	Vector3Df rayOrigin = interaction.position + interaction.normal * EPSILON;
	Ray ray(rayOrigin,  normalize(p_light->getRandomPointOn(p_sampler) - interaction.position));
	SurfaceInteraction lightInteraction = SurfaceInteraction();
	// Sample the light
	Triangle* p_triangles = p_sceneData->p_triangles;
#ifdef __CUDA_ARCH__
	dataPtr_t p_bvh = (dataPtr_t)p_sceneData->p_cudaTexObjects;
#else
	dataPtr_t p_bvh = (dataPtr_t)p_sceneData->p_bvh;
#endif
	bool intersectsLight = intersectBVH(p_bvh, p_triangles, lightInteraction, ray);
	if (intersectsLight && lightInteraction.p_hitTriangle->_triId == p_light->_triId) {
		float surfaceArea = p_light->_surfaceArea;
		float distanceSquared = ray.tMax*ray.tMax;
		// For directional lights also consider light direction
		// float incidenceAngle = fabs(dot(p_light->getNormal(lightInteraction.u, lightInteraction.v), -ray.dir));
		// Otherwise direct lighting is based on the diffuse term and obey Lambert's cosine law
		float cosTheta = fabs(dot(ray.dir, interaction.normal));
		float weightFactor = surfaceArea/distanceSquared * cosTheta;
		directLighting += lightColor * weightFactor;
	}
	return directLighting;
}

// This function provides a wrapper as a common interface for sequential and parallel versions for
// accessing actual textures (surface textures, not CUDA textures) from either CUDA texture or
// memory in p_sceneData
__host__ __device__ dataPtr_t textureContainerFactory(int i, TEXTURE_CONTAINER_FACTORY_ARGUMENTS) {
	if (i == NO_TEXTURE) {
		return NULL;
	}
#ifdef __CUDA_ARCH__
	cudaTextureObject_t* p_textureObject = &p_texObject[i + TEXTURES_OFFSET];
	return (dataPtr_t)(p_textureObject);
#else
	Vector3Df* p_textureData = &p_texData[p_texOffsets[i]];
	pixels_t* p_textureDimensions = &p_texDimensions[2 * i];
	return (dataPtr_t)(new TextureContainer(p_textureData, p_textureDimensions));
#endif
}

__host__ __device__ Fresnel getFresnelReflectance(const SurfaceInteraction& interaction, const float ior, Vector3Df& transmittedDir) {
	Fresnel fresnel;
	Vector3Df incedent = interaction.outputDirection;
	Vector3Df normal = interaction.normal;
	float cosi = dot(incedent, normal);
	float etai = 1, etat = ior;
	Vector3Df n = normal;
	if (cosi < 0) {
		cosi = -cosi;
	} else {
		float temp = etai;
		etai = etat;
		etat = temp;
		n = -normal;
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

__host__ __device__ void gammaCorrectPixel(uchar4 &p) {
	float invGamma = 1.f/2.2f;
	float3 fp = make_float3(0,0,0);
	fp.x = pow((float)p.x * 1.f/255.f, invGamma);
	fp.y = pow((float)p.y * 1.f/255.f, invGamma);
	fp.z = pow((float)p.z * 1.f/255.f, invGamma);
	p.x = (unsigned char)(fp.x * 255.f);
	p.y = (unsigned char)(fp.y * 255.f);
	p.z = (unsigned char)(fp.z * 255.f);
}
