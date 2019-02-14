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



__host__ __device__ bool intersectTriangles(Triangle* p_triangles, int numTriangles, SurfaceInteraction &interaction, Ray& ray);
__host__ __device__ bool rayIntersectsBox(Ray& ray, const Vector3Df& min, const Vector3Df& max);
__host__ __device__ Vector3Df sampleDiffuseBSDF(SurfaceInteraction* p_interaction, Triangle* p_hitTriangle, Material* p_material, Sampler* p_sampler);
__host__ __device__ Vector3Df sampleSpecularBSDF(SurfaceInteraction* p_interaction, Triangle* p_hitTriangle, Material* p_material);
__host__ __device__ Vector3Df estimateDirectLighting(Triangle* p_light, TrianglesData* p_trianglesData, Material* p_material, const SurfaceInteraction &interaction, Sampler* p_sampler);
__host__ __device__ bool intersectBVH(LinearBVHNode* p_bvh, Triangle* p_triangles, SurfaceInteraction &interaction, Ray& ray);
__host__ __device__ Vector3Df reflect(const Vector3Df& incedent, const Vector3Df& normal);
__host__ __device__ Vector3Df refract(const Vector3Df& incedent, const Vector3Df& normal, const float ior, float& rp, float& tp);
__host__ __device__ float getFresnelReflectance(const SurfaceInteraction &interaction, float ior);

#ifdef USE_SKYBOX
#ifdef __CUDA_ARCH__
__constant__ float skybox[] = { 0.0f, 0.025f, 0.05f };
#else
static const float skybox[] = { 0.0f, 0.025f, 0.05f };
#endif
#endif

Vector3Df** pp_textures;
Vector3Df** dp_textures;
pixels_t* p_textureDimensions;
pixels_t* d_textureDimensions;

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


__host__ void Renderer::createSettingsData(SettingsData* p_settingsData){
	p_settingsData->width = getWidth();
	p_settingsData->height = getHeight();
	p_settingsData->samples = getSamples();
}

__host__ void Renderer::createTrianglesData(TrianglesData* p_trianglesData, Triangle* p_triangles, LinearBVHNode* p_bvh, Material* p_materials) {
	p_trianglesData->p_triangles = p_triangles;
	p_trianglesData->p_bvh = p_bvh;
	p_trianglesData->p_materials = p_materials;
	p_trianglesData->numTriangles = p_scene->getNumTriangles();
	p_trianglesData->numBVHNodes = p_scene->getNumBvhNodes();
	p_trianglesData->numMaterials = p_scene->getNumMaterials();
}

__host__ void Renderer::createLightsData(LightsData* p_lightsData, Triangle* p_triangles) {
	p_lightsData->lightsPtr = p_triangles;
	p_lightsData->numLights = p_scene->getNumLights();
	p_lightsData->totalSurfaceArea = p_scene->getLightsSurfaceArea();
}

__host__ void Renderer::allocateTextures(pixels_t* p_texDimensions, uint numTextures) {
	pp_textures = (Vector3Df**)malloc(numTextures * sizeof(Vector3Df*));
	for (uint i = 0; i < numTextures; i++) {
		pixels_t width = p_texDimensions[2 * i];
		pixels_t height = p_texDimensions[2 * i + 1];
		size_t imageSize = width * height * sizeof(Vector3Df);
		pp_textures[i] = (Vector3Df*)malloc(imageSize);
		p_textureDimensions = (pixels_t*)malloc(2 * numTextures * sizeof(pixels_t));
	}
}

__host__ void Renderer::loadTextures(Vector3Df** pp_tex, pixels_t* p_texDimensions, uint numTextures) {
	memcpy(p_textureDimensions, p_texDimensions, sizeof(pixels_t) * numTextures * 2);
	for (uint i = 0; i < numTextures; i++) {
		pixels_t width = p_texDimensions[2 * i];
		pixels_t height = p_texDimensions[2 * i + 1];
		size_t imageSize = width * height * sizeof(Vector3Df);
		memcpy(pp_textures[i], pp_tex[i], imageSize);
	}
}

__host__ __device__ Vector3Df sampleTexture(uint idx, float u, float v) {
#ifdef __CUDA_ARCH__
	return Vector3Df(1,0,0);
#else
	pixels_t width = p_textureDimensions[idx * 2];
	pixels_t height = p_textureDimensions[idx * 2 + 1];
	float pixelCoordU = u * (float)width;
	float pixelCoordV = v * (float)height;
	float floorPixelCoordU = floorf(pixelCoordU);
	float floorPixelCoordV = floorf(pixelCoordV);
	float ceilPixelCoordU = ceilf(pixelCoordU);
	float ceilPixelCoordV = ceilf(pixelCoordV);
	pixels_t i = truncf(pixelCoordU);
	pixels_t j = truncf(pixelCoordV);

	// Bilinear interpolation
	Vector3Df* p_texture = pp_textures[idx];
	Vector3Df valA1 = p_texture[j * width + i + 1] * (pixelCoordU - floorPixelCoordU);
	Vector3Df valA2 = p_texture[j * width + i] * (ceilPixelCoordU - pixelCoordU);
	Vector3Df valB1 = p_texture[(j + 1) * width + i + 1] * (pixelCoordU - floorPixelCoordU);
	Vector3Df valB2 = p_texture[(j + 1) * width + i] * (ceilPixelCoordU - pixelCoordU);

	Vector3Df valC1 = (valA1 + valA2) * (ceilPixelCoordV - pixelCoordV);
	Vector3Df valC2 = (valB1 + valB2) * (pixelCoordV - floorPixelCoordV);
	return valC1 + valC2;
#endif
}

__host__ __device__ Vector3Df samplePixel(int x, int y, Camera* p_camera, TrianglesData* p_trianglesData, LightsData *p_lightsData, Material* p_materials, Sampler* p_sampler) {
	Ray ray = p_camera->computeCameraRay(x, y, p_sampler);

    Vector3Df color = Vector3Df(0.f, 0.f, 0.f);
    Vector3Df mask = Vector3Df(1.f, 1.f, 1.f);
    SurfaceInteraction interaction = SurfaceInteraction();
    Triangle* p_triangles = p_trianglesData->p_triangles;
    Triangle* p_hitTriangle = NULL;
    refl_t currentBsdf = DIFFUSE;
    refl_t previousBsdf = DIFFUSE;
    LinearBVHNode* p_bvh = p_trianglesData->p_bvh;
    for (unsigned bounces = 0; bounces < 6; bounces++) {
    	if (!intersectBVH(p_bvh, p_triangles, interaction, ray)) {
#ifdef USE_SKYBOX
    		color += mask * Vector3Df(skybox[0], skybox[1], skybox[2]);
#endif
    		break;
    	}

        p_hitTriangle = interaction.p_hitTriangle;
#ifdef SHOW_NORMALS
        return p_hitTriangle->getNormal(interaction.u, interaction.v);
#endif
        Material* p_material = &p_materials[p_hitTriangle->_materialId];
        if (bounces == 0 || previousBsdf == SPECULAR || previousBsdf == REFRACTIVE) {
        	color += mask * p_material->ka;
        }

        interaction.normal = p_hitTriangle->getNormal(interaction.u, interaction.v);
        interaction.position = ray.origin + ray.dir * ray.tMax;
        interaction.outputDirection = normalize(ray.dir);
        interaction.p_hitTriangle = p_hitTriangle;

        // SHADING CALCULATIONS
        currentBsdf = p_material->bsdf;

        // DIFFUSE AND SPECULAR BSDF
        if (currentBsdf == DIFFSPEC) {
			// use Russian roulette to decide whether to evaluate diffuse or specular BSDF
        	float p = p_material->diffuseCoefficient;
        	if (p_sampler->getNextFloat() < p) {
        		currentBsdf = DIFFUSE;
				mask *= 1.0f / p;
        	} else {
        		currentBsdf = SPECULAR;
        		mask *= 1.0f / (1.0f - p);
        	}
        }

        if (currentBsdf == REFRACTIVE) {
			Vector3Df incedent = interaction.outputDirection;
			Vector3Df normal = interaction.normal;
			float cosi = dot(incedent, normal);
			float etai = 1, etat = p_material->ni;
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
				currentBsdf = SPECULAR;
			} else {
				Vector3Df transmitted = incedent * eta + n * (eta * cosi - sqrtf(k));
				float R0 = (etat - etai) * (etat - etai) / (etat + etai) * (etat + etai);
				float c = 1.f - dot(transmitted, normal);
				float Re = R0 + (1.f - R0) * c * c * c * c * c;
				float Tr = 1.f - Re;
				float P = .25f + .5f * Re;
				float RP = Re/P;
				float TP = Tr / (1.f - P);
				if (p_sampler->getNextFloat() > .25f) {
					interaction.inputDirection = transmitted;
					mask *= p_material->ks * TP;
				} else {
					currentBsdf = SPECULAR;
					mask *= RP;
				}
			}
        }

        // DIFFUSE BSDF
        if (currentBsdf == DIFFUSE) {
        	Vector3Df diffuseSample = sampleDiffuseBSDF(&interaction, p_hitTriangle, p_material, p_sampler);
			mask = mask * diffuseSample / interaction.pdf;

			float randomNumber = p_sampler->getNextFloat() * ((float)p_lightsData->numLights - .00001f);
			int selectedLightIdx = (int)truncf(randomNumber);
			Triangle* p_light = &p_lightsData->lightsPtr[selectedLightIdx];
			Material* p_lightMaterial = &p_materials[p_light->_materialId];
			Vector3Df directLighting = estimateDirectLighting(p_light, p_trianglesData, p_lightMaterial, interaction, p_sampler);

#ifndef UNBIASED
			directLighting.x = clamp(directLighting.x, 0.0f, 1.0f);
			directLighting.y = clamp(directLighting.y, 0.0f, 1.0f);
			directLighting.z = clamp(directLighting.z, 0.0f, 1.0f);
#endif
			color += mask * directLighting;
		}

        // PURE SPECULAR BSDF
        if (currentBsdf == SPECULAR) {
        	Vector3Df perfectSpecularSample = sampleSpecularBSDF(&interaction, p_hitTriangle, p_material);
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
	while(numTriangles--) {
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
__host__ __device__ bool intersectBVH(LinearBVHNode* p_bvh, Triangle* p_triangles, SurfaceInteraction &interaction, Ray& ray) {

	bool hit = false;
	Vector3Df invDir = Vector3Df(1/ray.dir.x, 1/ray.dir.y, 1/ray.dir.z);
	int dirIsNeg[3] = { invDir.x < 0, invDir.y < 0, invDir.z < 0 };
	int toVisitOffset = 0, currentNodeIndex = 0;
	int stack[16];
	while(true) {
		const LinearBVHNode* p_node = &p_bvh[currentNodeIndex];
		if (rayIntersectsBox(ray, p_node->min, p_node->max)) {
			if (p_node->numTriangles > 0) {
				if (intersectTriangles(&p_triangles[p_node->trianglesOffset], p_node->numTriangles, interaction, ray)) {
					hit = true;
				}
				if (toVisitOffset == 0) break;
				currentNodeIndex = stack[--toVisitOffset];
			} else {
				if (dirIsNeg[p_node->axis]) {
					stack[toVisitOffset++] = currentNodeIndex + 1;
					currentNodeIndex = p_node->secondChildOffset;
				} else {
					stack[toVisitOffset++] = p_node->secondChildOffset;
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

__host__ __device__ Vector3Df sampleDiffuseBSDF(SurfaceInteraction* p_interaction, Triangle* p_hitTriangle, Material* p_material, Sampler* p_sampler) {
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
   return p_material->kd * cosineWeight;
}

__host__ __device__ Vector3Df sampleSpecularBSDF(SurfaceInteraction* p_interaction, Triangle* p_hitTriangle, Material* p_material) {
	Vector3Df normal = p_interaction->normal;
	Vector3Df incedent = p_interaction->outputDirection;
	p_interaction->inputDirection = reflect(incedent, normal);
	p_interaction->pdf = 1.0f;
	return p_material->ks;
}

__host__ __device__ Vector3Df reflect(const Vector3Df& incedent, const Vector3Df& normal) {
	return incedent - normal * dot(incedent, normal) * 2.f;
}

__host__ __device__ Vector3Df refract(const Vector3Df& incedent, const Vector3Df& normal, const float ior, float& rp, float& tp) {
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
		return Vector3Df(0,0,0);
	} else {
		Vector3Df transmitted = incedent * eta + n * (eta * cosi - sqrtf(k));
		float R0 = (etat - etai) * (etat - etai) / (etat + etai) * (etat + etai);
		float c = 1.f - dot(transmitted, n);
		float Re = R0 + (1.f - R0) * c * c * c * c * c;
		float TR = 1.f - Re;
		float P = .25f + 0.5f * Re;
		float rp = Re/P;
		float tp = TR / (1.f - P);
		return normalize(transmitted);
	}
}

__host__ __device__ Vector3Df estimateDirectLighting(Triangle* p_light, TrianglesData* p_trianglesData, Material* p_material, const SurfaceInteraction &interaction, Sampler* p_sampler) {
	Vector3Df directLighting(0.0f, 0.0f, 0.0f);
	if (sameTriangle(interaction.p_hitTriangle, p_light)) {
		return directLighting;
	}
	Vector3Df rayOrigin = interaction.position + interaction.normal * EPSILON;
	Ray ray(rayOrigin,  normalize(p_light->getRandomPointOn(p_sampler) - interaction.position));
	SurfaceInteraction lightInteraction = SurfaceInteraction();
	// Sample the light
	Triangle* p_triangles = p_trianglesData->p_triangles;
	LinearBVHNode* p_bvh = p_trianglesData->p_bvh;
#ifdef USE_BVH
	bool intersectsLight = intersectBVH(p_bvh, p_triangles, lightInteraction, ray);
#else
	bool intersectsLight = intersectTriangles(p_triangles, p_trianglesData->numTriangles, lightInteraction, ray);
#endif
	if (intersectsLight && sameTriangle(lightInteraction.p_hitTriangle, p_light)) {
		float surfaceArea = p_light->_surfaceArea;
		float distanceSquared = ray.tMax*ray.tMax;
		// For directional lights also consider light direction
//		float incidenceAngle = fabs(dot(p_light->getNormal(lightInteraction.u, lightInteraction.v), -ray.dir));
		// Otherwise direct lighting is based on the diffuse term and obey Lambert's cosine law
		float cosTheta = fabs(dot(ray.dir, interaction.normal));
		float weightFactor = surfaceArea/distanceSquared * cosTheta;
		directLighting += p_material->ka * weightFactor;
	}
	return directLighting;
}

__host__ __device__ float getFresnelReflectance(const SurfaceInteraction &interaction, float ior) {
	float n = ior;
	float cosI = -dot(interaction.outputDirection, interaction.normal);
	float sin2T = n * n * (1.0f - cosI * cosI);
	if (sin2T > 1.0f) {
		return 1.0f;
	}

	float cosT = sqrtf(1.0f - sin2T);
	float rPer = (n * cosI - cosT) / (n * cosI + cosT);
	float rPar = (cosI - n * cosT) / (cosI + n * cosT);
	return (rPer * rPer + rPar * rPar) / 2.0f;
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
