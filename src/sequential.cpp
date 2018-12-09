#include "camera.cuh"
#include "sequential.h"
#include <algorithm>
#include <cstdlib>
#include <vector>
#include <cfloat>

using namespace std;
using namespace geom;

bool intersectTriangles(Triangle* triPtr, int numTriangles, const Ray& ray, RayHit *hitData);
void averageSamplesAndGammaCorrect(Vector3Df* img, int width, int height, int samples);
Vector3Df radiance(Scene& scene, const Ray& ray, bool useBVH, int depth);
bool intersectBVH(BVHNode* bvh, const Ray& ray, RayHit *hitData);
Vector3Df getRandomPointOn(Triangle* tri);
float uniformRandom();

bool hitsBox(const Ray& ray, BVHNode* bbox) {
	float t0 = -FLT_MAX, t1 = FLT_MAX;
	//axes

	float invRayDir = 1.f/ray.dir.x;
	float tNear = (bbox->_bottom.x - ray.origin.x) * invRayDir;
	float tFar = (bbox->_top.x - ray.origin.x) * invRayDir;
	if (tNear > tFar) {
		float tmp = tNear;
		tNear = tFar;
		tFar = tmp;
	}
	t0 = tNear > t0 ? tNear : t0;
	t1 = tFar < t1 ? tFar : t1;
	if (t0 > t1) return false;

	invRayDir = 1.f/ray.dir.y;
	tNear = (bbox->_bottom.y - ray.origin.y) * invRayDir;
	tFar = (bbox->_top.y - ray.origin.y) * invRayDir;
	if (tNear > tFar) {
		float tmp = tNear;
		tNear = tFar;
		tFar = tmp;
	}
	t0 = tNear > t0 ? tNear : t0;
	t1 = tFar < t1 ? tFar : t1;
	if (t0 > t1) return false;

	invRayDir = 1.f/ray.dir.z;
	tNear = (bbox->_bottom.z - ray.origin.z) * invRayDir;
	tFar = (bbox->_top.z - ray.origin.z) * invRayDir;
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

bool intersectBVH(BVHNode* bvh,
				  const Ray& ray,
				  RayHit *hitData) {
	float t = FLT_MAX;
	float tprime = FLT_MAX;
	if (!(bvh->IsLeaf())) {   // INNER NODE
		if (hitsBox(ray, bvh)) {
			BVHInner *p = dynamic_cast<BVHInner*>(bvh);
			return (intersectBVH(p->_right, ray, hitData)
				 || intersectBVH(p->_left, ray, hitData));
		}
	}
	else { // LEAF NODE
		BVHLeaf *p = dynamic_cast<BVHLeaf*>(bvh);
		float u, v;
		t = FLT_MAX;
		tprime = FLT_MAX;
		for (auto tri: p->_triangles) {
			Triangle triangle = *tri;
			tprime = triangle.intersect(ray, u, v);
			if (tprime < t && tprime > 0.f) {
				t = tprime;
				hitData->pHitTriangle = &triangle;
				hitData->u = u;
				hitData->v = v;
				hitData->t = t;
			}
		}
		return t < FLT_MAX;
	}
	return false;
}

// TODO: This should be a function in geometry.cu
bool intersectTriangles(Triangle* triPtr, int numTriangles, const Ray& ray, RayHit *hitData) {
	float u, v;
	float t = FLT_MAX, tprime = FLT_MAX;
	Triangle* pTriangle = triPtr;
	for (int i = 0; i < numTriangles; i++){
		tprime = pTriangle->intersect(ray, u, v);
		if (tprime < t && tprime > 0.f) {
			t = tprime;
			hitData->pHitTriangle = pTriangle;
			hitData->u = u;
			hitData->v = v;
			hitData->t = t;
		}
		pTriangle++;
	}
	return t < FLT_MAX;
}


Vector3Df* Sequential::pathtraceWrapper(Scene& scene, int width, int height, int samples, int numStreams, bool useBVH) {

	Vector3Df* img = new Vector3Df[width*height];
	srand(0);
	Camera* camera = scene.getCameraPtr();
	BVHNode* bboxPtr = scene.getSceneBVHPtr();
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++){
			RayHit hitData;
			int idx = width*i + j;
			for (int s = 0; s < samples; s++) {
				Ray ray = camera->computeSequentialCameraRay(j, i);
				img[idx] += radiance(scene, ray, useBVH, 0);
			}
		}
	}
	averageSamplesAndGammaCorrect(img, width, height, samples);
	return img;
}

Vector3Df radiance(Scene& scene, const Ray& ray, bool useBVH, int depth) {
	Vector3Df color(0.0f, 0.0f, 0.0f);
	if (depth >= 5) {
		return color;
	}

	Vector3Df hitPt, normal, nextDir;
	RayHit hitData, lightHitData;
	Triangle* pHitTriangle;
	Triangle* pTriangles = scene.getTriPtr();
	unsigned numTriangles = scene.getNumTriangles();
	BVHNode* pBvh = scene.getSceneBVHPtr();

	bool intersection = false;
	if (useBVH) {
		intersection = intersectBVH(pBvh, ray, &hitData);
	} else {
		intersection = intersectTriangles(pTriangles, numTriangles, ray, &hitData);
	}
	if(intersection) {
		pHitTriangle = hitData.pHitTriangle;
		if (pHitTriangle->isEmissive()) {
			return pHitTriangle->_colorEmit;
		}

		hitPt = hitData.t;
		normal = pHitTriangle->getNormal(hitData);
		color += pHitTriangle->_colorEmit;

//		// Calculate direct lighting for diffuse bounces
		int selectedLightIndex = rand() % scene.getNumLights();
		Triangle* selectedLight = &scene.getLightsPtr()[selectedLightIndex];
		Vector3Df lightRayDir = normalize(getRandomPointOn(selectedLight) - hitPt);

		intersection = false;
		Ray lightRay(hitPt + normal * EPSILON, lightRayDir);
		if (useBVH) {
			intersection = intersectBVH(pBvh, lightRay, &lightHitData);
		} else {
			intersection = intersectTriangles(pTriangles, numTriangles, lightRay, &lightHitData);
		}
		if (intersection){
			// See if we've hit the light we tested for
			Triangle* pLightHitTri = lightHitData.pHitTriangle;
			if (pLightHitTri->_triId == selectedLight->_triId) {
				float t = lightHitData.t;
				float surfaceArea = selectedLight->_surfaceArea;
				float distanceSquared = t*t; // scale by factor of 10
				float incidenceAngle = fabs(dot(selectedLight->getNormal(lightHitData), -lightRayDir));
				float weightFactor = surfaceArea/distanceSquared * incidenceAngle;
				color += selectedLight->_colorEmit * hitData.pHitTriangle->_colorDiffuse * weightFactor;
			}
		} else {
			return color + hitData.pHitTriangle->_colorDiffuse;
		}
		return color;
		// Calculate indirect lighting
		// TODO: getNormal should just take u,v
//		float r1 = 2 * M_PI * uniformRandom();
//		float r2 = uniformRandom();
//		float r2sq = sqrtf(r2);
//		Vector3Df w = normal;
//		Vector3Df u = normalize(cross( (fabs(w.x) > 0.1f ?
//					Vector3Df(0.f, 1.f, 0.f) :
//					Vector3Df(1.f, 0.f, 0.f)), w));
//		Vector3Df v = cross(w, u);
//		nextDir = normalize(u * cosf(r1) * r2sq + v * sinf(r1) * r2sq + w * sqrtf(1.f - r2));
//		hitPt += normal * EPSILON;
//		return color + radiance(scene, Ray(hitPt, nextDir), useBVH, depth+1) * (pHitTriangle->_colorDiffuse * dot(nextDir, normal) * 2.f);
	}
	return color;
}

void averageSamplesAndGammaCorrect(Vector3Df* img, int width, int height, int samples) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			int idx = width*i + j;
			Vector3Df pixel = img[idx];
			float gamma = 2.2f;
			float invGamma = 1.0f/gamma;
			float invSamples = 1.0f/(float)samples;
			pixel *= invSamples;
			img[idx].x = powf(fminf(pixel.x, 1.0f), invGamma);
			img[idx].y = powf(fminf(pixel.y, 1.0f), invGamma);
			img[idx].z = powf(fminf(pixel.z, 1.0f), invGamma);
		}
	}
}

Vector3Df getRandomPointOn(Triangle* tri){
	float u = uniformRandom();
	float v = uniformRandom();
	if (u + v >= 1.0f) {
		u = 1.0f - u;
		v = 1.0f - v;
	}
	return Vector3Df(tri->_v1 + tri->_e1 * u + tri->_e2 * v);
}

float uniformRandom() {
	return (float) rand() / (RAND_MAX + 1.f);
}
