#include "camera.cuh"
#include "sequential.h"
#include <algorithm>
#include <cstdlib>
#include <vector>
#include <cfloat>

using namespace std;
using namespace geom;

void averageSamplesAndGammaCorrect(Vector3Df* img, int width, int height, int samples);
Vector3Df radiance(Scene& scene, const Ray& ray);

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

bool recursiveIntersectBVH(BVHNode* bvh,
				  const Ray& ray,
				  RayHit *hitData) {
	float t = FLT_MAX;
	float tprime = FLT_MAX;
	if (!(bvh->IsLeaf())) {   // INNER NODE
		if (hitsBox(ray, bvh)) {
			BVHInner *p = dynamic_cast<BVHInner*>(bvh);
			return (recursiveIntersectBVH(p->_right, ray, hitData)
				 || recursiveIntersectBVH(p->_left, ray, hitData));
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
				hitData->hitTriPtr = &triangle;
				hitData->u = u;
				hitData->v = v;
				hitData->t = t;
			}
		}
		return t < FLT_MAX;
	}
	return false;
}


Vector3Df* sequentialRenderWrapper(Scene& scene, int width, int height, int samples, int numStreams, bool &useTexMemory, int argc, char** argv) {

	Vector3Df* img = new Vector3Df[width*height];
	srand(0);
	Camera* camera = scene.getCameraPtr();
	BVHNode* bboxPtr = scene.getSceneBVHPtr();
	Triangle* triangles = scene.getTriPtr();
	Triangle* lights = scene.getLightsPtr();
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++){
			RayHit hitData;
			int idx = width*i + j;
			for (int s = 0; s < samples; s++) {
				Ray ray = camera->computeSequentialCameraRay(j, i);
				img[idx] += radiance(scene, ray);
//				if(recursiveIntersectBVH(bboxPtr, ray, &hitData)) {
//					img[idx] += hitData.hitTriPtr->_colorDiffuse;

//				}
			}
		}
	}
	averageSamplesAndGammaCorrect(img, width, height, samples);
	return img;
}

Vector3Df radiance(Scene& scene, const Ray& ray) {
	Vector3Df color(0.0f, 0.0f, 0.0f);
	Vector3Df hitPt, normal;
	RayHit hitData, lightHitData;
	Triangle* hitTriPtr;
	if(recursiveIntersectBVH(scene.getSceneBVHPtr(), ray, &hitData)) {
		hitTriPtr = hitData.hitTriPtr;
		hitPt = hitData.t;
		color += hitTriPtr->_colorDiffuse;
		// TODO: getNormal should just take u,v
//		normal = hitTriPtr->getNormal(hitData);
//		if (hitTriPtr->isEmissive()) {
//			color += hitTriPtr->_colorEmit;
//		}
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
