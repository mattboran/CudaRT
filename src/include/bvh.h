/*
 * bvh.h
 *
 *  Created on: Jan 7, 2019
 *      Author: matt
 */

#ifndef BVH_H_
#define BVH_H_
#include "linalg.h"
#include <stdint.h>


class Scene;

void constructBVH(Scene* p_scene);

struct LinearBVHNode {
	float3 min;
	float3 max;
	union {
		int trianglesOffset;
		int secondChildOffset;
	};
	int16_t numTriangles = 0;
	int16_t axis;
};

#endif /* BVH_H_ */
