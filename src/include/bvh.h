/*
 * bvh.h
 *
 *  Created on: Jan 7, 2019
 *      Author: matt
 */

#ifndef BVH_H_
#define BVH_H_
#include "linalg.h"


class Scene;

void constructBVH(Scene* p_scene);

struct LinearBVHNode {
	Vector3Df min;
	Vector3Df max;
	union {
		int trianglesOffset;
		int secondChildOffset;
	};
	int16_t numTriangles = 0;
	uint8_t axis;
	int8_t pad = 0;
};

#endif /* BVH_H_ */
