#ifndef MATERIAL_H
#define MATERIAL_H

#include "linalg.h"

enum refl_t {
    DIFFUSE,
    SPECULAR,
    COOKETORRENCE,
    REFRACTIVE,
    EMISSIVE
};

struct Material {
    refl_t bsdf = DIFFUSE;
    Vector3Df kd = Vector3Df(1,1,1);
    Vector3Df ka = Vector3Df(0,0,0);
    Vector3Df ks = Vector3Df(0,0,0);
    // Specular exponent
    float ns = 0.0f;
    // IOR
    float ni = 1.0f;
    inline __host__ bool operator==(const Material& m) const{
    	if (bsdf != m.bsdf) {
			return false;
		}
		if (kd != m.kd) {
			return false;
		}
		if (ka != m.ka) {
			return false;
		}
		if (ks != m.ks) {
			return false;
		}
		if (ns != m.ns) {
			return false;
		}
		if (ni != m.ni) {
			return false;
		}
		return true;
    }
};

struct materialComparator {
    bool operator() (const Material& lhs, const Material& rhs) const {
		if (lhs.bsdf != rhs.bsdf) {
			return lhs.bsdf < rhs.bsdf;
		}
		if (lhs.kd.x < rhs.kd.x || lhs.kd.y < rhs.kd.y || lhs.kd.z < rhs.kd.z) {
			return true;
		}
		if (lhs.kd.x < rhs.kd.x || lhs.kd.y < rhs.kd.y || lhs.kd.z < rhs.kd.z) {
			return true;
		}
		if (lhs.kd.x < rhs.kd.x || lhs.kd.y < rhs.kd.y || lhs.kd.z < rhs.kd.z) {
			return true;
		}
		if (lhs.ns < rhs.ns) {
			return true;
		}
		if (lhs.ni < rhs.ni) {
			return true;
		}
		return false;
    }
};

#endif
