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
    refl_t bsdf;
    Vector3Df kd;
    Vector3Df ka;
    Vector3Df ks;
    // Specular exponent
    float ns;
    // IOR
    float ni;
};

struct materialComparator {
    bool operator() (const Material& lhs, const Material& rhs) const {
    	if (lhs.bsdf != rhs.bsdf) {
    		return lhs.bsdf < rhs.bsdf;
    	}
    	if (distancesq(lhs.kd, rhs.kd) > 0) {
    		return true;
    	}
    	if (distancesq(lhs.ka, rhs.ka) > 0) {
			return true;
		}
    	if (distancesq(lhs.ks, rhs.ks) > 0) {
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
