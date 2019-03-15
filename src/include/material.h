#ifndef MATERIAL_H
#define MATERIAL_H

#include "linalg.h"

#define NO_TEXTURE -1

enum refl_t {
    DIFFUSE,
    SPECULAR,
    DIFFSPEC,
    MICROFACET,
    REFRACTIVE,
    EMISSIVE
};

struct Material {
    refl_t bsdf = DIFFUSE;
    // TODO: Union kd and ka because they'll never both be used in the same place
    float3 kd;// = make_float3(1,1,1);
    float3 ka;// = float3(0,0,0);
    float3 ks;// = float3(0,0,0);
    // Specular exponent
    float ns = 0.0f;
    // IOR
    float ni = 1.0f;
    // Diffuse Coefficient
    float diffuseCoefficient = 1.0f;
    int texKdIdx = NO_TEXTURE;
    
    inline __host__  bool operator==(const Material& m) const{
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
		if (diffuseCoefficient != m.diffuseCoefficient) {
			return false;
		}
		return true;
    }
}__attribute__((aligned(32)));

struct materialComparator {
    bool operator() (const Material& lhs, const Material& rhs) const {
		if (lhs.bsdf != rhs.bsdf) {
			return lhs.bsdf < rhs.bsdf;
		}
		if (lhs.kd.x < rhs.kd.x || lhs.kd.y < rhs.kd.y || lhs.kd.z < rhs.kd.z) {
			return true;
		}
		if (lhs.ka.x < rhs.ka.x || lhs.ka.y < rhs.ka.y || lhs.ka.z < rhs.ka.z) {
			return true;
		}
		if (lhs.ks.x < rhs.ks.x || lhs.ks.y < rhs.ks.y || lhs.ks.z < rhs.ks.z) {
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
