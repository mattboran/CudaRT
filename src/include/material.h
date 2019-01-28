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

#endif
