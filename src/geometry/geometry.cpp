#include "geometry.h"
#include <math.h>

using namespace geom;
using std::max;
using std::min;

extern Vector3Df max4(const Vector3Df& a, const Vector3Df& b, const Vector3Df& c, const Vector3Df& d) {
	float x = max(a.x, max(b.x, max(c.x, d.x)));
	float y = max(a.y, max(b.y, max(c.y, d.y)));
	float z = max(a.z, max(b.z, max(c.z, d.z)));
	return Vector3Df(x,y,z);
}

extern Vector3Df min4(const Vector3Df& a, const Vector3Df& b, const Vector3Df& c, const Vector3Df& d) {
	float x = min(a.x, min(b.x, min(c.x, d.x)));
	float y = min(a.y, min(b.y, min(c.y, d.y)));
	float z = min(a.z, min(b.z, min(c.z, d.z)));
	return Vector3Df(x,y,z);
}
