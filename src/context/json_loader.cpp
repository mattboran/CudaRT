/*
 * json_loader.cpp
 *
 *  Created on: Feb 4, 2019
 *      Author: matt
 */

#include "json_loader.h"
#include "linalg.h"

#include <exception>
#include <fstream>
#include <iostream>
#include <string>

using namespace std;

Vector3Df vector3FromArray(picojson::array arr);

JsonLoader::JsonLoader(std::string cam, std::string mat) : cameraFile(cam), materialsFile(mat) {
	ifstream c(cameraFile);
	string camJson((istreambuf_iterator<char>(c)),
			istreambuf_iterator<char>());
	string err = picojson::parse(cameraValue, camJson);
	if (!err.empty()) {
		throw std::runtime_error(err + " loading camera!");
	}
//	ifstream m(materialsFile);
//	string matJson((istreambuf_iterator<char>(m)),
//			istreambuf_iterator<char>());
//	err = picojson::parse(materialsValue, matJson);
//	if (!err.empty()) {
//		throw std::runtime_error(err + " loading materials!");
//	}
}

Camera JsonLoader::getCamera(int width, int height) {
	const picojson::value::object& obj = cameraValue.get<picojson::object>();
	float f = cameraValue.get("fieldOfView").get<double>();
	float focalLength = cameraValue.get("focalLength").get<double>();
	float fStop = cameraValue.get("fStop").get<double>();
	picojson::array e = cameraValue.get("eye").get<picojson::array>();
	picojson::array d = cameraValue.get("viewDirection").get<picojson::array>();
	picojson::array u = cameraValue.get("upDirection").get<picojson::array>();

	Camera camera;
	camera.xpixels = width;
	camera.ypixels = height;
	camera.fov = tanf(f * 0.5f * M_PI/180.0f);
	camera.eye = vector3FromArray(e);
	Vector3Df dir = vector3FromArray(d);
	camera.focusDistance = dir.length();
	camera.dir = normalize(dir);
	camera.up = normalize(vector3FromArray(u));
	camera.right = normalize(cross(camera.dir,camera.up));
	camera.apertureWidth = focalLength/fStop;
	camera.aspect = (float)width / (float)height;

	return camera;
}

// Note this is not efficient, it's O(n^2) for all materials
//Material JsonLoader::getMaterial(std::string name) {
//	Material mtl;
//	picojson::value::array materials = materialsValue.get("materials").get<picojson::array>();
//	for (auto it = materials.begin(); it != materials.end(); it++) {
//		string otherName = it->get("name").get<string>();
//		if (name.compare(otherName + "SG") != 0 ) {
//			continue;
//		}
//
//		mtl.ka = vector3FromArray(it->get("ka").get<picojson::array>());
//		mtl.kd = vector3FromArray(it->get("kd").get<picojson::array>());
//		mtl.ks = vector3FromArray(it->get("ks").get<picojson::array>());
//		mtl.diffuseCoefficient = it->get("diffuse").get<double>();
//		mtl.ni = it->get("ni").get<double>();
//		mtl.ns = it->get("roughness").get<double>();
//		mtl.bsdf = LAMBERT;
//		if (mtl.bsdf == DIFFSPEC && mtl.diffuseCoefficient == 0.0f) {
//			mtl.bsdf = SPECULAR;
//		}
//		else if (mtl.bsdf == DIFFSPEC && mtl.diffuseCoefficient == 1.0f) {
//			mtl.bsdf = LAMBERT;
//		}
//		if (mtl.ns > 0.0f) {
//			mtl.bsdf = MICROFACET;
//		}
//		cout << "BSDF = " << mtl.bsdf << endl;
//		break;
//	}
//
//	return mtl;
//}
//
//void JsonLoader::updateMaterialFields(std::string name, Material* p_matl) {
//	picojson::value::array materials = materialsValue.get("materials").get<picojson::array>();
//	for (auto it = materials.begin(); it != materials.end(); it++) {
//		string otherName = it->get("name").get<string>();
//		if (name.compare(otherName + "SG") != 0 ) {
//			continue;
//		}
//
//		// p_matl->ka = vector3FromArray(it->get("ka").get<picojson::array>());
//		// p_matl->kd = vector3FromArray(it->get("kd").get<picojson::array>());
//		// p_matl->ks = vector3FromArray(it->get("ks").get<picojson::array>());
//		// p_matl->diffuseCoefficient = it->get("diffuse").get<double>();
//		// p_matl->ni = it->get("ni").get<double>();
//		// p_matl->ns = it->get("roughness").get<double>();
//		// matl.bsdf = LAMBERT;
//		// cout << "Found material " << name << endl;
//		// cout << "kd : " << matl.kd.x << ", " << matl.kd.y << ", " << matl.kd.z << endl;
//		// cout << "ks : " << matl.ks.x << ", " << matl.ks.y << ", " << matl.ks.z << endl;
//		// cout << "ka : " << matl.ka.x << ", " << matl.ka.y << ", " << matl.ka.z << endl;
//		// cout << "Diffuse: " << matl.diffuseCoefficient << endl;
//		// cout << "Ni: " << matl.ni << endl;
//		// cout << "Ns: " << matl.ns << endl;
//		// if (matl.bsdf == DIFFSPEC && matl.diffuseCoefficient == 0.0f) {
//		// 	matl.bsdf = SPECULAR;
//		// }
//		// else if (matl.bsdf == DIFFSPEC && matl.diffuseCoefficient == 1.0f) {
//		// 	matl.bsdf = LAMBERT;
//		// }
//		break;
//	}
//	//matl.diffuseCoefficient = update.diffuseCoefficient;
//	//matl.ni = update.ni;
//	// matl.bsdf = update.bsdf;
//	//matl.ns = update.ns;
//}

Vector3Df vector3FromArray(picojson::array arr) {
	Vector3Df retVal;
	int idx = 0;
	for (picojson::array::iterator it = arr.begin(); it != arr.end(); it++)
	{
		float val = it->get<double>();
		retVal._v[idx] = val;
		idx++;
	}
	return retVal;
}
