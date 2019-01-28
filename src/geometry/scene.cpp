// Implementation for Scene functions. This file is responsible for setting up the scene for rendering

#include "bvh.h"
#include "linalg.h"
#include "scene.h"

#include <algorithm>
#include <cfloat>
#include <iostream>
#include <map>
#include <math.h>

using std::vector;
using std::map;


unsigned int populateMaterialsMap(vector<objl::Mesh> meshes);
Material materialFromMtl(objl::Material m);

static vector<Material> materialsList;
static map<Material, unsigned int, materialComparator> materialsMap;
// Constructors
Scene::Scene(std::string filename) {
	meshLoader = objl::Loader();
	std::cout << "Loading single .obj as scene from " << filename << std::endl;
	if (!meshLoader.LoadFile(filename)) {
		std::cerr << "Failed to load mesh for " << filename << std::endl;
	}
	p_triangles = loadTriangles();
	vertexIndices = &meshLoader.LoadedIndices[0];
	p_vertices = &meshLoader.LoadedVertices[0];
	constructBVH(this);
}

float Scene::getLightsSurfaceArea() {
	float surfaceArea = 0;
	for (auto light: lightsList) {
		surfaceArea += light._surfaceArea;
	}
	return surfaceArea;
}

Material materialFromMtl(objl::Material m) {
	Material material;
	material.ka = m.Ka;
	material.kd = m.Kd;
	material.ks = m.Ks;
	material.ns = m.Ns;
	material.ni = m.Ni;
	material.bsdf = DIFFUSE;
	if (material.ks.lengthsq() > 0.0f) {
		material.bsdf = SPECULAR;
		if (material.ns > 0.0f) {
			material.bsdf = COOKETORRENCE;
		}
	}
	if (material.ni != 1.0f) {
		material.bsdf = REFRACTIVE;
	}
	if (material.ka.lengthsq() > 0.0f) {
		material.bsdf = EMISSIVE;
	}
	return material;
}

// todo: change this to a set
unsigned int populateMaterialsMap(vector<objl::Mesh> meshes) {
	unsigned int idx = 0;
	for (auto const& mesh: meshes) {
		// TODO: Move this to Material.h
		Material material = materialFromMtl(mesh.MeshMaterial);
		if (materialsMap.count(material) == 0) {
			materialsMap.insert(std::pair<Material, unsigned int>(material, idx));
			idx++;
		}
	}
	return materialsMap.size();
}

unsigned int findMaterial(Material* p_materials, const Material& material) {

}

Triangle* Scene::loadTriangles() {
	Triangle* p_tris = (Triangle*)malloc(sizeof(Triangle) * getNumTriangles());
	Triangle* p_current = p_tris;
	vector<objl::Mesh> meshes = meshLoader.LoadedMeshes;

	// Allocate and populate materials array
	numMaterials = populateMaterialsMap(meshes);
	p_materials = new Material[numMaterials];
	for (auto it = materialsMap.begin(); it != materialsMap.end(); it++) {
		p_materials[it->second] = it->first;
	}
	for (unsigned i = 0; i < numMaterials; i++) {
		materialsList.push_back(p_materials[i]);
	}

	unsigned triId = 0;
	for (auto const& mesh: meshes) {
		vector<objl::Vertex> vertices = mesh.Vertices;
		vector<unsigned> indices = mesh.Indices;
		objl::Material material = mesh.MeshMaterial;
		Material m = materialFromMtl(material);
		auto it = std::find(materialsList.begin(), materialsList.end(), m);
		for (unsigned int i = 0; i < vertices.size()/3; i++) {
			p_current->_id1 = indices[i*3];
			p_current->_id2 = indices[i*3 + 1];
			p_current->_id3 = indices[i*3 + 2];
			objl::Vertex v1 = vertices[indices[i*3]];
			objl::Vertex v2 = vertices[indices[i*3 + 1]];
			objl::Vertex v3 = vertices[indices[i*3 + 2]];
			Vector3Df _v1(v1.Position);
			Vector3Df _v2(v2.Position);
			Vector3Df _v3(v3.Position);
			p_current->_v1 = _v1;
			p_current->_n1 = Vector3Df(v1.Normal);
			p_current->_n2 = Vector3Df(v2.Normal);
			p_current->_n3 = Vector3Df(v3.Normal);
			p_current->_e1 = _v2 - _v1;
			p_current->_e2 = _v3 - _v1;

			p_current->_materialId = it - materialsList.begin();
			// Materials

			p_current->_surfaceArea = cross(p_current->_e1, p_current->_e2).length()/2.0f;
			p_current->_triId = triId++;

			if (m.bsdf == EMISSIVE) {
				lightsList.push_back(*p_current);
			}

			p_current++;
		}
	}
	std::sort(lightsList.begin(), lightsList.end(),
			[](const Triangle &a, const Triangle &b) -> bool {
		return a._surfaceArea > b._surfaceArea;
	});
	return p_tris;
}
