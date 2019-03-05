// Implementation for Scene functions. This file is responsible for setting up the scene for rendering

#include "linalg.h"
#include "bvh.h"
#include "loaders.h"
#include "scene.h"

#include <algorithm>
#include <cfloat>
#include <iostream>
#include <map>
#include <math.h>
#include <string>

using std::vector;
using std::string;
using std::map;

#define LIGHTS_GAIN 3.0f

unsigned int populateMaterialsMap(vector<objl::Mesh> meshes);
Material materialFromMtl(objl::Material m);

static vector<Material> materialsList;
static map<Material, uint, materialComparator> materialsMap;

const static std::map<std::string, refl_t> reflDict = {
		{"LAMBERT", DIFFUSE},
		{"SPECULAR", SPECULAR},
		{"DIFFSPEC", DIFFSPEC},
		{"MICROFACET", MICROFACET},
		{"REFRACTIVE", REFRACTIVE},
		{"EMISSIVE", EMISSIVE}
};

Material materialFromMtl(objl::Material m);
unsigned int populateMaterialsMap(vector<objl::Mesh> meshes);

// Note - this function should be called before the other load functions
void Scene::loadObj(string filename) {
	meshLoader = objl::Loader();
	std::cout << "Loading single .obj as scene from " << filename << std::endl;
	if (!meshLoader.LoadFile(filename)) {
		std::cerr << "Failed to load mesh for " << filename << std::endl;
	}
}

void Scene::loadCamera(string cameraPath, pixels_t width, pixels_t height) {
	CameraJsonLoader loader(cameraPath);
	p_camera = new Camera();
	Camera camera = loader.getCamera(width, height);
	*p_camera = camera;
}

void Scene::loadTextures(std::string texturesPath) {
	auto materials = meshLoader.LoadedMaterials;
	for (auto material: materials) {
		if (!material.map_Kd.empty()) {
			auto foundTex = std::find(textureFiles.begin(), textureFiles.end(), material.map_Kd);
			if (foundTex == textureFiles.end()) {
				textureFiles.push_back(material.map_Kd);
			}
		}
	}
	p_textureStore = new TextureStore();
	vector<string> fullTextureFiles;
	for (string file: textureFiles) {
		fullTextureFiles.push_back(texturesPath + file);
	}
	p_textureStore->loadAll(&fullTextureFiles[0], textureFiles.size());
}

// This function should be called after loadTextures
void Scene::loadTriangles() {
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

	if (p_textureStore == NULL) {
		throw std::runtime_error("p_textureStore is null!");
	}
	unsigned triId = 0;
	for (auto const& mesh: meshes) {
		objl::Material material = mesh.MeshMaterial;
		Material m = materialFromMtl(material);
		auto it = std::find(materialsList.begin(), materialsList.end(), m);
		int materialId = it - materialsList.begin();
		if(!material.map_Kd.empty()) {
			auto foundTex = std::find(textureFiles.begin(), textureFiles.end(), material.map_Kd);
			if (foundTex != textureFiles.end()) {
				p_materials[materialId].texKdIdx = foundTex - textureFiles.begin();
			}
		}

		vector<objl::Vertex> vertices = mesh.Vertices;
		vector<unsigned> indices = mesh.Indices;
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
			p_current->_uv1 = Vector2Df(v1.TextureCoordinate);
			p_current->_uv2 = Vector2Df(v2.TextureCoordinate);
			p_current->_uv3 = Vector2Df(v3.TextureCoordinate);
			p_current->_e1 = _v2 - _v1;
			p_current->_e2 = _v3 - _v1;

			p_current->_materialId = materialId;
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
	p_triangles = p_tris;
}

void Scene::constructBvh() {
	p_vertexIndices = &meshLoader.LoadedIndices[0];
	p_vertices = &meshLoader.LoadedVertices[0];
	// This is a hook into bvh.cpp.
	// todo: find a more graceful way to do this
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
	material.ka = m.Ka * LIGHTS_GAIN;
	material.kd = m.Kd;
	material.ks = m.Ks;
	material.ns = m.Ns;
	material.ni = m.Ni;
	material.diffuseCoefficient = m.diffuse;
	material.bsdf = DIFFUSE;

	auto it = reflDict.find(m.type);
	if (it != reflDict.end()) {
		material.bsdf = it->second;
	}
	return material;
}

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
