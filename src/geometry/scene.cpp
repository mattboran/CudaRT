// Implementation for Scene functions. This file is responsible for setting up the scene for rendering

#include "bvh.h"
#include "linalg.h"
#include "scene.h"

#include <algorithm>
#include <iostream>
#include <cfloat>
#include <math.h>

using std::vector;


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

Triangle* Scene::loadTriangles() {
	Triangle* p_tris = (Triangle*)malloc(sizeof(Triangle) * getNumTriangles());
	Triangle* p_current = p_tris;
	vector<objl::Mesh> meshes = meshLoader.LoadedMeshes;
	unsigned triId = 0;
	for (auto const& mesh: meshes) {
		vector<objl::Vertex> vertices = mesh.Vertices;
		vector<unsigned> indices = mesh.Indices;
		objl::Material material = mesh.MeshMaterial;
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

			// Materials
			p_current->_colorDiffuse = Vector3Df(material.Kd);
			p_current->_colorSpec = Vector3Df(material.Ks);
			p_current->_colorEmit = Vector3Df(material.Ka);

			p_current->_surfaceArea = cross(p_current->_e1, p_current->_e2).length()/2.0f;
			p_current->_triId = triId++;

			if (p_current->_colorEmit.lengthsq() > 0.0f) {
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
