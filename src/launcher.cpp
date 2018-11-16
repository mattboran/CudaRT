// This file is responsible for parsing command line arguments,
// getting the scene set up, and launching the pathtrace kernel

#include "pathtrace.h"
#include "scene.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <time.h>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stbi_save_image.h"
using namespace std;
using namespace geom;

static void saveImageToPng(string filename, int width, int height, const Vector3Df* data);

int main(int argc, char* argv[]) {
	string outFile;
	int samples = 10;
	int width = 768;
	int height = 512;
	string objPath = "../meshes/cornell.obj";
	bool multipleObjs = false;
	bool useTextureMemory = false;

	//
	//	Parse command line arguments
	// TODO: config.json instead of command line args
	//
	if (argc < 3) {
		cerr << "Usage: CudaRT <options>\n" \
				"-o \t<output file>\n" \
				"-s \t<number of samples>\tdefault:1000\n" \
				"-w \t<width>\tdefault:800px\n" \
				"-h \t<height>\tdefault:400px\n" \
				"-f \t<path to .obj to render>\tdefault:./meshes/cornell.obj\n" \
				"-F \t<path to .obj directory>\tdefault:./meshes\n" \
				"-t \t<use texture memory for triangles, default = false>"
				;
		return(1);
	}
	vector<string> args(argv + 1, argv + argc);

	// Output
	if (!(find(args.begin(), args.end(), "-o") < args.end() - 1)) {
		cerr << "-o and filename is required!" << endl;
		return(1);
	} else {
		outFile = *(find(args.begin(), args.end(), "-o") + 1);
	}

	// Samples
	if ((find(args.begin(), args.end(), "-s") < args.end() - 1)) {
		try {
			samples = stoi(*(find(args.begin(), args.end(), "-s") + 1));
		} catch (invalid_argument& e) {
			cerr << "Invalid argument to -s!" << endl;
		}
	}

	// Width
	if ((find(args.begin(), args.end(), "-w") < args.end() - 1)) {
		try {
			width = stoi(*(find(args.begin(), args.end(), "-w") + 1));
			if (width % (blockWidth * blockWidth) != 0) {
				cout << "Width should be a multiple of " << blockWidth \
						<< ". You may see something weird happen because of this!" << endl;
			}
		} catch (invalid_argument& e) {
			cerr << "Invalid argument to -w!" << endl;
		}
	}

	// Height
	if ((find(args.begin(), args.end(), "-h") < args.end() - 1)) {
		try {
			height = stoi(*(find(args.begin(), args.end(), "-h") + 1));
			if (height % (blockWidth * blockWidth) != 0) {
				cout << "Height should be a multiple of " << blockWidth \
						<< ". You may see something weird happen because of this!" << endl;
			}
		} catch (invalid_argument& e) {
			cerr << "Invalid argument to -h!" << endl;
		}
	}

	// .obj path
	if ((find(args.begin(), args.end(), "-F") < args.end() - 1)) {
		objPath = *(find(args.begin(), args.end(), "-F") + 1);
		multipleObjs = true;
		cerr << "Can't load multiple OBJs as of yet!" << endl;
	}

	if ((find(args.begin(), args.end(), "-f") < args.end() - 1)) {
		objPath = *(find(args.begin(), args.end(), "-f") + 1);
		multipleObjs = false;
	}

	if ((find(args.begin(), args.end(), "-t") < args.end())) {
		useTextureMemory = true;
	}

	cout << "Samples: " << samples << endl \
			<< "Width: " << width << endl \
			<< "Height: " << height << endl \
			<< "Obj path: " << objPath << endl \
			<< "Output: " << outFile << endl;

	//
	// Initialize Scene : Load .obj
	//
	Scene scene = Scene(objPath);
	cout << "\nLoaded " << scene.getNumMeshes() << " meshes " << endl;
	for (int i = 0; i < scene.getNumMeshes(); i++) {
		objl::Mesh mesh = scene.getMesh(i);
		cout << "Mesh " << i << ": `" << mesh.MeshName << "`"<< endl \
				<< "\t" << mesh.Vertices.size() << " vertices | \t" << mesh.Vertices.size() / 3 << " triangles" << endl;
	}

	cout << "Total number of triangles:\t" << scene.getNumTriangles() << endl;

	//
	// Initialize Scene : Camera
	// TODO: Load this from .obj using cam meshes
	// alternatively, use a camera.json file
	//
	Vector3Df camPos(14.0f, 5.0f, 0.0f);
	Vector3Df camTarget(0.0f, 5.0f, 0.0f);
	Vector3Df camUp(0.0f, 7.0f, 0.0f);
	Vector3Df camRt(-1.0f, 0.0f, 0.0f);

	Camera camera = Camera(camPos, camTarget, camUp, camRt, 90.0f, width, height);
	scene.setCamera(camera);
	clock_t start = clock();
	Vector3Df* imgData = pathtraceWrapper(scene, width, height, samples, useTextureMemory);
	saveImageToPng(outFile, width, height, imgData);
	clock_t end = clock();

	cout << "Sizeof(triangle) = " << sizeof(Triangle) << endl;
	cout << "\nOutput to " << outFile << ":\t\t" << ((double)(end - start))/ CLOCKS_PER_SEC << " seconds " << endl;

	delete[] imgData;
	return(0);
}


static void saveImageToPng(string filename, int width, int height, const Vector3Df* data) {
	const unsigned comp = 4;
	const unsigned strideBytes = width * 4;
	unsigned char* imageData = new unsigned char[width * height * comp];

	unsigned char* currentPixelPtr = imageData;
	for (int i = 0; i < width * height; i++) {
		Vector3Df currentColor = data[i] * 255;
		*currentPixelPtr = (unsigned char)currentColor.x; currentPixelPtr++;
		*currentPixelPtr = (unsigned char)currentColor.y; currentPixelPtr++;
		*currentPixelPtr = (unsigned char)currentColor.z; currentPixelPtr++;
		*currentPixelPtr = (unsigned char)255u; currentPixelPtr++;
	}
	stbi_write_png(filename.c_str(), width, height, comp, imageData, strideBytes);
	delete[] imageData;
}

