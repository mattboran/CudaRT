// This file is responsible for parsing command line arguments,
// getting the scene set up, and launching the pathtrace kernel

#include "scene.h"
#include "camera.h"
#include "json_loader.h"
#include "launcher.h"
#include "renderer.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cuda.h>
#include <iostream>
#include <string>
#include <vector>

using namespace std;
using namespace std::chrono;

int main(int argc, char* argv[]) {
	string outFile;
	int samples = 10;
	int width = 640;
	int height = 480;
	string sceneName = "cornell";
	string objPath = "../meshes/" + sceneName + ".obj";
	string cameraPath = "../settings/" + sceneName + "-camera.json";
	string materialsPath = "../settings/" + sceneName + "-materials.json";
	bool useSequential = false;
	bool renderToScreen = false;
	int cudaCapableDevices = 0;

	//
	//	Parse command line arguments
	//
	if (argc < 2) {
		cerr << "Usage: CudaRT <options>\n" \
				"-o \t<output file>\n" \
				"-s \t<number of samples>\tdefault:10\n" \
				"-w \t<width>\tdefault:480px\n" \
				"-h \t<height>\tdefault:320px\n" \
				"-f \t<scene>\tdefault: cornell\n" \
				"--cpu \t<flag to run sequential code on CPU only>\tdefault: false\n" \
				"--X \t<flag to render to screen>\tdefault: false\n" \
				"Note: BVH has bugs in both CUDA and CPU version. CPU version is worse.\n"\
				;
		return(1);
	}
	vector<string> args(argv + 1, argv + argc);

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
		} catch (invalid_argument& e) {
			cerr << "Invalid argument to -w!" << endl;
		}
	}

	// Height
	if ((find(args.begin(), args.end(), "-h") < args.end() - 1)) {
		try {
			height = stoi(*(find(args.begin(), args.end(), "-h") + 1));
		} catch (invalid_argument& e) {
			cerr << "Invalid argument to -h!" << endl;
		}
	}

	// Sequential flag
	if ((find(args.begin(), args.end(), "--cpu") < args.end())) {
		useSequential = true;
	}


	// Render To Screen flag
	if ((find(args.begin(), args.end(), "--X") < args.end())) {
		renderToScreen = true;
	}


	// .obj path
	if ((find(args.begin(), args.end(), "-f") < args.end() - 1)) {
		sceneName = *(find(args.begin(), args.end(), "-f") + 1);
		objPath = "../meshes/" + sceneName + ".obj";
		cameraPath = "../settings/" + sceneName + "-camera.json";
		materialsPath = "../settings/" + sceneName + "-materials.json";
		outFile = sceneName + ".png";
	}

	cout << "Samples: " << samples << endl \
			<< "Width: " << width << endl \
			<< "Height: " << height << endl \
			<< "Obj path: " << objPath << endl \
			<< "Camera path: " << cameraPath << endl \
			<< "Output: " << outFile << endl;

	//
	// Initialize Scene
	//
	JsonLoader loader(cameraPath, materialsPath);
	Scene scene(objPath);
	Camera camera = loader.getCamera(width, height);
	scene.setCameraPtr(&camera);

	for (int i = 0; i < scene.getNumMeshes(); i++) {
		objl::Mesh mesh = scene.getMesh(i);
		cout << "Mesh " << i << ": `" << mesh.MeshName << "`"<< endl \
				<< "\t" << mesh.Vertices.size() << " vertices | \t" << mesh.Vertices.size() / 3 << " triangles" << endl;
	}
	cout << "Total number of triangles:\t" << scene.getNumTriangles() << endl;

	Renderer* p_renderer;
	Launcher* p_launcher;

	cudaGetDeviceCount(&cudaCapableDevices);
	if (useSequential || cudaCapableDevices == 0) {
		p_renderer = new SequentialRenderer(&scene, width, height, samples);
	} else {
		p_renderer = new ParallelRenderer(&scene, width, height, samples);
	}
	if (renderToScreen) {
		p_launcher = new WindowedLauncher(p_renderer, outFile.c_str());
	} else {
		p_launcher = new TerminalLauncher(p_renderer, outFile.c_str());
	}

	auto start = high_resolution_clock::now();
	p_launcher->render();
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);

	p_launcher->saveToImage();

	float elapsedTime = duration.count()/1000000.0f;
	int samplesRendered = p_renderer->getSamplesRendered();
	float averageSamplesPerPixel = samplesRendered / elapsedTime;
	cout << "Rendered to " << outFile << " for " << samplesRendered << " samples per pixel. " << endl;
	cout << "Elapsed time in seconds = " << elapsedTime << endl;
	cout << "Average samples per pixel = " << averageSamplesPerPixel << endl;

	return(0);
}
