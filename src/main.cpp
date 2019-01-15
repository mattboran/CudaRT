// This file is responsible for parsing command line arguments,
// getting the scene set up, and launching the pathtrace kernel

#include "scene.h"
#include "launcher.h"
#include "renderer.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cuda.h>
#include <fstream>
#include <iostream>
#include "picojson.h"
#include <streambuf>
#include <string>
#include <vector>

using namespace std;
using namespace std::chrono;

Camera loadCameraFromJson(string path);

int main(int argc, char* argv[]) {
	string outFile;
	int samples = 10;
	int width = 640;
	int height = 480;
	string objPath = "../meshes/cornell.obj";
	bool useCameraJson = false;
	bool useSequential = false;
	bool renderToScreen = false;
	int numStreams = 1;
	int cudaCapableDevices = 0;

	//
	//	Parse command line arguments
	// TODO: config.json instead of command line args
	//
	if (argc < 3) {
		cerr << "Usage: CudaRT <options>\n" \
				"-o \t<output file>\n" \
				"-s \t<number of samples>\tdefault:10\n" \
				"-w \t<width>\tdefault:480px\n" \
				"-h \t<height>\tdefault:320px\n" \
				"-f \t<path to .obj to render>\tdefault:./meshes/cornell.obj\n" \
				"-c \t<flag to use camera/camera.json>\tdefault: false\n" \
				"--cpu \t<flag to run sequential code on CPU only>\tdefault: false\n" \
				"--X \t<flag to render to screen>\tdefault: false\n" \
				"Note: BVH has bugs in both CUDA and CPU version. CPU version is worse.\n"\
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

	// useCameraJson Flag
	if ((find(args.begin(), args.end(), "-c") < args.end())) {
		useCameraJson = true;
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
		objPath = *(find(args.begin(), args.end(), "-f") + 1);
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
	Camera camera;
	if (useCameraJson) {
		camera = loadCameraFromJson("../camera/camera.json");
		return 0;
	} else {
		float scale = 0.1f;
		Vector3Df camPos(14.0f, 5.0f, 0.0f);
		Vector3Df camTarget(0.0f, 5.0f, 0.0f);
		Vector3Df camUp(0.0f, 1.0f, 0.0f);
		Vector3Df camRt(-1.0f, 0.0f, 0.0f);
		camera = Camera(camPos * scale, camTarget * scale, camUp, camRt, 90.0f, width, height);
	}
	Renderer* p_renderer;
	Launcher* p_launcher;

	scene.setCamera(camera);
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
	delete p_renderer;
	delete p_launcher;

	cout << "Rendered to " << outFile << " for " << p_renderer->getSamplesRendered() << " samples per pixel. " << endl;
	cout << "Elapsed time in seconds = " << duration.count()/1000000.0f << endl;

	return(0);
}

Camera loadCameraFromJson(string path) {
	ifstream t(path);
	string json((std::istreambuf_iterator<char>(t)),
	                 std::istreambuf_iterator<char>());
	picojson::value v;
	std::string err = picojson::parse(v, json);
	if (! err.empty()) {
	  std:cerr << err << std::endl;
	}
	const picojson::value::object& obj = v.get<picojson::object>();
	for (picojson::value::object::const_iterator i = obj.begin();
	     i != obj.end();
	     ++i) {
	  std::cout << i->first << " : " << i->second.to_str() << std::endl;
	}
	string name = v.get("name").get<string>();
	array<float> eye = v.get("eye").get<array<float>>();
	cout<< "eye x y z = " << eye[0] << " " << eye[1] << " " << eye[2];
	return Camera();
}
