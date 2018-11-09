// This file is responsible for parsing command line arguments,
// getting the scene set up, and launching the pathtrace kernel

#include <iostream>
#include <algorithm>
#include <vector>
#include <string>

using namespace std;

int main(int argc, char* argv[]) {
	string outFile;
	int samples = 1000;
	int width = 800;
	int height = 400;
	string objPath = "./meshes/cornell.obj";

	if (argc < 3) {
		cerr << "Usage: CudaRT <options>\n" \
				"-o \t<output file>\n" \
				"-s \t<number of samples>\tdefault:1000\n" \
				"-w \t<width>\tdefault:800px\n" \
				"-h \t<height>\tdefault:400px\n" \
				"-f \t<path to .obj to render>\tdefault:./meshes/cornell.obj\n" \
				"-F \t<path to .obj directory>\tdefault:./meshes\n"
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

	// .obj path
	if ((find(args.begin(), args.end(), "-f") < args.end() - 1)) {
		objPath = *(find(args.begin(), args.end(), "-f") + 1);
	}

	cout << "Samples: " << samples << endl \
			<< "Width: " << width << endl \
			<< "Height: " << height << endl \
			<< "Obj path: " << objPath << endl \
			<< "Output: " << outFile << endl;

	return(0);

}
