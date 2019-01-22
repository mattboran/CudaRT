This CUDA-Accelerated Path Tracer was written by Tudor Matei Boran
for CS-GA: 3801 - GPU Programming for the Fall 2018 semester at NYU Courant

To compile, you need CMake and libraries for OpenGL, GLFW, and glew3 installed.
Conditional compilation will be included in a future release where CMake will compile 
a command-line only version of this program without these dependencies. 

Clone this repository, and `cd CudaRT`

Create a folder called `build` with `mkdir build` and `cd build`

Then run `cmake ..` and `make all`

To run this program, the following pattern should be used while still in the `CudaRT/Release` directory
`./CudaRT <parameters>`

To see the parameters that this program accepts, simply run 
`./CudaRT`

Required Parameter:
`-o <output file.png>`: the PNG file that the output should be written to

Optional Parameters:
`-f <path to mesh to render>` : the .obj file that will be loaded and rendered
				note: the available files are in CudaRT/meshes/
				so this path should take the form '../meshes/<file>`
				
`-c`:  	 			this flag tells the engine whether or not to use 
				camera information stored at '../settings/camera.json'
				during ray-scene traversal
				
`-s <samples per pixel>`:	number of samples per pixel. Default = 10. 
				reccommendation is to use at least 128, but
				512 generates a fairly variance/noise-free image

`-w <width (pixels) of output>`

`-h <height (pixels) of output>`

`--cpu`:			flag to use sequential CPU version instead of parallel GPU version

`--X`: 				flag to use OpenGL to render to window. Saves to output file on close.
