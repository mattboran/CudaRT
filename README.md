This CUDA-Accelerated Path Tracer was written by Tudor Matei Boran
originally for CS-GA: 3801 - GPU Programming for the Fall 2018 semester at NYU Courant

It continued on as a personal project where I added many more features and optimizations.

To compile, you need CMake and libraries for OpenGL, GLFW, and glew3 installed.
Conditional compilation will be included in a future release where CMake will compile 
a command-line only version of this program without these dependencies. 

Clone this repository, and `cd CudaRT`

Create a folder called `build` with `mkdir build` and `cd build`

Then run `cmake ..` and `make all`

To run this program, the following pattern should be used while still in the `CudaRT/build` directory
`./CudaRT <parameters>`

CudaRT comes with a Maya plugin, whose source code can be found at https://github.com/mattboran/CudaRT-Maya-Plugin

Ensure that you have the `.env` file in the parent directory containing the paths:

`MESHES_PATH`: the scene to be loaded references `<scene>.obj` and `<scene>.mtl`
Note that the .mtl files used are different. They will be named `.cmtl` at some point soon
because they contain ray-tracing specific material properties that are exported from Maya

`CAMERA_PATH`: the directory that contains the `<scene>-camera.json` file describing the camera,
also exported from Maya. 

`TEXTURES_PATH`: the directory containing the textures referenced by `<scene>.mtl`

Required Parameter:
`-f <scene>`: the scene that should be loaded
Optional Parameters:
`-s <samples per pixel>`:	number of samples per pixel. Default = 10. 
				reccommendation is to use at least 128, but
				512 generates a fairly variance/noise-free image

`-w <width (pixels) of output>`

`-h <height (pixels) of output>`

`--cpu`:			flag to use sequential CPU version instead of parallel GPU version

`--x`: 				flag to use OpenGL to render to window. Saves to output file on close.
