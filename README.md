This CUDA-Accelerated Path Tracer was written by Tudor Matei Boran
for CS-GA: 3801 - GPU Programming for the Fall 2018 semester at NYU Courant

Eventually, the script to unpack and compile this program will be included
in this GitHub repository. Until then, use the following instructions to compile
and run this software.

On a Linux device,

To compile this program, unzip ./CudaRT.tar.gz to a directory with 

`$ tar xzvh ./CudaRt.tar.gz`

then 

`$ cd CudaRT/Release`

then 

`make all`

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
				
				default '../meshes/cornell.obj' - 36 triangles
				
				'../meshes/cornellDonut.obj' - 836 triangles
				
				'../meshes/monkeyBox.obj' - 3972 triangles
				
				'../meshes/dragonBox.obj' - 100012 triangles
				
`-b`:  	 			this flag tells the engine whether or not to use 
				the bounding volume heirarchy acceleration structure
				during ray-scene traversal
				
`-s <samples per pixel>`:	number of samples per pixel. Default = 10. 
				reccommendation is to use at least 128, but
				512 generates a fairly variance/noise-free image

`-w <width (pixels) of output>`

`-h <height (pixels) of output>`

`--cpu`:			flag to use sequential CPU version instead of parallel GPU version

Note that the time complexity of this renderer is O(T\*S\*W\*H) without Bounding Volume
Heirarchy, and O(log(T)\*S\*W\*H) using Bounding Volume Heirarchy, where

T = triangles

S = samples per pixel

W = width in pixels of output

H = height in pixels of output

The actual performance gain by using the BVH is proportional to the size on-screen that the geometry of the scene occupies. 
