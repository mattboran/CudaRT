#ifndef __WINDOW_CONTEXT_H__
#define __WINDOW_CONTEXT_H__

#include "renderer.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GL/gl.h>
#include "cuda_gl_interop.h"
#include <string>

class WindowManager {
public:
	WindowManager() {}
	WindowManager(int width, int height, std::string name, bool useCudaFlag=false);
	GLFWwindow* window;
	void mainWindowLoop(Renderer* p_renderer);
	~WindowManager() { }
private:
	GLFWwindow* createWindow(int width, int height, std::string name);
	void initWindow();
	void initializePixelBuffer(int width, int height);
	void initializeVertexBuffer();
	GLint compileAndLinkShaders();
	void deleteShadersAndBuffers();

	GLuint vboIndex;
	GLuint vaoIndex;
	GLuint texIndex;
	GLuint pboIndex;
	GLuint vertexShader;
	GLuint fragmentShader;
	GLint shaderProgram;
	bool useCuda;
	struct cudaGraphicsResource *cudaPboResource;
};


#endif
