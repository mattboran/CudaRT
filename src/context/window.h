#ifndef __WINDOW_CONTEXT_H__
#define __WINDOW_CONTEXT_H__

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GL/gl.h>
#include <string>
#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "glu32.lib")
class WindowManager {
public:
	WindowManager(int width, int height, std::string name);
	GLFWwindow* window;
	void mainWindowLoop();
private:
	GLFWwindow* createWindow(int width, int height, std::string name);
	void initWindow();
	void initializeVertexBuffer(GLfloat screenQuadVertices[]);
	GLint compileAndLinkShaders();
	void deleteShadersAndBuffers();

	GLfloat screenQuadVertices[30];
	GLuint vboIndex;
	GLuint vaoIndex;
	GLuint texIndex;
	GLuint vertexShader;
	GLuint fragmentShader;
	GLint shaderProgram;

};


#endif
