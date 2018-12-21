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
private:
	GLFWwindow* createWindow(int width, int height, std::string name);
};


#endif
