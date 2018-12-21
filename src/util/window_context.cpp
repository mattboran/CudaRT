#include "window_context.h"

#include <cuda.h>
#include <cuda_runtime.h>

void initGLFW();

GLFWwindow* WindowManager::createWindow(int width, int height, std::string name){
	GLFWwindow* window = glfwCreateWindow(width, height, name.c_str(), NULL, NULL);
	int windowWidth, windowHeight;
	glfwGetFramebufferSize(window, &windowWidth, &windowHeight);
	if (window == NULL) {
		printf("Failed to create GLFW window.\n");
		return NULL;
	}
	glfwMakeContextCurrent(window);
	glewExperimental = GL_TRUE;
	if (GLEW_OK != glewInit()) {
		printf("Failed to initialize GLEW\n");
		return NULL;
	}
	glViewport(0, 0, windowWidth, windowHeight);


}

void initGLFW() {
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
	glfwWindowHint(GLFW_OPENGL_COMPAT_PROFILE, GL_TRUE);
	glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);
}
