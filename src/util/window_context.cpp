#include "window_context.h"

#include <cuda.h>
#include <cuda_runtime.h>

//Shader code
const GLchar* vertex_shader_code =
	"#version 450 core\n"
	"layout (location = 0) in vec3 position;"
	"layout (location = 1) in vec2 texCoord;"
	"	"
	"out vec2 f_texCoord;"
	"void main()"
	"{"
	"	gl_Position = vec4(position, 1.0f);"
	"   f_texCoord = texCoord;"
	"}";
const GLchar* fragment_shader_code =
	"#version 450 core\n"
	"out vec4 frag_color;"
	"   "
	"in vec2 f_texCoord;"
	"   "
	"uniform sampler2D texSampler;"
	"void main()"
	"{"
	"    frag_color = texture(texSampler, f_texCoord);"
	"    frag_color.xyz = vec3(frag_color.x*frag_color.x, frag_color.y*frag_color.y, frag_color.z*frag_color.z);"
	"}";


GLFWwindow* WindowManager::createWindow(int width, int height, std::string name){
	initWindow();
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

void WindowManager::initWindow() {
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
	glfwWindowHint(GLFW_OPENGL_COMPAT_PROFILE, GL_TRUE);
	glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);
}
