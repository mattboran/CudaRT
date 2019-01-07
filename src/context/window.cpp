#include "window.h"
#include <iostream>

#define INFO_LOG_BUFFER_SIZE 512

WindowManager::WindowManager(int width, int height, std::string name, bool useCudaFlag) {
	useCuda = useCudaFlag;
	window = createWindow(width, height, name);
}

void WindowManager::mainWindowLoop(Renderer* p_renderer) {
	int width = p_renderer->getWidth();
	int height = p_renderer->getHeight();
	uchar4* p_img = p_renderer->getImgBytesPointer();
	size_t bufferSize = 4 *width*height* sizeof(GLubyte);

	while (!glfwWindowShouldClose(window))
	{
		glBindVertexArray(vaoIndex);
		glUseProgram(shaderProgram);
		if (useCuda) {
			cudaGraphicsMapResources(1, &cudaPboResource, 0);
			cudaGraphicsResourceGetMappedPointer((void**)&p_img, NULL, cudaPboResource);
			p_renderer->renderOneSamplePerPixel(p_img);
			cudaGraphicsUnmapResources(1, &cudaPboResource, 0);
		}
		else {
			p_renderer->renderOneSamplePerPixel(p_img);
			glBufferData(GL_PIXEL_UNPACK_BUFFER, bufferSize, p_img, GL_STREAM_DRAW);
		}
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		glDrawArrays(GL_TRIANGLES, 0, 6);
		glfwSwapBuffers(window);
		glfwPollEvents();
		glClear(GL_COLOR_BUFFER_BIT);
		glClearColor(0.0f, 1.0f, 0.0f, 1.0f);
	}
	glDisable(GL_TEXTURE_2D);
	p_renderer->copyImageBytes();
	deleteShadersAndBuffers();
}

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

	initializeVertexBuffer();
	initializePixelBuffer(width, height);
	GLint shaderCompilationSuccess = compileAndLinkShaders();
	if (!shaderCompilationSuccess) {
		return NULL;
	}
	return window;
}

void WindowManager::initWindow() {
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_OPENGL_ANY_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
	glfwWindowHint(GLFW_OPENGL_COMPAT_PROFILE, GL_TRUE);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
}

void WindowManager::initializeVertexBuffer(){
	GLfloat vertices[] = {
		// positions			// UV
		-1.0f, 1.0f, 0.0f,		0.0f, 0.0f,
		1.0f, 1.0f, 0.0f,		1.0f, 0.0f,
		-1.0f, -1.0f, 0.0f,		0.0f, 1.0f,
		-1.0f, -1.0f, 0.0f,		0.0f, 1.0f,
		1.0f, 1.0f, 0.0f,		1.0f, 0.0f,
		1.0f, -1.0f, 0.0f,		1.0f, 1.0f
	};
	glGenBuffers(1, &vboIndex);
	glGenVertexArrays(1, &vaoIndex);
	glBindVertexArray(vaoIndex);
	glBindBuffer(GL_ARRAY_BUFFER, vboIndex);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	// Setup Vertex attribute pointer
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (void*)0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (void*)(3 * sizeof(GLfloat)));
	glEnableVertexAttribArray(1);
}

void WindowManager::initializePixelBuffer(int width, int height) {
	size_t bufferSize = 4 *width*height* sizeof(GLubyte);
	glGenBuffers(1, &pboIndex);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboIndex);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, bufferSize, 0, GL_STREAM_DRAW);

	glGenTextures(1, &texIndex);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texIndex);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	if(useCuda) {
		std::cout<< "Using CUDA!" << std::endl;
		cudaGraphicsGLRegisterBuffer(&cudaPboResource, pboIndex, cudaGraphicsMapFlagsWriteDiscard);
	}
}

GLint WindowManager::compileAndLinkShaders() {
	GLint success;
	char infoLogBuffer[INFO_LOG_BUFFER_SIZE];

	const GLchar* vertexShaderCode =
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

	const GLchar* fragmentShaderCode =
		"#version 450 core\n"
		"out vec4 frag_color;"
		"   "
		"in vec2 f_texCoord;"
		"   "
		"uniform sampler2D texSampler;"
		"void main()"
		"{"
		"    frag_color = texture2D(texSampler, f_texCoord);"
		"	 float gamma = 2.2f;"
		"    frag_color.xyz = pow(frag_color.xyz, vec3(1.0/gamma));"
		"}";

	vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &vertexShaderCode, NULL);
	glCompileShader(vertexShader);
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
	if (!success) {
		glGetShaderInfoLog(vertexShader, INFO_LOG_BUFFER_SIZE, NULL, infoLogBuffer);
		std::cout << "ERROR: Vertex shader compilation failed." << std::endl << infoLogBuffer << std::endl;
	}

	fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &fragmentShaderCode, NULL);
	glCompileShader(fragmentShader);
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
	if (!success) {
		glGetShaderInfoLog(fragmentShader, INFO_LOG_BUFFER_SIZE, NULL, infoLogBuffer);
		std::cout << "ERROR: Fragment shader compilation failed." << std::endl << infoLogBuffer << std::endl;
	}

	shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);
	glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
	if(!success)
	{
		glGetProgramInfoLog(shaderProgram, INFO_LOG_BUFFER_SIZE, NULL, infoLogBuffer);
		std::cout << "Error linking GLSL program." << std::endl << infoLogBuffer << std::endl;
	}
	return success;
}

void WindowManager::deleteShadersAndBuffers() {
	if (useCuda) {
		cudaGraphicsUnregisterResource(cudaPboResource);
	}
	glDeleteBuffers(1, &pboIndex);
	glDeleteTextures(1, &texIndex);
	glDeleteBuffers(1, &vboIndex);
	glDeleteVertexArrays(1, &vaoIndex);
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);
	glDeleteProgram(shaderProgram);
	glfwTerminate();
}
