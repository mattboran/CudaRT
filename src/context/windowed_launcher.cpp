#include "launcher.h"

WindowedLauncher::WindowedLauncher(Renderer* _renderer, const char* _outFilename)
	: Launcher(_renderer, _outFilename)
{
	int width = p_renderer->getWidth();
	int height = p_renderer->getHeight();
	windowManager = WindowManager(width, height, "CudaRT - Cuda Path Tracer by Tudor Boran");
}
