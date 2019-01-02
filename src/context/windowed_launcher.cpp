#include "launcher.h"

WindowedLauncher::WindowedLauncher(Renderer* _renderer, const char* _outFilename)
	: Launcher(_renderer, _outFilename)
{
	int width = p_renderer->getWidth();
	int height = p_renderer->getHeight();
	bool useCuda = p_renderer->useCuda;
	p_windowManager = new WindowManager(width, height, "CudaRT - Cuda Path Tracer by Tudor Boran", useCuda);
}

void WindowedLauncher::render() {
	p_windowManager->mainWindowLoop(p_renderer);
}
