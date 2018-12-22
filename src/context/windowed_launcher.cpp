#include "launcher.h"

WindowedLauncher::WindowedLauncher(Renderer* _renderer) : BaseLauncher(_renderer) {
	int width = renderer->getWidth();
	int height = renderer->getHeight();
	windowManager = WindowManager(width, height, "CudaRT - Cuda Path Tracer by Tudor Boran");
}
