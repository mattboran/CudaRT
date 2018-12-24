/*
 * launcher.h
 *
 *  Created on: Dec 22, 2018
 *      Author: matt
 */

#ifndef __LAUNCHER_H__
#define __LAUNCHER_H__

#include "renderer.h"
#include "window.h"

class Launcher
{
protected:
	Launcher(Renderer* _renderer, const char* _outFilename) :
		p_renderer(_renderer), outFilename(_outFilename) {}
	Renderer* p_renderer;
	const char* outFilename;
public:
	virtual ~Launcher() {}
	void saveToImage();
	void render();
};

class WindowedLauncher : public Launcher {
public:
	WindowedLauncher(Renderer* _renderer, const char* _outFilename);
private:
	WindowManager windowManager;
};

class TerminalLauncher : public Launcher {
public:
	TerminalLauncher(Renderer* _renderer, const char* _outFilename) : Launcher(_renderer, _outFilename) {}
};

#endif /* __LAUNCHER_H__ */
