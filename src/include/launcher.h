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
	virtual void render() = 0;
};

class WindowedLauncher : public Launcher {
public:
	WindowedLauncher(Renderer* _renderer, const char* _outFilename);
	void render();
private:
	WindowManager* p_windowManager;
};

class TerminalLauncher : public Launcher {
public:
	TerminalLauncher(Renderer* _renderer, const char* _outFilename) : Launcher(_renderer, _outFilename) {}
	void render();
};

#endif /* __LAUNCHER_H__ */
