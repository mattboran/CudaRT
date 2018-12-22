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

class BaseLauncher
{
protected:
	BaseLauncher(Renderer* _renderer) : renderer(_renderer) {}
	Renderer* renderer;
public:
	virtual ~BaseLauncher() {}
	void saveToImage();
};

class WindowedLauncher : public BaseLauncher {
public:
	WindowedLauncher(Renderer* _renderer);
private:
	WindowManager windowManager;
};

class TerminalLauncher : public BaseLauncher {
	TerminalLauncher(Renderer* _renderer) : BaseLauncher(_renderer) {}
};

#endif /* __LAUNCHER_H__ */
