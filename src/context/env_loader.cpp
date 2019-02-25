#include "loaders.h"

#include <exception>
#include <fstream>
#include <iostream>

using namespace std;

static const string kMeshesPath = "MESHES_PATH";
static const string kCameraPath = "CAMERA_PATH";
static const string kTexturesPath = "TEXTURES_PATH";
static const string gEnvKeys[] {kMeshesPath, kCameraPath, kTexturesPath};

EnvLoader::EnvLoader(string envPath) {
    fstream dotenv(envPath);
    if (!dotenv.is_open()) {
        string err = "Error opening " + envPath;
        throw runtime_error(err.c_str());
    }
    string line;
    size_t numKeys = sizeof(gEnvKeys) / sizeof(gEnvKeys[0]);
    while(getline(dotenv, line)) {
        if (line[0] == '#' || line[0] == '\n') {
            continue;
        }
        for (size_t i = 0; i < numKeys; i++) {
            string currKey = gEnvKeys[i];
            size_t foundKey = line.find(currKey);
            if (foundKey != string::npos) {
                size_t keyLength = currKey.length() + 1;
                string value = line.substr(keyLength);
                settingsDict[currKey] = value;
                break;
            }
        }
    }
}

string EnvLoader::getMeshesPath() {
    return settingsDict[kMeshesPath];
}

string EnvLoader::getCameraPath() {
    return settingsDict[kCameraPath];
}

string EnvLoader::getTexturesPath() {
    return settingsDict[kTexturesPath];
}
