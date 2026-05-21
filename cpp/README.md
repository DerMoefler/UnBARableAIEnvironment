# UnBARableAIEnvironment: C++
This folder contains the C++ contents of the repo.

## Installation
First, you require a toolchain to build C++ software, meaning a compiler, linker, build system (cmake), ...
This is fairly simple to get on Linux. On Windows, it sucks ass. I can recommend using msys2 (https://www.msys2.org/) which uses pacman (the arch linux package manager) to acquire build tools.
### Conan
You should already have conan installed, otherwise run:
```console
uv tool install conan
```
If this is your first time using conan, you have to supply a profile to conan (containing compiler information etc.). You can just use the default like this (the force is probably optional):
```console
conan profile detect (--force)
```
Personally, I also had to create a file *~/.conan2/setting_user.yml* containing
```
compiler:
  gcc:
    version: ["16"]
```
since my compiler version is newer than conans list of versions.

Then, you are ready to run conan:
```console
conan install . --build=missing -s build_type=Release
```
### CMake
Conan generates build instructions for CMake, which CMake has to be told about. To do that, run:
```console
cmake -S . -B build/Release \
  -DCMAKE_TOOLCHAIN_FILE=build/Release/generators/conan_toolchain.cmake \
  -DCMAKE_BUILD_TYPE=Release
```
You can then invoke CMake as you usually would:
```console
cmake --build build/Release -j
```
VS Code offers an extension for CMake. VS Code might automatically prompt you to select a preset. It also allows running CMake (autoruns when something changes in CMakeLists.txt for example). 