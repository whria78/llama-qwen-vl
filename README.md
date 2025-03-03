# Clone all repositories
## Prerequisites
Before building OpenCV, make sure you have the following tools installed:

- **Visual Studio 2022 Community Edition** (Including C++ Development Tools)
- **CMake** ([Download](https://cmake.org/download/))
- **Git** ([Download](https://git-scm.com/downloads))

After installation, open **"x64 Native Tools Command Prompt for VS 2022"** to proceed.

```sh
"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
```

Clone the Repository

```sh
git clone https://github.com/whria78/llama.cpp
```

# OpenCV Windows Build Guide (Using CMake & MSVC)

## 1. Configure Build with CMake

Create a build directory and configure the project:

```sh
mkdir build
cd build
```

Run the following CMake command:

```sh
cmake -G "Visual Studio 17 2022" -A x64 -D CMAKE_BUILD_TYPE=Release -D BUILD_opencv_world=ON -D BUILD_SHARED_LIBS=ON ..
```

## 2. Build OpenCV

### **🔹 Build Release Mode**
```sh
cmake --build . --config Release --target INSTALL
```

### 📌 Build Output
- DLL files: `..../opencv/install/x64/vc17/bin/opencv_world490.dll`
- LIB files: `..../opencv/install/x64/vc17/lib/opencv_world490.lib`

---


# Compile ONNXRUNTIME (CPU)

## Basic CPU Build

### Prerequisites

#### Install Python 3.10+
Ensure that you have Python 3.10 or later installed on your system.

#### Install CMake 3.28 or higher

  ```sh
  python3 -m pip install cmake
  which cmake
  ```

## Build Instructions

1. Open **Developer Command Prompt for Visual Studio** (matching the version you intend to use). This ensures the correct environment settings.

2. Run the following command to build ONNX Runtime:
   ```sh
   .\build.bat --config Release --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync
   ```

   - The default CMake generator on Windows is **Visual Studio 2022**.
   - Other Visual Studio versions are **not supported**.

### Note
- Ensure that your **Python interpreter** is a **64-bit Windows application**.
- **32-bit builds are no longer supported.**


# Building llama.cpp with CUDA 12.4 on Windows

## Prerequisites
Before building `llama.cpp`, ensure you have the following installed:

- **CUDA 12.4** ([Download](https://developer.nvidia.com/cuda-downloads))
- **CMake** ([Download](https://cmake.org/download/))
- **Visual Studio 2022** (Including C++ Development Tools)

Additionally, make sure that the CUDA installation path is correct:
```sh
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4
```

---


## 2. Configure CMake
Run the following command to configure the build with CUDA support:
```sh
cmake.exe -S . -B build ^
    -DGGML_CCACHE=OFF ^
    -DBUILD_SHARED_LIBS=ON ^
    -DGGML_CUDA=ON ^
    -DCMAKE_GENERATOR_TOOLSET="cuda=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
```


## 3. Build llama.cpp
Run the following command to compile the project:
```sh
cmake.exe --build build --config Release -j 8
```
