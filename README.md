## Memory Requirements
To ensure optimal performance, at least **64GB of RAM** is recommended. If your system has less memory, you may experience slow processing times or application crashes.

## Microsoft Visual C++ Redistributable
This application requires an updated version of the **Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017, 2019, and 2022**.

If you encounter an issue related to missing or outdated redistributable packages, please download the latest version from the official Microsoft website:

[Download Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170#visual-studio-2015-2017-2019-and-2022)

## Model Download
The application requires GGUF model files to function correctly. If the `./gguf` directory does not contain any `.gguf` files, please download the required models from Hugging Face.

### Qwen2-VL 72B Model:
- Repository: [Qwen2-VL-72B-Instruct-GGUF](https://huggingface.co/second-state/Qwen2-VL-72B-Instruct-GGUF)
- Required files:
  - `Qwen2-VL-72B-Instruct-Q4_K_M.gguf`
  - `Qwen2-VL-72B-Instruct-vision-encoder.gguf`

### Qwen2-VL 7B Model:
- Repository: [Qwen2-VL-7B-Instruct-GGUF](https://huggingface.co/second-state/Qwen2-VL-7B-Instruct-GGUF)
- Required files:
  - `Qwen2-VL-7B-Instruct-Q4_K_M.gguf`
  - `Qwen2-VL-7B-Instruct-vision-encoder.gguf`

Download and place the appropriate files inside the `./gguf` directory.

## Unicode Support Issue
If you receive an error related to Unicode support, ensure that your system is using **UTF-8** encoding.

### How to Enable UTF-8 Support:

![RegionSetting](https://whria78.github.io/medicalphoto/imgs/RegionSetting.png)

1. Open **Settings** > **Time & Language** > **Language & region**.
2. Click **Administrative language settings**.
3. Under **Language for non-Unicode programs**, click **Change system locale**.
4. Check **Beta: Use Unicode UTF-8 for worldwide language support**.
5. Restart your computer.

If you continue to experience issues, please contact support or refer to the documentation.


---

# How to Run

Execute `VLRunner.exe`.  
The basic execution command is saved in `VLRunner.txt`:

```
vl.exe -m ./gguf/Qwen2-VL-72B-Instruct-Q4_K_M.gguf --mmproj ./gguf/Qwen2-VL-72B-Instruct-vision-encoder.gguf --temp 0.1 -p "Extract the patient's name and registration number. Response must be in JSON format ('Name','ID')." -t 16 --organize-photo --image
```

### GPU Acceleration  
If your **GPU has more than 12GB of VRAM**, you can replace `vl.exe` with `vl-gpu.exe` for faster execution.

### Running the Application  
1. Click the **Browse** button to select the folder containing the images.  
2. Click **Run** to execute the stored command.  

### Performance Estimations  
- **High-end GPU (e.g., RTX 3080 Ti)** → `vl-gpu.exe` takes about **1 minute per image**.  
- **Low-end GPU (e.g., GTX 1050 Ti)** → `vl.exe` takes about **10 minutes per image**.  
- **CPU Execution** → `vl.exe` takes about **15–20 minutes per image**.  

### Output  
- The results are saved as `folder_name.json` inside the selected folder.  
- Clinical photos are stored in the `/RESULT` folder, organized by date.


# How to Build
## Prerequisites
Make sure you have the following tools installed:

- **Visual Studio 2022 Community Edition** (Including C++ Development Tools)
- **CMake** ([Download](https://cmake.org/download/))
- **Git** ([Download](https://git-scm.com/downloads))

Clone the Repository

```sh
git clone https://github.com/whria78/llama-qwen-vl
```

---

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

### Build Output
- DLL files: `..../opencv/install/x64/vc17/bin/opencv_world4120.dll`
- LIB files: `..../opencv/install/x64/vc17/lib/opencv_world4120.lib`

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

Run the following command to build ONNX Runtime:

   ```sh
   .\build.bat --config Release --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync
   ```

   - The default CMake generator on Windows is **Visual Studio 2022**.
   - Other Visual Studio versions are **not supported**.

## Build Output
- DLL files: `..../build/Windows/Release/Release/onnxruntime.dll`, `..../build/Windows/Release/Release/onnxruntime_providers_shared.dll`
- LIB files: `..../build/Windows/Release/Release/onnxruntime.lib`

---

### Note
- Ensure that your **Python interpreter** is a **64-bit Windows application**.
- **32-bit builds are no longer supported.**


---


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
cmake.exe -S . -B build -DGGML_CCACHE=OFF -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=ON -DCMAKE_GENERATOR_TOOLSET="cuda=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
```


## 3. Build llama.cpp
Run the following command to compile the project:
```sh
cmake.exe --build build --config Release -j 8
```
