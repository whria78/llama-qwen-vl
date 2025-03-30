## Digital Transformation (DX) using Vision-Language AI

![Screenshot-before](https://github.com/whria78/llama-qwen-vl/raw/main/capture-before.PNG)
![Screenshot-after](https://github.com/whria78/llama-qwen-vl/raw/main/capture-after.PNG)

- The sample photos (ID&Name) are available at: https://github.com/whria78/llama-qwen-vl/tree/main/samples
- The sample photos (ID&Name&Dx) are available at: https://github.com/whria78/llama-qwen-vl/tree/main/samples_advanced

---

## Memory Requirements
To ensure optimal performance, at least **64GB of RAM** is recommended. If your system has less memory, you may experience slow processing times or application crashes.

## Microsoft Visual C++ Redistributable

![Screenshot-down-msvc](https://github.com/whria78/llama-qwen-vl/raw/main/capture-down-msvc.PNG)

This application requires an updated version of the **Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017, 2019, and 2022**.

If you encounter an issue related to missing or outdated redistributable packages, please download the latest version from the official Microsoft website:

[Download Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170#visual-studio-2015-2017-2019-and-2022)


## Model Download (Qwen2-VL 72B)
![Screenshot-down-gguf](https://github.com/whria78/llama-qwen-vl/raw/main/capture-down-gguf.PNG)

The application requires GGUF model files to function correctly. If the `./gguf` directory does not contain any `.gguf` files, please download the required models from Hugging Face.

### Qwen2-VL 72B Model:
- Repository: [Qwen2-VL-72B-Instruct-GGUF](https://huggingface.co/second-state/Qwen2-VL-72B-Instruct-GGUF)
- Required files:
  - `Qwen2-VL-72B-Instruct-Q4_K_M.gguf`
  - `Qwen2-VL-72B-Instruct-vision-encoder.gguf`

Download and place the appropriate files inside the `./gguf` directory.


## Unicode Support Issue
If you receive an error related to Unicode support, ensure that your system is using **UTF-8** encoding.

### How to Enable UTF-8 Support:

![Screenshot0](https://github.com/whria78/llama-qwen-vl/raw/main/capture0.png)

1. Open **Settings** > **Time & Language** > **Language & region**.
2. Click **Administrative language settings**.
3. Under **Language for non-Unicode programs**, click **Change system locale**.
4. Check **Beta: Use Unicode UTF-8 for worldwide language support**.
5. Restart your computer.

If you continue to experience issues, please contact support or refer to the documentation.


---


# How to Run  

![Screenshot1](https://github.com/whria78/llama-qwen-vl/raw/main/capture1.PNG)

`VLRunner.exe` provides a GUI interface to execute the following tasks. The default command is stored in `VLRunner.txt`.  cf) `VLRunner - Dx.txt`, `VLRunner - DX SEX AGE SITE.txt`

You can also run it from the Windows **Command Prompt (cmd)** using the following format:  

```
vl.exe -m ./gguf/Qwen2-VL-72B-Instruct-Q4_K_M.gguf --mmproj ./gguf/Qwen2-VL-72B-Instruct-vision-encoder.gguf --temp 0.1 -p "Extract the patient's name and registration number. Response must be in JSON format ('Name','ID')." -t 16 --organize-photo --image [folder_name]
```

**Note:** Replace `[folder_name]` with the actual path of the folder containing the images. Or use the GUI helper (VLRunner.exe). 


### GPU Acceleration  
If you have any **NVIDIA GPU**, you can replace `vl.exe` with `vl-gpu.exe` for faster execution. **`vl-gpu.exe` also works in CPU mode.**

### Performance Estimations  
- **High-end GPU (e.g., RTX 3080 Ti 12GB VRAM)** → `vl-gpu.exe` takes about **less than 1 minute per image**.  
- **Low-end GPU (e.g., GTX 1050 Ti 4GB VRAM)** → `vl-gpu.exe` or `vl.exe` takes about **5 minutes per image**.  
- **CPU Execution** → `vl.exe` or `vl-gpu.exe` takes about **10 minutes per image**.  
**Note:**  **System memory must be at least 64 GB.**

### Output  

- The results are saved as `folder_name.json` inside the selected folder.  
- You can check the results immediately in the `/RESULT` folder, where clinical photos are organized by date.

![Screenshot-tt-json](https://github.com/whria78/llama-qwen-vl/raw/main/capture-tt-json.PNG)

- In the example above, the command is executed in the `D:/qwen/tt` folder, and the output is saved as `tt.json`.  

![Screenshot-medicalphoto1](https://github.com/whria78/llama-qwen-vl/raw/main/capture-medicalphoto1.PNG)

![Screenshot-medicalphoto2](https://github.com/whria78/llama-qwen-vl/raw/main/capture-medicalphoto2.PNG)

- If you upload the `D:/qwen/tt` folder to **MedicalPhoto**, the JSON data will be applied, and the photos will be saved accordingly.  



### Custom Metadata Extraction

- You can extract **diagnoses** using the following command:

```sh
vl-gpu.exe -m ./gguf/Qwen2-VL-72B-Instruct-Q4_K_M.gguf --mmproj ./gguf/Qwen2-VL-72B-Instruct-vision-encoder.gguf --temp 0.1   -p "Extract the patient's name and registration number. Response must be in JSON format ('Name','ID')."   --index-confirm-prompt "Does it include the patient's name and registration number? Response must be YES or NO"   --json-meta-list "Dx"   --custom-confirm-prompt "Does it include a diagnosis in dermatology? Response must be YES or NO"   --custom-prompt "Extract and list all diagnoses. Response must be in JSON format ('Dx')."   --organize-photo --image [folder]
```

🔗 [Sample Images , log, JSON result](https://github.com/whria78/llama-qwen-vl/tree/main/samples_advanced)


The json result is as following:
```json
[
    {
        "Date": "2025:03:28 09:38:40",
        "Dx": [
            "Suspected Dysplastic Nevus",
            "Melanocytic Nevus"
        ],
        "Filename": "[ROOT_PATH]/20250328_093839.jpg",
        "ID": "Not provided",
        "Name": "John Doe",
        "confirm": "",
        "err": "",
        "is_index": true,
        "response": "{\n  \"Name\": \"John Doe\",\n  \"ID\": \"Not provided\"\n}"
    },
    {
        "Date": "2025:03:28 09:40:08",
        "Filename": "[ROOT_PATH]/20250328_094008.jpg",
        "ID": "287282",
        "Name": "John Doe",
        "confirm": "yes",
        "err": "",
        "is_index": true,
        "response": "{\n  \"Name\": \"John Doe\",\n  \"ID\": \"1985\"\n}"
    },
    {
        "Date": "2025:03:28 09:40:19",
        "Dx": [
            "Suspected Dysplastic Nevus",
            "Melanocytic Nevus"
        ],
        "Filename": "[ROOT_PATH]/20250328_094018.jpg",
        "ID": "287282",
        "Name": "John Doe",
        "confirm": "yes",
        "err": "",
        "is_index": true,
        "response": "{\n  \"Name\": \"John Doe\",\n  \"ID\": \"123456\"\n}"
    },
    {
        "Date": "2025:03:28 09:43:41",
        "Dx": [
            "Suspected Dysplastic Nevus",
            "Melanocytic Nevus"
        ],
        "Filename": "[ROOT_PATH]/20250328_094341.jpg",
        "ID": "287282",
        "Name": "John Doe",
        "confirm": "yes",
        "err": "",
        "is_index": true,
        "response": "{\n  \"Name\": \"John Doe\",\n  \"ID\": \"287282\"\n}"
    },
    {
        "Date": "2025:03:28 09:44:05",
        "Dx": [
            "Melanocytic nevus",
            "Dysplastic nevus"
        ],
        "Filename": "[ROOT_PATH]/20250328_094405.jpg",
        "ID": "287282",
        "Name": "John Doe",
        "confirm": "yes",
        "err": "",
        "is_index": true,
        "response": "{\n  \"Name\": \"John Doe\",\n  \"ID\": \"287282\"\n}"
    },
    {
        "Date": "2025:03:28 09:45:05",
        "Dx": [
            "Suspected Dysplastic Nevus",
            "Melanocytic Nevus"
        ],
        "Filename": "[ROOT_PATH]/20250328_094505.jpg",
        "ID": "123456",
        "Name": "John Doe",
        "confirm": "",
        "err": "",
        "is_index": true,
        "response": "{\n  \"Name\": \"John Doe\",\n  \"ID\": \"123456789\"\n}"
    },
    {
        "Date": "2025:03:28 09:45:13",
        "Dx": [
            "Suspected Dysplastic Nevus",
            "Melanocytic Nevus"
        ],
        "Filename": "[ROOT_PATH]/20250328_094513.jpg",
        "ID": "123456",
        "Name": "John Doe",
        "confirm": "",
        "err": "",
        "is_index": true,
        "response": "{\n  \"Name\": \"John Doe\",\n  \"ID\": \"123456\"\n}"
    },
    {
        "Date": "2025:03:28 09:45:30",
        "Dx": [
            "Suspected Dysplastic Nevus",
            "Melanocytic Nevus"
        ],
        "Filename": "[ROOT_PATH]/20250328_094530.jpg",
        "ID": "123456",
        "Name": "John Doe",
        "confirm": "",
        "err": "",
        "is_index": true,
        "response": "{\n  \"Name\": \"John Doe\",\n  \"ID\": \"123456\"\n}"
    },
    {
        "Date": "2025:03:28 09:45:54",
        "Dx": [
            "Melanocytic nevus",
            "Dysplastic nevus"
        ],
        "Filename": "[ROOT_PATH]/20250328_094553.jpg",
        "ID": "287282",
        "Name": "John Doe",
        "confirm": "yes",
        "err": "",
        "is_index": true,
        "response": "{\n  \"Name\": \"John Doe\",\n  \"ID\": \"287282\"\n}"
    },
    {
        "Dx": [
            "Suspected Dysplastic Nevus",
            "Melanocytic Nevus"
        ],
        "Filename": "[ROOT_PATH]/20250328_093859.jpg",
        "ID": "Not provided",
        "Name": "John Doe",
        "confirm": ""
    },
    {
        "Dx": [
            "Suspected Dysplastic Nevus",
            "Melanocytic Nevus"
        ],
        "Filename": "[ROOT_PATH]/20250328_093911.jpg",
        "ID": "Not provided",
        "Name": "John Doe",
        "confirm": ""
    },
    {
        "Dx": [
            "Suspected Dysplastic Nevus",
            "Melanocytic Nevus"
        ],
        "Filename": "[ROOT_PATH]/20250328_093920.jpg",
        "ID": "Not provided",
        "Name": "John Doe",
        "confirm": ""
    },
    {
        "Dx": [
            "Suspected Dysplastic Nevus",
            "Melanocytic Nevus"
        ],
        "Filename": "[ROOT_PATH]/20250328_093939.jpg",
        "ID": "Not provided",
        "Name": "John Doe",
        "confirm": ""
    },
    {
        "Dx": [
            "Melanocytic nevus",
            "Dysplastic nevus"
        ],
        "Filename": "[ROOT_PATH]/20250328_094416.jpg",
        "ID": "287282",
        "Name": "John Doe",
        "confirm": "yes"
    },
    {
        "Dx": [
            "Melanocytic nevus",
            "Dysplastic nevus"
        ],
        "Filename": "[ROOT_PATH]/20250328_094423.jpg",
        "ID": "287282",
        "Name": "John Doe",
        "confirm": "yes"
    },
    {
        "Dx": [
            "Suspected Dysplastic Nevus",
            "Melanocytic Nevus"
        ],
        "Filename": "[ROOT_PATH]/20250328_094538.jpg",
        "ID": "123456",
        "Name": "John Doe",
        "confirm": ""
    },
    {
        "Dx": [
            "Suspected Dysplastic Nevus",
            "Melanocytic Nevus"
        ],
        "Filename": "[ROOT_PATH]/20250328_094540.jpg",
        "ID": "123456",
        "Name": "John Doe",
        "confirm": ""
    },
    {
        "Dx": [
            "Melanocytic nevus",
            "Dysplastic nevus"
        ],
        "Filename": "[ROOT_PATH]/20250328_094607.jpg",
        "ID": "287282",
        "Name": "John Doe",
        "confirm": "yes"
    }
]
```

- You can also extract additional metadata such as **age, sex, and body site**, although the accuracy may be lower:

```sh
vl-gpu.exe -m ./gguf/Qwen2-VL-72B-Instruct-Q4_K_M.gguf --mmproj ./gguf/Qwen2-VL-72B-Instruct-vision-encoder.gguf --temp 0.1   -p "Extract the patient's name and registration number. Response must be in JSON format ('Name','ID')."   --index-confirm-prompt "Does it include the patient's name and registration number? Response must be YES or NO"   --json-meta-list "Dx,Sex,Age,BodySite"   --custom-prompt "Extract all diagnoses, age, sex, and body site. Response must be in JSON format ('Dx','Sex','Age','BodySite')."   --organize-photo --image
```


---

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
cd opencv
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
  cd onnxruntime
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
## Build Output
- EXE files: `..../build/bin/Release/llama-qwen2vl-cli.exe`

## `clip.cpp` GPU Issue  
[#10896](https://github.com/ggml-org/llama.cpp/pull/10896).  