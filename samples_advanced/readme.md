
## 📸 Example: Extracting Name, ID, Diagnosis, and Body Site from Dermatology Images

This example demonstrates how to process dermatology images using `vl-gpu.exe` in two steps to extract:

- **Patient name and registration number (ID)**
- **Diagnosis (Dx)**
- **Body site (BodySite)**

### 🔧 Requirements

- A GGUF model (e.g., `Qwen2-VL-72B-Instruct-Q4_K_M.gguf`)
- A vision encoder model (e.g., `Qwen2-VL-72B-Instruct-vision-encoder.gguf`)
- A folder containing dermatology images (e.g., `D:\qwen\tt`)

### 🧩 Step-by-Step Commands

#### 1️⃣ Step 1: Extract Patient Name, ID, and Diagnosis

This command:
- Extracts `Name` and `ID` from each image.
- Confirms if diagnosis-related information is present.
- Extracts diagnosis (`Dx`) if available.
- Saves all metadata in `result.csv` and logs in `log.txt`.

```bash
vl-gpu.exe \
  -m ./gguf/Qwen2-VL-72B-Instruct-Q4_K_M.gguf \
  --mmproj ./gguf/Qwen2-VL-72B-Instruct-vision-encoder.gguf \
  --temp 0.1 \
  -p "Extract the patient's name and registration number. Response must be in JSON format ('Name','ID')." \
  --index-confirm-prompt "Does it include patient's name and registration number? Response must be YES or NO" \
  --json-meta-list "Dx,BodySite" \
  --custom-confirm-prompt "Does it include diagnosis in dermatology? Response must be YES or NO" \
  --custom-prompt "Extract and list all diagnoses. Response must be in JSON format ('Dx')." \
  --organize-photo \
  --image D:\qwen\tt
```

#### 2️⃣ Step 2: Extract Body Site Information

This command:
- Analyzes the same images to determine which body part is shown.
- Merges new metadata with the existing results from Step 1.
- Updates `result.csv` and `log.txt`.

```bash
vl-gpu.exe \
  -m ./gguf/Qwen2-VL-72B-Instruct-Q4_K_M.gguf \
  --mmproj ./gguf/Qwen2-VL-72B-Instruct-vision-encoder.gguf \
  --temp 0.1 \
  --json-meta-list "Dx,BodySite" \
  --custom-prompt "Which part of the body is it? The response must be in JSON format ('BodySite')." \
  --merge-metadata \
  --custom-prompt-for C \
  --organize-photo \
  --image D:\qwen\tt
```

### 🗂 Output

- `result.csv` – Contains all extracted metadata (Name, ID, Dx, BodySite).
- `log.txt` – Processing logs and prompt-response pairs for auditing or debugging.
