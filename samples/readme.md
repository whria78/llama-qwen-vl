
## 📸 Simple Example: Extracting ID, Name, and Diagnosis from Indexed Photos

This example demonstrates how to process indexed dermatology photos using `vl-gpu.exe` to extract:

- **Patient (Name) and Registration Number (ID)**
- **Diagnosis (Dx)**

### 🔧 Requirements

- A GGUF model (e.g., `Qwen2-VL-72B-Instruct-Q4_K_M.gguf`)
- A vision encoder model (e.g., `Qwen2-VL-72B-Instruct-vision-encoder.gguf`)
- A folder containing dermatology images (e.g., `D:\qwen\tt`)

### 🧩 Command

This command:
- Checks if an image contains `Name` and `ID`.
- Checks if an image contains diagnosis information.
- Extracts `Name`, `ID`, and `Dx` using the VL model.
- Saves the results in `result.csv` and logs the process in `log.txt`.

```bash
vl-gpu.exe \
  -m ./gguf/Qwen2-VL-72B-Instruct-Q4_K_M.gguf \
  --mmproj ./gguf/Qwen2-VL-72B-Instruct-vision-encoder.gguf \
  --temp 0.1 \
  -p "Extract the patient's name and registration number. Response must be in JSON format ('Name','ID')." \
  --index-confirm-prompt "Does it include patient's name and registration number? Response must be YES or NO" \
  --json-meta-list "Dx" \
  --custom-confirm-prompt "Does it include diagnosis in dermatology? Response must be YES or NO" \
  --custom-prompt "Extract and list all diagnoses. Response must be in JSON format ('Dx')." \
  --organize-photo \
  --image D:\qwen\tt
```

### 🗂 Output

- `result.csv` – Contains extracted metadata (Name, ID, Dx).
- `log.txt` – Logs the processing steps and prompt responses.

