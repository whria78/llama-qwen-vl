#include <windows.h>
#include <shlobj.h> // 폴더 선택 다이얼로그
#include <string>
#include <fstream>
#include <filesystem>
#include <shellapi.h>
#include <sysinfoapi.h>
#include <iostream>
#include <sstream>

#define ID_BUTTON_BROWSE  1
#define ID_BUTTON_RUN     2
#define ID_BUTTON_RESET   3
#define ID_EDIT_COMMAND   4
#define ID_STATIC_FOLDER  5

HWND hEditCommand, hStaticFolder;
std::wstring folderPath = L"";
std::wstring defaultCommand = L"vl -m ./gguf/Qwen2-VL-72B-Instruct-Q4_K_M.gguf --mmproj ./gguf/Qwen2-VL-72B-Instruct-vision-encoder.gguf --temp 0.1 -p \"Extract the patient's name and registration number. Response must be in JSON format ('Name','ID').\" -t 16 --organize-photo --image ";

void LoadDefaultCommand() {
    std::wifstream file(L"VLRunner.txt");
    if (file) {
        std::wstring line;
        std::getline(file, line);
        defaultCommand = line;
    }
}
bool IsVersionGreaterOrEqual(const std::wstring& currentVersion, const std::wstring& requiredVersion) {
    std::wstringstream curStream(currentVersion);
    std::wstringstream reqStream(requiredVersion);
    int curMajor, curMinor, curBuild, curRev;
    int reqMajor, reqMinor, reqBuild, reqRev;

    wchar_t dot;
    curStream >> curMajor >> dot >> curMinor >> dot >> curBuild >> dot >> curRev;
    reqStream >> reqMajor >> dot >> reqMinor >> dot >> reqBuild >> dot >> reqRev;

    if (curMajor > reqMajor) return true;
    if (curMajor < reqMajor) return false;
    if (curMinor > reqMinor) return true;
    if (curMinor < reqMinor) return false;
    if (curBuild > reqBuild) return true;
    if (curBuild < reqBuild) return false;
    if (curRev >= reqRev) return true;
    return false;
}

void TrimString(std::wstring& str) {
    while (!str.empty() && (str.back() == L'\n' || str.back() == L'\r' || str.back() == L' ' || str.back() == L'\t')) {
        str.pop_back();
    }
    while (!str.empty() && (str.front() == L' ' || str.front() == L'\t' || str.front() == L'\n' || str.front() == L'\r' || !iswdigit(str.front()))) {
        str.erase(str.begin());
    }
}

void CheckRedistributableVersion(HWND hwnd) {
    std::wstring command = L"reg query \"HKLM\\SOFTWARE\\Microsoft\\VisualStudio\\14.0\\VC\\Runtimes\\x64\" /v Version";
    FILE* pipe = _wpopen(command.c_str(), L"r");
    if (!pipe) return;

    wchar_t buffer[128];
    std::wstring result = L"";
    while (fgetws(buffer, 128, pipe)) {
        result += buffer;
    }
    _pclose(pipe);

    size_t pos = result.find(L"REG_SZ");
    if (pos != std::wstring::npos) {
        std::wstring currentVersion = result.substr(pos + 6);
        TrimString(currentVersion);

        std::wstring requiredVersion = L"14.42.34400.0";
        if (!IsVersionGreaterOrEqual(currentVersion, requiredVersion)) {
            int res = MessageBoxW(hwnd, L"Redistributable packages for Visual Studio 2015, 2017, 2019, and 2022 are not up-to-date. Please visit https://whria78.github.io/medicalphoto/warning.", L"Version Warning", MB_OK | MB_ICONWARNING);
            if (res == IDOK) ShellExecuteW(NULL, L"open", L"https://whria78.github.io/medicalphoto/warning", NULL, NULL, SW_SHOWNORMAL);
        }
    }
}



void CheckGGUFDirectory(HWND hwnd) {
    std::filesystem::path ggufPath = L"./gguf";
    bool empty = true;
    if (std::filesystem::exists(ggufPath) && std::filesystem::is_directory(ggufPath)) {
        for (const auto& entry : std::filesystem::directory_iterator(ggufPath)) {
            if (entry.path().extension() == L".gguf") {
                empty = false;
                break;
            }
        }
    }
    if (empty) {
        int result = MessageBoxW(hwnd, L"Please visit https://whria78.github.io/medicalphoto/warning and follow the instructions to download the model file.", L"Missing Model", MB_OKCANCEL | MB_ICONWARNING);
        if (result == IDOK) ShellExecuteW(NULL, L"open", L"https://github.com/whria78/medicalphoto/warning", NULL, NULL, SW_SHOWNORMAL);
        
    }
}

void CheckMemory(HWND hwnd) {
    MEMORYSTATUSEX memStatus;
    memStatus.dwLength = sizeof(memStatus);
    GlobalMemoryStatusEx(&memStatus);

    if (memStatus.ullTotalPhys / (1024 * 1024 * 1024) < 60) {
        MessageBoxW(hwnd, L"At least 64GB of RAM is recommended for optimal performance.", L"Memory Warning", MB_OK | MB_ICONWARNING);
    }
}

void CheckUnicodeSupport(HWND hwnd) {
    UINT codePage = GetACP();
    if (codePage != 65001) { // 65001은 UTF-8 코드 페이지
        int result=MessageBoxW(hwnd, L"Command prompt does not support Unicode. Please visit https://whria78.github.io/medicalphoto/warning", L"Unicode Warning", MB_OK | MB_ICONWARNING);
        if (result == IDOK) ShellExecuteW(NULL, L"open", L"https://whria78.github.io/medicalphoto/warning", NULL, NULL, SW_SHOWNORMAL);
    }
}

void UpdateCommandBox() {
    std::wstring command = defaultCommand + folderPath;
    SetWindowTextW(hEditCommand, command.c_str());
}

void SelectFolder(HWND hwnd) {
    BROWSEINFO bi = { 0 };
    bi.lpszTitle = L"Select Folder";
    bi.ulFlags = BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE;

    LPITEMIDLIST pidl = SHBrowseForFolder(&bi);
    if (pidl) {
        wchar_t path[MAX_PATH];
        if (SHGetPathFromIDList(pidl, path)) {
            folderPath = path;
            SetWindowTextW(hStaticFolder, folderPath.c_str());
            UpdateCommandBox();
        }
        CoTaskMemFree(pidl);
    }
}

void RunCommand(HWND hwnd) {
    int length = GetWindowTextLengthW(hEditCommand) + 1;
    wchar_t* buffer = new wchar_t[length];
    GetWindowTextW(hEditCommand, buffer, length);
    std::wstring command = buffer;
    delete[] buffer;

    if (folderPath.empty()) {
        MessageBoxW(hwnd, L"Please select a folder first.", L"Warning", MB_OK | MB_ICONWARNING);
        return;
    }

    std::wstring fullCommand = L"cmd.exe /C \"" + command + L" & echo. & echo Press SPACE to continue... & pause >nul\"";
    ShellExecuteW(NULL, L"open", L"cmd.exe", fullCommand.c_str(), NULL, SW_SHOW);
}

void ResetUI() {
    folderPath = L"";
    SetWindowTextW(hStaticFolder, L"(No folder selected)");
    SetWindowTextW(hEditCommand, defaultCommand.c_str());
}

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
    case WM_CREATE:
        LoadDefaultCommand();
        CheckGGUFDirectory(hwnd);
        CheckMemory(hwnd);
        CheckUnicodeSupport(hwnd);
        CheckRedistributableVersion(hwnd);
        CreateWindowW(L"BUTTON", L"Reset", WS_VISIBLE | WS_CHILD, 5, 5, 390, 30, hwnd, (HMENU)ID_BUTTON_RESET, NULL, NULL);
        CreateWindowW(L"BUTTON", L"Browse", WS_VISIBLE | WS_CHILD, 5, 40, 70, 25, hwnd, (HMENU)ID_BUTTON_BROWSE, NULL, NULL);
        hStaticFolder = CreateWindowW(L"STATIC", L"(No folder selected)", WS_VISIBLE | WS_CHILD, 80, 40, 300, 25, hwnd, (HMENU)ID_STATIC_FOLDER, NULL, NULL);
        hEditCommand = CreateWindowW(L"EDIT", defaultCommand.c_str(), WS_VISIBLE | WS_CHILD | WS_BORDER | ES_MULTILINE | ES_AUTOVSCROLL | ES_WANTRETURN,
            10, 70, 380, 170, hwnd, (HMENU)ID_EDIT_COMMAND, NULL, NULL);
        CreateWindowW(L"BUTTON", L"Run", WS_VISIBLE | WS_CHILD, 5, 245, 390, 30, hwnd, (HMENU)ID_BUTTON_RUN, NULL, NULL);
        return 0;

    case WM_COMMAND:
        switch (LOWORD(wParam)) {
        case ID_BUTTON_BROWSE:
            SelectFolder(hwnd);
            break;
        case ID_BUTTON_RUN:
            RunCommand(hwnd);
            break;
        case ID_BUTTON_RESET:
            ResetUI();
            break;
        }
        return 0;

    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    WNDCLASSW wc = { 0 };
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = L"Commander";
    RegisterClassW(&wc);

    HWND hwnd = CreateWindowW(wc.lpszClassName, L"Commander", WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        CW_USEDEFAULT, CW_USEDEFAULT, 420, 320, NULL, NULL, hInstance, NULL);

    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    return (int)msg.wParam;
}
