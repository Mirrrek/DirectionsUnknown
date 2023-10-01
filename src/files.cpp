#include "headers/files.hpp"
#include "headers/log.hpp"
#include <Windows.h>

#define MAX_PATH_LENGTH 1024

static const wchar_t* tag = L"Files";

wchar_t* Files::basePath = nullptr;

void Files::SetBasePath() {
    if (Files::basePath != nullptr) {
        free(Files::basePath);
    }
    Files::basePath = (wchar_t*)malloc(MAX_PATH_LENGTH * sizeof(wchar_t));
    uint16_t pathLength = (uint16_t)GetModuleFileNameW(NULL, Files::basePath, MAX_PATH_LENGTH);
    if (pathLength == 0 || pathLength >= MAX_PATH_LENGTH - 1) {
        Log::Critical(tag, L"Base path is greater than %d characters", MAX_PATH_LENGTH);
    }
    pathLength--;
    while (Files::basePath[pathLength] != L'\\') {
        pathLength--;
    }
    Files::basePath[pathLength] = L'\0';
}

bool Files::GetSize(const wchar_t* path, uint64_t* size) {
    if (Files::basePath == nullptr) {
        Log::Critical(tag, L"Base path is not set");
        return false;
    }
    if (path[0] == L'\\' || path[0] == L'/') {
        path++;
    }
    wchar_t fullPath[MAX_PATH_LENGTH];
    wcscpy_s(fullPath, MAX_PATH_LENGTH, Files::basePath);
    wcscat_s(fullPath, MAX_PATH_LENGTH, L"\\");
    if (wcscat_s(fullPath, MAX_PATH_LENGTH, path) != 0) {
        Log::Error(tag, L"Path \"%s\" is too long", path);
        return false;
    }
    HANDLE file = CreateFileW(fullPath, GENERIC_READ, 0, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
    if (file == INVALID_HANDLE_VALUE) {
        Log::Error(tag, L"Failed to open file \"%s\"", path);
        CloseHandle(file);
        return false;
    }
    *size = GetFileSize(file, NULL);
    CloseHandle(file);
    return true;
}

bool Files::Read(const wchar_t* path, void* data, uint64_t maxSize, bool unscramble) {
    if (Files::basePath == nullptr) {
        Log::Critical(tag, L"Base path is not set");
        return false;
    }
    if (path[0] == L'\\' || path[0] == L'/') {
        path++;
    }
    wchar_t fullPath[MAX_PATH_LENGTH];
    wcscpy_s(fullPath, MAX_PATH_LENGTH, Files::basePath);
    wcscat_s(fullPath, MAX_PATH_LENGTH, L"\\");
    if (wcscat_s(fullPath, MAX_PATH_LENGTH, path) != 0) {
        Log::Error(tag, L"Path \"%s\" is too long", path);
        return false;
    }
    HANDLE file = CreateFileW(fullPath, GENERIC_READ, 0, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
    if (file == INVALID_HANDLE_VALUE) {
        Log::Error(tag, L"Failed to open file \"%s\"", path);
        CloseHandle(file);
        return false;
    }
    uint64_t fileSize = GetFileSize(file, NULL);
    if (fileSize > maxSize) {
        Log::Error(tag, L"File \"%s\" is too large", path);
        CloseHandle(file);
        return false;
    }
    DWORD read;
    if (!ReadFile(file, data, (DWORD)fileSize, &read, NULL)) {
        Log::Error(tag, L"Failed to read file \"%s\"", path);
        CloseHandle(file);
        return false;
    }
    if (read != fileSize) {
        Log::Error(tag, L"Failed to read file \"%s\" (expected %llu, read %llu)", path, fileSize, read);
        CloseHandle(file);
        return false;
    }
    if (unscramble) {
        uint64_t i;
        for (i = 0; i < fileSize - 5; i += 5) {
            ((uint8_t*)data)[i] ^= 0xe5;
            ((uint8_t*)data)[i + 1] ^= 0x19;
            ((uint8_t*)data)[i + 2] ^= 0xa5;
            ((uint8_t*)data)[i + 3] ^= 0x3a;
            ((uint8_t*)data)[i + 4] ^= 0x78;
        }
        for (; i < fileSize; i++) {
            ((uint8_t*)data)[i] ^= 0xe5;
        }
    }
    CloseHandle(file);
    return true;
}

bool Files::Write(const wchar_t* path, void* data, uint64_t size, bool scramble) {
    if (Files::basePath == nullptr) {
        Log::Critical(tag, L"Base path is not set");
        return false;
    }
    if (path[0] == L'\\' || path[0] == L'/') {
        path++;
    }
    wchar_t fullPath[MAX_PATH_LENGTH];
    wcscpy_s(fullPath, MAX_PATH_LENGTH, Files::basePath);
    wcscat_s(fullPath, MAX_PATH_LENGTH, L"\\");
    if (wcscat_s(fullPath, MAX_PATH_LENGTH, path) != 0) {
        Log::Error(tag, L"Path \"%s\" is too long", path);
        return false;
    }
    HANDLE file = CreateFileW(fullPath, GENERIC_WRITE, 0, 0, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, 0);
    if (file == INVALID_HANDLE_VALUE) {
        Log::Error(tag, L"Failed to open file \"%s\"", path);
        CloseHandle(file);
        return false;
    }
    void* data2 = malloc(size);
    memcpy(data2, data, size);
    if (scramble) {
        uint64_t i;
        for (i = 0; i < size - 5; i += 5) {
            ((uint8_t*)data2)[i] ^= 0xe5;
            ((uint8_t*)data2)[i + 1] ^= 0x19;
            ((uint8_t*)data2)[i + 2] ^= 0xa5;
            ((uint8_t*)data2)[i + 3] ^= 0x3a;
            ((uint8_t*)data2)[i + 4] ^= 0x78;
        }
        for (; i < size; i++) {
            ((uint8_t*)data2)[i] ^= 0xe5;
        }
    }
    DWORD written;
    if (!WriteFile(file, data2, (DWORD)size, &written, NULL)) {
        Log::Error(tag, L"Failed to write to file \"%s\"", path);
        CloseHandle(file);
        free(data2);
        return false;
    }
    if (written != size) {
        Log::Error(tag, L"Failed to write to file \"%s\" (expected %llu, written %llu)", path, size, written);
        CloseHandle(file);
        free(data2);
        return false;
    }
    CloseHandle(file);
    free(data2);
    return true;
}
