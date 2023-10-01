#include "headers/defines.hpp"
#include "headers/log.hpp"

#include <Windows.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define LOG_BUFFER_SIZE 1024

HANDLE Log::hConsole = NULL;

Log::Log() {
#ifdef _DEBUG
    AllocConsole();
    SetConsoleTitleW(GAME_TITLE L" - Debug Console");
    Log::hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
#endif
}

void Log::Info(const wchar_t* tag, const wchar_t* message, ...) {
#ifdef _DEBUG
    va_list args;
    va_start(args, message);
    wchar_t buffer1[LOG_BUFFER_SIZE];
    vswprintf_s(buffer1, LOG_BUFFER_SIZE, message, args);
    va_end(args);
    wchar_t buffer2[LOG_BUFFER_SIZE];
    swprintf_s(buffer2, LOG_BUFFER_SIZE, L"[%d] [%s] [INFO] %s\n", clock(), tag, buffer1);
    SetConsoleTextAttribute(Log::hConsole, FOREGROUND_INTENSITY | FOREGROUND_BLUE);
    WriteConsoleW(Log::hConsole, buffer2, (DWORD)wcslen(buffer2), NULL, NULL);
#endif
}

void Log::Warn(const wchar_t* tag, const wchar_t* message, ...) {
#ifdef _DEBUG
    va_list args;
    va_start(args, message);
    wchar_t buffer1[LOG_BUFFER_SIZE];
    vswprintf_s(buffer1, LOG_BUFFER_SIZE, message, args);
    va_end(args);
    wchar_t buffer2[LOG_BUFFER_SIZE];
    swprintf_s(buffer2, LOG_BUFFER_SIZE, L"[%d] [%s] [WARN] %s\n", clock(), tag, buffer1);
    SetConsoleTextAttribute(Log::hConsole, FOREGROUND_INTENSITY | FOREGROUND_RED | FOREGROUND_GREEN);
    WriteConsoleW(Log::hConsole, buffer2, (DWORD)wcslen(buffer2), NULL, NULL);
#endif
}

void Log::Error(const wchar_t* tag, const wchar_t* message, ...) {
#ifdef _DEBUG
    va_list args;
    va_start(args, message);
    wchar_t buffer1[LOG_BUFFER_SIZE];
    vswprintf_s(buffer1, LOG_BUFFER_SIZE, message, args);
    va_end(args);
    wchar_t buffer2[LOG_BUFFER_SIZE];
    swprintf_s(buffer2, LOG_BUFFER_SIZE, L"[%d] [%s] [ERROR] %s\n", clock(), tag, buffer1);
    SetConsoleTextAttribute(Log::hConsole, FOREGROUND_INTENSITY | FOREGROUND_RED);
    WriteConsoleW(Log::hConsole, buffer2, (DWORD)wcslen(buffer2), NULL, NULL);
#endif
}

void Log::Critical(const wchar_t* tag, const wchar_t* message, ...) {
    va_list args;
    va_start(args, message);
    wchar_t buffer1[LOG_BUFFER_SIZE];
    vswprintf_s(buffer1, LOG_BUFFER_SIZE, message, args);
    va_end(args);
#ifdef _DEBUG
    wchar_t buffer2[LOG_BUFFER_SIZE];
    swprintf_s(buffer2, LOG_BUFFER_SIZE, L"[%d] [%s] [CRITICAL] %s\n", clock(), tag, buffer1);
    SetConsoleTextAttribute(Log::hConsole, FOREGROUND_INTENSITY | FOREGROUND_RED | FOREGROUND_BLUE);
    WriteConsoleW(Log::hConsole, buffer2, (DWORD)wcslen(buffer2), NULL, NULL);
#endif
    MessageBoxW(NULL, buffer1, L"Critical Error", MB_OK);
    ExitProcess(1);
}
