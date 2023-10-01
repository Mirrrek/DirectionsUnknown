#ifndef LOG_HPP
#define LOG_HPP

#include <Windows.h>
#include <string>

class Log {
public:
    Log();
    static void Info(const wchar_t* tag, const wchar_t* message, ...);
    static void Warn(const wchar_t* tag, const wchar_t* message, ...);
    static void Error(const wchar_t* tag, const wchar_t* message, ...);
    static void Critical(const wchar_t* tag, const wchar_t* message, ...);

private:
    static HANDLE hConsole;
};

#endif
