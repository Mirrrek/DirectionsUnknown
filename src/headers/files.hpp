#ifndef FILES_HPP
#define FILES_HPP

#include <stdint.h>

class Files {
public:
    static void SetBasePath();
    static bool GetSize(const wchar_t* path, uint64_t* size);
    static bool Read(const wchar_t* path, void* data, uint64_t maxSize, bool unscramble = false);
    static bool Write(const wchar_t* path, void* data, uint64_t size, bool scramble = false);
private:
    static wchar_t* basePath;
};

#endif
