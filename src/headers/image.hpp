#ifndef IMAGE_HPP
#define IMAGE_HPP

#include <stdint.h>

class Image {
public:
    Image(const wchar_t* path);
    ~Image();

    uint16_t GetWidth();
    uint16_t GetHeight();
    uint32_t* GetPixelBuffer();
    uint32_t* GetDevicePixelBuffer();
private:
    uint16_t width;
    uint16_t height;
    uint8_t* file;
    uint32_t* pixelBuffer;
    uint32_t* devicePixelBuffer;
};

#endif
