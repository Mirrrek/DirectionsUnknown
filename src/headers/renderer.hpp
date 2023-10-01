#ifndef RENDERER_HPP
#define RENDERER_HPP

#include "headers/image.hpp"
#include <stdint.h>

class Renderer {
public:
    struct RenderOutput {
        uint32_t* pixelBuffer;
        uint16_t width;
        uint16_t height;
    };

    struct Rectangle {
        uint16_t x;
        uint16_t y;
        uint16_t width;
        uint16_t height;
    };

    Renderer(uint16_t width, uint16_t height);
    ~Renderer();
    void UpdateResolution(uint16_t newWidth, uint16_t newHeight);
    void StartRender();
    void Clear(uint32_t color = 0x000000);
    void Render();
    void RenderImage(Image* image, Rectangle source, Rectangle destination);
    RenderOutput FinishRender();
    uint16_t GetWidth();
    uint16_t GetHeight();

private:
    uint16_t width;
    uint16_t height;
    uint32_t* pixelBuffer;
    uint32_t* devicePixelBuffer;
};

uint32_t* CopyImageToDevice(Image* image);
void FreeDeviceImage(Image* image);

#endif
