#ifndef WINDOW_HPP
#define WINDOW_HPP

#include "headers/emitter.hpp"
#include "headers/keys.hpp"
#include "headers/image.hpp"
#include <Windows.h>
#include <stdint.h>

class Window : public Emitter {
public:
    Window(bool fullscreen = false);
    void StartMessageLoop();
    void SetFullscreen(bool fullscreen);
    bool IsFullscreen();
    void RefreshScreen(uint32_t* pixelBuffer, uint16_t sourceWidth, uint16_t sourceHeight);
    void CenterCursor();
    void SetCursorLocked(bool locked);
    bool IsCursorLocked();
    uint16_t GetWidth();
    uint16_t GetHeight();
    void Close();
    void OnPaint();
    void OnResize(uint16_t newWidth, uint16_t newHeight);
    void OnResizeEnd();
    void OnMouseMove(int16_t x, int16_t y);
    struct MouseMoveEventData {
        int16_t x;
        int16_t y;
    };
    Keys keys;

private:
    HWND hWnd;
    BITMAPINFO bitmapInfo;
    HDC hdc;
    bool lockedCursor;
    uint16_t width;
    uint16_t height;
};

#endif
