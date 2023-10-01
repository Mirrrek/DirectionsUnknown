#include "headers/defines.hpp"
#include "headers/log.hpp"
#include "headers/window.hpp"

#include <Windows.h>

static const wchar_t* tag = L"Window";

LRESULT CALLBACK WndProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

Window::Window(bool fullscreen) {
    this->lockedCursor = false;
    this->width = 0;
    this->height = 0;

    WNDCLASSEXW windowClass = { 0 };
    windowClass.cbSize = sizeof(WNDCLASSEXW);
    windowClass.style = CS_HREDRAW | CS_VREDRAW;
    windowClass.lpfnWndProc = WndProc;
    windowClass.hInstance = GetModuleHandleW(NULL);
    windowClass.lpszClassName = GAME_TITLE;

    if (!RegisterClassExW(&windowClass)) {
        Log::Critical(tag, L"Failed to register window class");
        return;
    }

    this->hWnd = CreateWindowExW(0, GAME_TITLE, GAME_TITLE, 0, 0, 0, 0, 0, NULL, NULL, GetModuleHandleW(NULL), NULL);

    if (!this->hWnd) {
        Log::Critical(tag, L"Failed to create window");
        return;
    }

    RAWINPUTDEVICE keyboard = { 0 };
    keyboard.usUsagePage = 0x01;
    keyboard.usUsage = 0x06;
    keyboard.dwFlags = 0;
    keyboard.hwndTarget = this->hWnd;

    RAWINPUTDEVICE mouse = { 0 };
    mouse.usUsagePage = 0x01;
    mouse.usUsage = 0x02;
    mouse.dwFlags = 0;
    mouse.hwndTarget = this->hWnd;

    RegisterRawInputDevices(&(keyboard), 1, sizeof(keyboard));
    RegisterRawInputDevices(&(mouse), 1, sizeof(mouse));

    SetWindowLongPtrW(this->hWnd, GWLP_USERDATA, (LONG_PTR)this);

    this->SetFullscreen(fullscreen);

    this->bitmapInfo = { 0 };
    this->bitmapInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    this->bitmapInfo.bmiHeader.biWidth = 0;
    this->bitmapInfo.bmiHeader.biHeight = 0;
    this->bitmapInfo.bmiHeader.biPlanes = 1;
    this->bitmapInfo.bmiHeader.biBitCount = 32;
    this->bitmapInfo.bmiHeader.biCompression = BI_RGB;

    this->hdc = GetDC(this->hWnd);
}

void Window::StartMessageLoop() {
    MSG msg = {};
    while (1) {
        if (!GetMessageW(&msg, NULL, 0, 0)) {
            break;
        }

        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
}

void Window::SetFullscreen(bool fullscreen) {
    uint16_t windowX = 0;
    uint16_t windowY = 0;
    uint16_t windowWidth = (uint16_t)GetSystemMetrics(SM_CXSCREEN);
    uint16_t windowHeight = (uint16_t)GetSystemMetrics(SM_CYSCREEN);

    if (!fullscreen) {
        RECT windowRect = { 0, 0, windowWidth, windowHeight };
        AdjustWindowRect(&windowRect, WS_OVERLAPPEDWINDOW, false);
        windowX = max(0, (windowWidth - (uint16_t)(windowRect.right - windowRect.left)) / 2);
        windowY = max(0, (windowHeight - (uint16_t)(windowRect.bottom - windowRect.top)) / 2);
        windowWidth = (uint16_t)(windowRect.right - windowRect.left);
        windowHeight = (uint16_t)(windowRect.bottom - windowRect.top);
    }

    SetWindowLongW(this->hWnd, GWL_STYLE, fullscreen ? WS_OVERLAPPED : WS_OVERLAPPEDWINDOW);
    SetWindowPos(this->hWnd, HWND_NOTOPMOST, windowX, windowY, windowWidth, windowHeight, SWP_FRAMECHANGED | SWP_NOACTIVATE);
    ShowWindow(this->hWnd, SW_MAXIMIZE);

    this->width = windowWidth;
    this->height = windowHeight;
}

bool Window::IsFullscreen() {
    return (GetWindowLongW(this->hWnd, GWL_STYLE) & WS_OVERLAPPEDWINDOW) == WS_OVERLAPPED;
}

void Window::RefreshScreen(uint32_t* pixelBuffer, uint16_t sourceWidth, uint16_t sourceHeight) {
    this->bitmapInfo.bmiHeader.biWidth = sourceWidth;
    this->bitmapInfo.bmiHeader.biHeight = -sourceHeight;
    StretchDIBits(this->hdc, 0, 0, this->width, this->height, 0, 0, sourceWidth, sourceHeight, pixelBuffer, &this->bitmapInfo, DIB_RGB_COLORS, SRCCOPY);
}

void Window::CenterCursor() {
    RECT clientRect;
    GetClientRect(this->hWnd, &clientRect);
    POINT centerPoint = { clientRect.right / 2, clientRect.bottom / 2 };
    ClientToScreen(this->hWnd, &centerPoint);
    SetCursorPos(centerPoint.x, centerPoint.y);
}

void Window::SetCursorLocked(bool locked) {
    this->lockedCursor = locked;
    ShowCursor(this->lockedCursor);
    if (this->lockedCursor) {
        this->CenterCursor();
    }
}

bool Window::IsCursorLocked() {
    return this->lockedCursor;
}

uint16_t Window::GetWidth() {
    return this->width;
}

uint16_t Window::GetHeight() {
    return this->height;
}

void Window::Close() {
    PostMessageW(this->hWnd, WM_CLOSE, 0, 0);
}

void Window::OnPaint() {
    this->Emit(L"paint", 0);
}

void Window::OnResize(uint16_t newWidth, uint16_t newHeight) {
    this->width = newWidth;
    this->height = newHeight;
    this->Emit(L"resize", 0);
}

void Window::OnResizeEnd() {
    this->Emit(L"resizeend", 0);
}

void Window::OnMouseMove(int16_t x, int16_t y) {
    MouseMoveEventData event = { x, y };
    this->Emit(L"mousemove", &event);
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
    Window* window = (Window*)GetWindowLongPtrW(hWnd, GWLP_USERDATA);
    switch (message) {
    case WM_PAINT:
    {
        window->OnPaint();
    }
    break;
    case WM_SIZE:
    {
        RECT clientRect;
        GetClientRect(hWnd, &clientRect);
        window->OnResize((uint16_t)(clientRect.right - clientRect.left), (uint16_t)(clientRect.bottom - clientRect.top));
    }
    break;
    case WM_EXITSIZEMOVE:
    {
        RECT clientRect;
        GetClientRect(hWnd, &clientRect);
        window->OnResizeEnd();
    }
    break;
    case WM_INPUT:
    {
        UINT dataLength;
        GetRawInputData((HRAWINPUT)lParam, RID_INPUT, NULL, &dataLength, sizeof(RAWINPUTHEADER));
        uint8_t* data = new uint8_t[dataLength];
        GetRawInputData((HRAWINPUT)lParam, RID_INPUT, data, &dataLength, sizeof(RAWINPUTHEADER));

        RAWINPUT* raw = (RAWINPUT*)data;

        if (raw->header.dwType == RIM_TYPEKEYBOARD) {
            uint16_t makeCode = raw->data.keyboard.MakeCode;
            if (raw->data.keyboard.Flags & RI_KEY_E0) {
                makeCode |= 0xe000;
            }
            if (raw->data.keyboard.Flags & RI_KEY_E1) {
                makeCode |= 0xe100;
            }

            if (raw->data.keyboard.Flags & RI_KEY_BREAK) {
                window->keys.KeyUp(makeCode);
            }
            else {
                window->keys.KeyDown(makeCode);
            }
        }
        else if (raw->header.dwType == RIM_TYPEMOUSE) {
            if (raw->data.mouse.usFlags == MOUSE_MOVE_RELATIVE) {
                window->OnMouseMove((int16_t)raw->data.mouse.lLastX, (int16_t)raw->data.mouse.lLastY);
                if (window->IsCursorLocked()) {
                    window->CenterCursor();
                }
            }
        }
        return 0;
    }
    break;
    case WM_SETCURSOR:
    {
        SetCursor(LoadCursorA(NULL, (LPSTR)IDC_ARROW));
        return 1;
    }
    break;
    case WM_DESTROY:
    {
        PostQuitMessage(0);
    }
    break;
    default:
        return DefWindowProcW(hWnd, message, wParam, lParam);
    }

    return 0;
}
