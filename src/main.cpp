#include "headers/defines.hpp"
#include "headers/log.hpp"
#include "headers/files.hpp"
#include "headers/window.hpp"
#include "headers/renderer.hpp"
#include "headers/keys.hpp"
#include "headers/image.hpp"

#include "headers/emitter.hpp"

static const wchar_t* tag = L"Main";

struct RenderThreadParams {
    Window* window;
    Renderer* renderer;
};
DWORD WINAPI RenderThread(LPVOID lpParam) {
    RenderThreadParams params = *(RenderThreadParams*)lpParam;

    params.window->keys.On(L"keydown", [params] (void* data) -> bool {
        switch (((Keys::Key*)data)->code) {
        case Keys::CODE_ESC: {
            Log::Info(tag, L"Exiting...");
            params.window->Close();
        } break;
        case Keys::CODE_F11: {
            bool fullscreen = !params.window->IsFullscreen();
            Log::Info(tag, L"Setting fullscreen to %s", fullscreen ? L"true" : L"false");
            params.window->SetFullscreen(fullscreen);
            params.renderer->UpdateResolution(params.window->GetWidth(), params.window->GetHeight());
        } break;
        }
        return true;
        });

    params.window->On(L"resizeend", [params] (void*) -> bool {
        Log::Info(tag, L"Window resized, updating renderer resolution to %dx%d", params.window->GetWidth(), params.window->GetHeight());
        params.renderer->UpdateResolution(params.window->GetWidth(), params.window->GetHeight());
        return true;
        });

    Image exampleImage = Image(L"example.ppm");

    while (true) {
        params.renderer->StartRender();
        params.renderer->Clear(0x00ff00);
        params.renderer->Render();
        params.renderer->DrawImage(&exampleImage, {
                0,
                0,
                exampleImage.GetWidth(),
                exampleImage.GetHeight()
            }, {
                (uint16_t)(params.renderer->GetWidth() / 4),
                (uint16_t)(params.renderer->GetHeight() / 4),
                (uint16_t)(params.renderer->GetWidth() / 2),
                (uint16_t)(params.renderer->GetHeight() / 2)
            });
        Renderer::RenderOutput renderOutput = params.renderer->FinishRender();
        params.window->RefreshScreen(renderOutput.pixelBuffer, renderOutput.width, renderOutput.height);
    }
    return 0;
}

int WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int) {
    Log();
    Log::Info(tag, L"Initializing files...");
    Files::SetBasePath();
    Log::Info(tag, L"Initializing window...");
    Window window;
    Log::Info(tag, L"Window initialized");
    Log::Info(tag, L"Initializing renderer...");
    Renderer renderer(window.GetWidth(), window.GetHeight());
    Log::Info(tag, L"Renderer initialized");
    Log::Info(tag, L"Starting render thread...");
    RenderThreadParams renderParams = { &window, &renderer };
    CreateThread(NULL, 0, RenderThread, &renderParams, 0, NULL);
    Log::Info(tag, L"Starting window message loop...");
    window.StartMessageLoop();
}
