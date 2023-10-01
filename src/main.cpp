#include "headers/defines.hpp"
#include "headers/log.hpp"
#include "headers/files.hpp"
#include "headers/window.hpp"
#include "headers/math.hpp"
#include "headers/renderer.hpp"
#include "headers/world.hpp"
#include "headers/keys.hpp"
#include "headers/image.hpp"
#include <time.h>

static const wchar_t* tag = L"Main";

struct RenderThreadParams {
    Window* window;
    World* world;
    Renderer* renderer;
};
DWORD WINAPI RenderThread(LPVOID lpParam) {
    RenderThreadParams params = *(RenderThreadParams*)lpParam;

    Vector3 cameraPosition = Vector3(0.0f, 0.0f, 0.0f);
    Vector2 cameraRotation = Vector2(0.0f, 0.0f);

    Vector3* cameraPositionPtr = &cameraPosition;
    Vector2* cameraRotationPtr = &cameraRotation;

    params.window->keys.On(L"keydown", [params, &cameraPositionPtr] (void* data) -> bool {
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

    params.window->On(L"mousemove", [params, &cameraRotationPtr] (void* data) -> bool {
        Vector2i* delta = (Vector2i*)data;
        cameraRotationPtr->x = fminf(fmaxf(cameraRotationPtr->x + delta->y / 1000.0f, -1.57079632679f), 1.57079632679f);
        cameraRotationPtr->y = fmodf(cameraRotationPtr->y + delta->x / 1000.0f, 6.28318530718f);
        return true;
        });

    params.window->On(L"resizeend", [params] (void*) -> bool {
        Log::Info(tag, L"Window resized, updating renderer resolution to %dx%d", params.window->GetWidth(), params.window->GetHeight());
        params.renderer->UpdateResolution(params.window->GetWidth(), params.window->GetHeight());
        return true;
        });

    Image exampleImage = Image(L"example.ppm");

    clock_t lastTime = clock();
    clock_t currentTime = clock();
    while (true) {
        currentTime = clock();
        float deltaTime = (float)(currentTime - lastTime) / (float)CLOCKS_PER_SEC;
        lastTime = currentTime;

        float speed = 2.0f;
        float speedMultiplier = params.window->keys.LSHIFT.isDown ? 4.0f : 1.0f;

        if (params.window->keys.W.isDown) {
            cameraPositionPtr->x += deltaTime * sinf(cameraRotationPtr->y) * speed * speedMultiplier;
            cameraPositionPtr->z += deltaTime * cosf(cameraRotationPtr->y) * speed * speedMultiplier;
        }
        if (params.window->keys.S.isDown) {
            cameraPositionPtr->x -= deltaTime * sinf(cameraRotationPtr->y) * speed;
            cameraPositionPtr->z -= deltaTime * cosf(cameraRotationPtr->y) * speed;
        }
        if (params.window->keys.A.isDown) {
            cameraPositionPtr->x -= deltaTime * cosf(cameraRotationPtr->y) * speed;
            cameraPositionPtr->z += deltaTime * sinf(cameraRotationPtr->y) * speed;
        }
        if (params.window->keys.D.isDown) {
            cameraPositionPtr->x += deltaTime * cosf(cameraRotationPtr->y) * speed;
            cameraPositionPtr->z -= deltaTime * sinf(cameraRotationPtr->y) * speed;
        }
        if (params.window->keys.SPACE.isDown) {
            cameraPositionPtr->y += deltaTime;
        }
        if (params.window->keys.LCTRL.isDown) {
            cameraPositionPtr->y -= deltaTime;
        }

        params.renderer->StartRender();
        params.renderer->Render(cameraPosition, cameraRotation, params.world->GetBlockIDs());
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
    window.SetCursorLocked(true);
    Log::Info(tag, L"Window initialized");
    Log::Info(tag, L"Initializing world...");
    World world;
    Log::Info(tag, L"World initialized");
    Log::Info(tag, L"Initializing renderer...");
    Renderer renderer(window.GetWidth(), window.GetHeight(), world.GetBlockDescriptors());
    Log::Info(tag, L"Renderer initialized");
    Log::Info(tag, L"Starting render thread...");
    RenderThreadParams renderParams = { &window, &world, &renderer };
    CreateThread(NULL, 0, RenderThread, &renderParams, 0, NULL);
    Log::Info(tag, L"Starting window message loop...");
    window.StartMessageLoop();
}
