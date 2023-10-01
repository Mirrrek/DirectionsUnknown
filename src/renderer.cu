#include "headers/renderer.hpp"
#include "headers/locker.hpp"
#include "headers/log.hpp"

#define WIDTH_SCROLL_SPEED 0.0016545f
#define HEIGHT_SCROLL_SPEED 0.0003549f

__global__ void ClearKernel(uint32_t* devicePixelBuffer, uint16_t width, uint16_t height, uint32_t color) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;

    for (uint32_t i = index; i < (uint32_t)width * (uint32_t)height; i += stride) {
        devicePixelBuffer[i] = color;
    }
}

__global__ void RenderKernel(uint32_t* devicePixelBuffer, uint16_t width, uint16_t height, uint64_t time) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;

    for (uint32_t i = index; i < (uint32_t)width * (uint32_t)height; i += stride) {
        uint16_t x = (i % width);
        uint16_t y = i / width;

        uint8_t r = (0.5f * cosf((float)time * (float)WIDTH_SCROLL_SPEED + (float)x / (float)width * 3.1415f * 2.0f) + 0.5f) * (float)x / (float)width * 0xff;
        uint8_t g = 0x00;
        uint8_t b = (0.5f * cosf((float)time * (float)HEIGHT_SCROLL_SPEED + (float)y / (float)height * 3.1415f * 2.0f) + 0.5f) * (float)y / (float)height * 0xff;

        devicePixelBuffer[i] = (r << 16) | (g << 8) | b;
    }
}

__global__ void DrawImageKernel(uint32_t* devicePixelBuffer, uint16_t screenWidth, uint16_t screenHeight, uint32_t* deviceImagePixelBuffer, uint16_t imageWidth, uint16_t imageHeight, Renderer::Rectangle source, Renderer::Rectangle destination) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;

    for (uint32_t i = index; i < (uint32_t)destination.width * (uint32_t)destination.height; i += stride) {
        uint16_t x = (i % destination.width);
        uint16_t y = i / destination.width;

        uint16_t imageX = (float)x / (float)destination.width * (float)source.width + source.x;
        uint16_t imageY = (float)y / (float)destination.height * (float)source.height + source.y;

        if (imageX >= imageWidth || imageY >= imageHeight) {
            continue;
        }

        devicePixelBuffer[((uint32_t)screenHeight - (uint32_t)y - (uint32_t)destination.y) * (uint32_t)screenWidth + ((uint32_t)x + (uint32_t)destination.x)] = deviceImagePixelBuffer[(uint32_t)imageY * (uint32_t)imageWidth + (uint32_t)imageX];
    }
}


Renderer::Renderer(uint16_t _width, uint16_t _height) {
    this->width = _width;
    this->height = _height;
    this->pixelBuffer = (uint32_t*)malloc((uint32_t)this->width * (uint32_t)this->height * sizeof(uint32_t));
    cudaMalloc(&this->devicePixelBuffer, (uint32_t)this->width * (uint32_t)this->height * sizeof(uint32_t));
}

Renderer::~Renderer() {
    free(this->pixelBuffer);
    cudaFree(this->devicePixelBuffer);
}

void Renderer::UpdateResolution(uint16_t newWidth, uint16_t newHeight) {
    Locker::rendererBuffers.Lock();

    this->width = newWidth;
    this->height = newHeight;
    free(this->pixelBuffer);
    this->pixelBuffer = (uint32_t*)malloc((uint32_t)this->width * (uint32_t)this->height * sizeof(uint32_t));
    cudaFree(this->devicePixelBuffer);
    cudaMalloc(&this->devicePixelBuffer, (uint32_t)this->width * (uint32_t)this->height * sizeof(uint32_t));

    Locker::rendererBuffers.Unlock();
}

void Renderer::StartRender() {
    Locker::rendererBuffers.Lock();
}

void Renderer::Clear(uint32_t color) {
    uint16_t blockSize = 1024;
    uint32_t blockCount = ((uint32_t)this->width * (uint32_t)this->height + blockSize - 1) / blockSize;

    ClearKernel << <blockCount, blockSize >> > (this->devicePixelBuffer, this->width, this->height, color);
}

void Renderer::Render() {
    uint16_t blockSize = 1024;
    uint32_t blockCount = ((uint32_t)this->width * (uint32_t)this->height + blockSize - 1) / blockSize;

    RenderKernel << <blockCount, blockSize >> > (this->devicePixelBuffer, this->width, this->height, clock());
}

void Renderer::DrawImage(Image* image, Renderer::Rectangle source, Renderer::Rectangle destination) {
    if (destination.x >= this->width || destination.y >= this->height || destination.x + destination.width <= 0 || destination.y + destination.height <= 0) {
        return;
    }
    if (destination.x + destination.width > this->width) {
        destination.width = this->width - destination.x;
    }
    if (destination.y + destination.height > this->height) {
        destination.height = this->height - destination.y;
    }

    uint16_t blockSize = 1024;
    uint32_t blockCount = ((uint32_t)destination.width * (uint32_t)destination.height + blockSize - 1) / blockSize;

    DrawImageKernel << <blockCount, blockSize >> > (this->devicePixelBuffer, this->width, this->height, image->GetDevicePixelBuffer(), image->GetWidth(), image->GetHeight(), source, destination);
}

Renderer::RenderOutput Renderer::FinishRender() {
    cudaDeviceSynchronize();

    cudaMemcpy(this->pixelBuffer, this->devicePixelBuffer, (uint32_t)this->width * (uint32_t)this->height * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    Locker::rendererBuffers.Unlock();

    return {
        this->pixelBuffer,
        this->width,
        this->height,
    };
}

uint16_t Renderer::GetWidth() {
    return this->width;
}

uint16_t Renderer::GetHeight() {
    return this->height;
}

uint32_t* CopyImageToDevice(Image* image) {
    uint32_t* deviceImagePixelBuffer;

    cudaMalloc(&deviceImagePixelBuffer, (uint32_t)image->GetWidth() * (uint32_t)image->GetHeight() * sizeof(uint32_t));
    cudaMemcpy(deviceImagePixelBuffer, image->GetPixelBuffer(), (uint32_t)image->GetWidth() * (uint32_t)image->GetHeight() * sizeof(uint32_t), cudaMemcpyHostToDevice);

    return deviceImagePixelBuffer;
}

void FreeDeviceImage(Image* image) {
    cudaFree(image->GetDevicePixelBuffer());
}
