#include "headers/renderer.hpp"
#include "headers/locker.hpp"
#include "headers/world.hpp"
#include "headers/math.hpp"
#include "headers/log.hpp"

__global__ void ClearKernel(uint32_t* devicePixelBuffer, uint16_t width, uint16_t height, uint32_t color) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;

    for (uint32_t i = index; i < (uint32_t)width * (uint32_t)height; i += stride) {
        devicePixelBuffer[i] = color;
    }
}

__global__ void RenderKernel(uint32_t* devicePixelBuffer, uint16_t width, uint16_t height, Vector3 cameraPosition, Vector2 cameraRotation, uint8_t* blockIDs, BlockDescriptor* blockDescriptors) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;

    for (uint32_t i = index; i < (uint32_t)width * (uint32_t)height; i += stride) {
        uint16_t pixelX = (i % width);
        uint16_t pixelY = i / width;

        Vector3 rayOrigin = cameraPosition;
        Vector3 rayDirection = Vector3(
            ((float)pixelX - (float)width * 0.5f) / ((float)width * 0.5f),
            ((float)pixelY - (float)height * 0.5f) / ((float)width * 0.5f),
            1.0f
        );

        rayDirection = Vector3(
            rayDirection.x,
            rayDirection.y * cosf(cameraRotation.x) - rayDirection.z * sinf(cameraRotation.x),
            rayDirection.y * sinf(cameraRotation.x) + rayDirection.z * cosf(cameraRotation.x)
        );

        rayDirection = Vector3(
            rayDirection.x * cosf(cameraRotation.y) + rayDirection.z * sinf(cameraRotation.y),
            rayDirection.y,
            rayDirection.z * cosf(cameraRotation.y) - rayDirection.x * sinf(cameraRotation.y)
        );

        rayDirection = rayDirection.Normalized();

        Vector3i currentVoxel = Vector3i(
            (int32_t)floorf(rayOrigin.x),
            (int32_t)floorf(rayOrigin.y),
            (int32_t)floorf(rayOrigin.z)
        );

        Vector3i step = Vector3i(
            rayDirection.x < 0.0f ? -1 : 1,
            rayDirection.y < 0.0f ? -1 : 1,
            rayDirection.z < 0.0f ? -1 : 1
        );
        Vector3 tMax = Vector3(
            rayDirection.x == 0.0f ? INFINITY : ((float)currentVoxel.x + (step.x > 0 ? 1.0f : 0.0f) - rayOrigin.x) / rayDirection.x,
            rayDirection.y == 0.0f ? INFINITY : ((float)currentVoxel.y + (step.y > 0 ? 1.0f : 0.0f) - rayOrigin.y) / rayDirection.y,
            rayDirection.z == 0.0f ? INFINITY : ((float)currentVoxel.z + (step.z > 0 ? 1.0f : 0.0f) - rayOrigin.z) / rayDirection.z
        );
        Vector3 tDelta = Vector3(
            rayDirection.x == 0.0f ? INFINITY : (float)step.x / rayDirection.x,
            rayDirection.y == 0.0f ? INFINITY : (float)step.y / rayDirection.y,
            rayDirection.z == 0.0f ? INFINITY : (float)step.z / rayDirection.z
        );

        uint8_t blockID = BLOCK_ID_AIR;
        enum {
            AXIS_X,
            AXIS_Y,
            AXIS_Z
        } hitAxis = AXIS_X;

        while (blockID == BLOCK_ID_AIR) {
            if (tMax.x < tMax.y) {
                if (tMax.x < tMax.z) {
                    currentVoxel.x += step.x;
                    if (currentVoxel.x < 0 || currentVoxel.x >= WORLD_SIZE) {
                        blockID = BLOCK_ID_INVALID;
                        break;
                    }
                    tMax.x += tDelta.x;
                    hitAxis = AXIS_X;
                }
                else {
                    currentVoxel.z += step.z;
                    if (currentVoxel.z < 0 || currentVoxel.z >= WORLD_SIZE) {
                        blockID = BLOCK_ID_INVALID;
                        break;
                    }
                    tMax.z += tDelta.z;
                    hitAxis = AXIS_Z;
                }
            }
            else {
                if (tMax.y < tMax.z) {
                    currentVoxel.y += step.y;
                    if (currentVoxel.y < 0 || currentVoxel.y >= WORLD_HEIGHT) {
                        blockID = BLOCK_ID_INVALID;
                        break;
                    }
                    tMax.y += tDelta.y;
                    hitAxis = AXIS_Y;
                }
                else {
                    currentVoxel.z += step.z;
                    if (currentVoxel.z < 0 || currentVoxel.z >= WORLD_SIZE) {
                        blockID = BLOCK_ID_INVALID;
                        break;
                    }
                    tMax.z += tDelta.z;
                    hitAxis = AXIS_Z;
                }
            }
            blockID = blockIDs[currentVoxel.x + currentVoxel.y * WORLD_SIZE + currentVoxel.z * WORLD_HEIGHT * WORLD_SIZE];
        }

        Vector3 hitPosition = Vector3(
            (float)currentVoxel.x + (hitAxis == AXIS_X ? (float)step.x : tMax.x * rayDirection.x),
            (float)currentVoxel.y + (hitAxis == AXIS_Y ? (float)step.y : tMax.y * rayDirection.y),
            (float)currentVoxel.z + (hitAxis == AXIS_Z ? (float)step.z : tMax.z * rayDirection.z)
        );

        Vector3 color = Vector3(
            (float)((blockDescriptors[blockID].color >> 16) & 0xff),
            (float)((blockDescriptors[blockID].color >> 8) & 0xff),
            (float)(blockDescriptors[blockID].color & 0xff)
        );

        if (hitAxis == AXIS_X) {
            color *= (step.x < 0 ? 0.9f : 0.6f);
        }
        else if (hitAxis == AXIS_Y) {
            color *= (step.y < 0 ? 1.0f : 0.2f);
        }
        else if (hitAxis == AXIS_Z) {
            color *= (step.z < 0 ? 0.7f : 0.4f);
        }

        if (blockID == BLOCK_ID_INVALID) {
            color.x = 55.0f;
            color.y = 155.0f;
            color.z = 255.0f;
        }

        devicePixelBuffer[i] = (uint32_t)color.x << 16 | (uint32_t)color.y << 8 | (uint32_t)color.z;
    }
}

__global__ void RenderImageKernel(uint32_t* devicePixelBuffer, uint16_t screenWidth, uint16_t screenHeight, uint32_t* deviceImagePixelBuffer, uint16_t imageWidth, uint16_t imageHeight, Renderer::Rectangle source, Renderer::Rectangle destination) {
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

Renderer::Renderer(uint16_t _width, uint16_t _height, BlockDescriptor* _blockDescriptors) {
    this->width = _width;
    this->height = _height;
    this->pixelBuffer = (uint32_t*)malloc((uint32_t)this->width * (uint32_t)this->height * sizeof(uint32_t));
    cudaMalloc(&this->devicePixelBuffer, (uint32_t)this->width * (uint32_t)this->height * sizeof(uint32_t));
    cudaMalloc(&this->blockIDs, WORLD_SIZE * WORLD_HEIGHT * WORLD_SIZE * sizeof(uint8_t));
    cudaMalloc(&this->blockDescriptors, 256 * sizeof(BlockDescriptor));
    for (uint16_t i = 0; i < 256; i++) {
        cudaMemcpy(&this->blockDescriptors[i], &_blockDescriptors[i], sizeof(BlockDescriptor), cudaMemcpyHostToDevice);
    }
}

Renderer::~Renderer() {
    free(this->pixelBuffer);
    cudaFree(this->devicePixelBuffer);
    cudaFree(this->blockIDs);
    cudaFree(this->blockDescriptors);
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

    cudaDeviceSynchronize();
}

void Renderer::Render(Vector3 cameraPosition, Vector2 cameraRotation, uint8_t* _blockIDs) {
    uint16_t blockSize = 1024;
    uint32_t blockCount = ((uint32_t)this->width * (uint32_t)this->height + blockSize - 1) / blockSize;

    cudaMemcpy(this->blockIDs, _blockIDs, WORLD_SIZE * WORLD_HEIGHT * WORLD_SIZE * sizeof(uint8_t), cudaMemcpyHostToDevice);

    RenderKernel << <blockCount, blockSize >> > (this->devicePixelBuffer, this->width, this->height, cameraPosition, cameraRotation, this->blockIDs, this->blockDescriptors);

    cudaDeviceSynchronize();
}

void Renderer::RenderImage(Image* image, Renderer::Rectangle source, Renderer::Rectangle destination) {
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

    RenderImageKernel << <blockCount, blockSize >> > (this->devicePixelBuffer, this->width, this->height, image->GetDevicePixelBuffer(), image->GetWidth(), image->GetHeight(), source, destination);

    cudaDeviceSynchronize();
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
