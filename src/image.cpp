#include "headers/image.hpp"
#include "headers/renderer.hpp"
#include "headers/files.hpp"
#include "headers/log.hpp"

#include <malloc.h>

static const wchar_t* tag = L"Image";

Image::Image(const wchar_t* path) {
    this->width = 0;
    this->height = 0;
    this->file = nullptr;
    this->pixelBuffer = nullptr;

    uint64_t fileSize;
    if (!Files::GetSize(path, &fileSize)) {
        Log::Error(tag, L"Failed to get file size of \"%s\"", path);
        return;
    }

    this->file = (uint8_t*)malloc(fileSize);
    if (!Files::Read(path, this->file, fileSize, false)) {
        Log::Error(tag, L"Failed to read file \"%s\"", path);
        free(this->file);
        this->file = nullptr;
        return;
    }

    if (this->file[0] != 'P' || this->file[1] != '6') {
        Log::Error(tag, L"File \"%s\" contains invalid signature", path);
        free(this->file);
        this->file = nullptr;
        return;
    }

    if (this->file[2] != ' ' && this->file[2] != '\t' && this->file[2] != '\r' && this->file[2] != '\n') {
        Log::Error(tag, L"File \"%s\" is improperly formatted", path);
        free(this->file);
        this->file = nullptr;
        return;
    }

    uint64_t i = 3;
    uint16_t imageWidth = 0;
    uint16_t imageHeight = 0;
    uint8_t maxValue = 0;

    while (i < fileSize) {
        if (this->file[i] == ' ' || this->file[i] == '\t' || this->file[i] == '\r' || this->file[i] == '\n') {
            i++;
            break;
        }

        if (this->file[i] < '0' || this->file[i] > '9') {
            Log::Error(tag, L"File \"%s\" is improperly formatted", path);
            free(this->file);
            this->file = nullptr;
            return;
        }

        imageWidth *= 10;
        imageWidth += this->file[i] - '0';

        i++;
    }

    if (imageWidth <= 0 || imageWidth > 32768) {
        Log::Error(tag, L"File \"%s\" is too wide", path);
        free(this->file);
        this->file = nullptr;
        return;
    }

    while (i < fileSize) {
        if (this->file[i] == ' ' || this->file[i] == '\t' || this->file[i] == '\r' || this->file[i] == '\n') {
            i++;
            break;
        }

        if (this->file[i] < '0' || this->file[i] > '9') {
            Log::Error(tag, L"File \"%s\" is improperly formatted", path);
            free(this->file);
            this->file = nullptr;
            return;
        }

        imageHeight *= 10;
        imageHeight += this->file[i] - '0';

        i++;
    }

    if (imageHeight <= 0 || imageHeight > 32768) {
        Log::Error(tag, L"File \"%s\" is too high", path);
        free(this->file);
        this->file = nullptr;
        return;
    }

    while (i < fileSize) {
        if (this->file[i] == ' ' || this->file[i] == '\t' || this->file[i] == '\r' || this->file[i] == '\n') {
            i++;
            break;
        }

        if (this->file[i] < '0' || this->file[i] > '9') {
            Log::Error(tag, L"File \"%s\" is improperly formatted", path);
            free(this->file);
            this->file = nullptr;
            return;
        }

        maxValue *= 10;
        maxValue += this->file[i] - '0';

        i++;
    }

    if (maxValue != 0xff) {
        Log::Error(tag, L"File \"%s\" is not 8bit deep", path);
        free(this->file);
        this->file = nullptr;
        return;
    }

    if (i + (imageWidth * imageHeight * 3) != fileSize) {
        Log::Error(tag, L"File \"%s\" is improperly formatted", path);
        free(this->file);
        this->file = nullptr;
        return;
    }

    this->width = imageWidth;
    this->height = imageHeight;
    this->pixelBuffer = (uint32_t*)malloc(this->width * this->height * sizeof(uint32_t));

    for (uint16_t y = 0; y < this->height; y++) {
        for (uint16_t x = 0; x < this->width; x++) {
            uint32_t pixel = 0x00000000;
            pixel |= this->file[i++] << 16;
            pixel |= this->file[i++] << 8;
            pixel |= this->file[i++];
            this->pixelBuffer[(y * this->width) + x] = pixel;
        }
    }

    this->devicePixelBuffer = CopyImageToDevice(this);
}

Image::~Image() {
    if (this->file != nullptr) {
        free(this->file);
    }

    if (this->pixelBuffer != nullptr) {
        free(this->pixelBuffer);
    }

    if (this->devicePixelBuffer != nullptr) {
        FreeDeviceImage(this);
    }
}

uint16_t Image::GetWidth() {
    return this->width;
}

uint16_t Image::GetHeight() {
    return this->height;
}

uint32_t* Image::GetPixelBuffer() {
    return this->pixelBuffer;
}

uint32_t* Image::GetDevicePixelBuffer() {
    return this->devicePixelBuffer;
}
