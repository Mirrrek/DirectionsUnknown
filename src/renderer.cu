#include "headers/renderer.hpp"
#include "headers/locker.hpp"
#include "headers/world.hpp"
#include "headers/math.hpp"
#include "headers/log.hpp"

uint8_t textFont[128][8] = {
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+0000 (nul)
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+0001
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+0002
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+0003
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+0004
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+0005
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+0006
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+0007
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+0008
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+0009
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+000A
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+000B
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+000C
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+000D
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+000E
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+000F
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+0010
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+0011
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+0012
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+0013
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+0014
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+0015
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+0016
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+0017
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+0018
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+0019
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+001A
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+001B
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+001C
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+001D
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+001E
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+001F
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+0020 (space)
    { 0x18, 0x3C, 0x3C, 0x18, 0x18, 0x00, 0x18, 0x00},   // U+0021 (!)
    { 0x36, 0x36, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+0022 (")
    { 0x36, 0x36, 0x7F, 0x36, 0x7F, 0x36, 0x36, 0x00},   // U+0023 (#)
    { 0x0C, 0x3E, 0x03, 0x1E, 0x30, 0x1F, 0x0C, 0x00},   // U+0024 ($)
    { 0x00, 0x63, 0x33, 0x18, 0x0C, 0x66, 0x63, 0x00},   // U+0025 (%)
    { 0x1C, 0x36, 0x1C, 0x6E, 0x3B, 0x33, 0x6E, 0x00},   // U+0026 (&)
    { 0x06, 0x06, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+0027 (')
    { 0x18, 0x0C, 0x06, 0x06, 0x06, 0x0C, 0x18, 0x00},   // U+0028 (()
    { 0x06, 0x0C, 0x18, 0x18, 0x18, 0x0C, 0x06, 0x00},   // U+0029 ())
    { 0x00, 0x66, 0x3C, 0xFF, 0x3C, 0x66, 0x00, 0x00},   // U+002A (*)
    { 0x00, 0x0C, 0x0C, 0x3F, 0x0C, 0x0C, 0x00, 0x00},   // U+002B (+)
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C, 0x06},   // U+002C (,)
    { 0x00, 0x00, 0x00, 0x3F, 0x00, 0x00, 0x00, 0x00},   // U+002D (-)
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C, 0x00},   // U+002E (.)
    { 0x60, 0x30, 0x18, 0x0C, 0x06, 0x03, 0x01, 0x00},   // U+002F (/)
    { 0x3E, 0x63, 0x73, 0x7B, 0x6F, 0x67, 0x3E, 0x00},   // U+0030 (0)
    { 0x0C, 0x0E, 0x0C, 0x0C, 0x0C, 0x0C, 0x3F, 0x00},   // U+0031 (1)
    { 0x1E, 0x33, 0x30, 0x1C, 0x06, 0x33, 0x3F, 0x00},   // U+0032 (2)
    { 0x1E, 0x33, 0x30, 0x1C, 0x30, 0x33, 0x1E, 0x00},   // U+0033 (3)
    { 0x38, 0x3C, 0x36, 0x33, 0x7F, 0x30, 0x78, 0x00},   // U+0034 (4)
    { 0x3F, 0x03, 0x1F, 0x30, 0x30, 0x33, 0x1E, 0x00},   // U+0035 (5)
    { 0x1C, 0x06, 0x03, 0x1F, 0x33, 0x33, 0x1E, 0x00},   // U+0036 (6)
    { 0x3F, 0x33, 0x30, 0x18, 0x0C, 0x0C, 0x0C, 0x00},   // U+0037 (7)
    { 0x1E, 0x33, 0x33, 0x1E, 0x33, 0x33, 0x1E, 0x00},   // U+0038 (8)
    { 0x1E, 0x33, 0x33, 0x3E, 0x30, 0x18, 0x0E, 0x00},   // U+0039 (9)
    { 0x00, 0x0C, 0x0C, 0x00, 0x00, 0x0C, 0x0C, 0x00},   // U+003A (:)
    { 0x00, 0x0C, 0x0C, 0x00, 0x00, 0x0C, 0x0C, 0x06},   // U+003B (;)
    { 0x18, 0x0C, 0x06, 0x03, 0x06, 0x0C, 0x18, 0x00},   // U+003C (<)
    { 0x00, 0x00, 0x3F, 0x00, 0x00, 0x3F, 0x00, 0x00},   // U+003D (=)
    { 0x06, 0x0C, 0x18, 0x30, 0x18, 0x0C, 0x06, 0x00},   // U+003E (>)
    { 0x1E, 0x33, 0x30, 0x18, 0x0C, 0x00, 0x0C, 0x00},   // U+003F (?)
    { 0x3E, 0x63, 0x7B, 0x7B, 0x7B, 0x03, 0x1E, 0x00},   // U+0040 (@)
    { 0x0C, 0x1E, 0x33, 0x33, 0x3F, 0x33, 0x33, 0x00},   // U+0041 (A)
    { 0x3F, 0x66, 0x66, 0x3E, 0x66, 0x66, 0x3F, 0x00},   // U+0042 (B)
    { 0x3C, 0x66, 0x03, 0x03, 0x03, 0x66, 0x3C, 0x00},   // U+0043 (C)
    { 0x1F, 0x36, 0x66, 0x66, 0x66, 0x36, 0x1F, 0x00},   // U+0044 (D)
    { 0x7F, 0x46, 0x16, 0x1E, 0x16, 0x46, 0x7F, 0x00},   // U+0045 (E)
    { 0x7F, 0x46, 0x16, 0x1E, 0x16, 0x06, 0x0F, 0x00},   // U+0046 (F)
    { 0x3C, 0x66, 0x03, 0x03, 0x73, 0x66, 0x7C, 0x00},   // U+0047 (G)
    { 0x33, 0x33, 0x33, 0x3F, 0x33, 0x33, 0x33, 0x00},   // U+0048 (H)
    { 0x1E, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00},   // U+0049 (I)
    { 0x78, 0x30, 0x30, 0x30, 0x33, 0x33, 0x1E, 0x00},   // U+004A (J)
    { 0x67, 0x66, 0x36, 0x1E, 0x36, 0x66, 0x67, 0x00},   // U+004B (K)
    { 0x0F, 0x06, 0x06, 0x06, 0x46, 0x66, 0x7F, 0x00},   // U+004C (L)
    { 0x63, 0x77, 0x7F, 0x7F, 0x6B, 0x63, 0x63, 0x00},   // U+004D (M)
    { 0x63, 0x67, 0x6F, 0x7B, 0x73, 0x63, 0x63, 0x00},   // U+004E (N)
    { 0x1C, 0x36, 0x63, 0x63, 0x63, 0x36, 0x1C, 0x00},   // U+004F (O)
    { 0x3F, 0x66, 0x66, 0x3E, 0x06, 0x06, 0x0F, 0x00},   // U+0050 (P)
    { 0x1E, 0x33, 0x33, 0x33, 0x3B, 0x1E, 0x38, 0x00},   // U+0051 (Q)
    { 0x3F, 0x66, 0x66, 0x3E, 0x36, 0x66, 0x67, 0x00},   // U+0052 (R)
    { 0x1E, 0x33, 0x07, 0x0E, 0x38, 0x33, 0x1E, 0x00},   // U+0053 (S)
    { 0x3F, 0x2D, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00},   // U+0054 (T)
    { 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x3F, 0x00},   // U+0055 (U)
    { 0x33, 0x33, 0x33, 0x33, 0x33, 0x1E, 0x0C, 0x00},   // U+0056 (V)
    { 0x63, 0x63, 0x63, 0x6B, 0x7F, 0x77, 0x63, 0x00},   // U+0057 (W)
    { 0x63, 0x63, 0x36, 0x1C, 0x1C, 0x36, 0x63, 0x00},   // U+0058 (X)
    { 0x33, 0x33, 0x33, 0x1E, 0x0C, 0x0C, 0x1E, 0x00},   // U+0059 (Y)
    { 0x7F, 0x63, 0x31, 0x18, 0x4C, 0x66, 0x7F, 0x00},   // U+005A (Z)
    { 0x1E, 0x06, 0x06, 0x06, 0x06, 0x06, 0x1E, 0x00},   // U+005B ([)
    { 0x03, 0x06, 0x0C, 0x18, 0x30, 0x60, 0x40, 0x00},   // U+005C (\)
    { 0x1E, 0x18, 0x18, 0x18, 0x18, 0x18, 0x1E, 0x00},   // U+005D (])
    { 0x08, 0x1C, 0x36, 0x63, 0x00, 0x00, 0x00, 0x00},   // U+005E (^)
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF},   // U+005F (_)
    { 0x0C, 0x0C, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+0060 (`)
    { 0x00, 0x00, 0x1E, 0x30, 0x3E, 0x33, 0x6E, 0x00},   // U+0061 (a)
    { 0x07, 0x06, 0x06, 0x3E, 0x66, 0x66, 0x3B, 0x00},   // U+0062 (b)
    { 0x00, 0x00, 0x1E, 0x33, 0x03, 0x33, 0x1E, 0x00},   // U+0063 (c)
    { 0x38, 0x30, 0x30, 0x3e, 0x33, 0x33, 0x6E, 0x00},   // U+0064 (d)
    { 0x00, 0x00, 0x1E, 0x33, 0x3f, 0x03, 0x1E, 0x00},   // U+0065 (e)
    { 0x1C, 0x36, 0x06, 0x0f, 0x06, 0x06, 0x0F, 0x00},   // U+0066 (f)
    { 0x00, 0x00, 0x6E, 0x33, 0x33, 0x3E, 0x30, 0x1F},   // U+0067 (g)
    { 0x07, 0x06, 0x36, 0x6E, 0x66, 0x66, 0x67, 0x00},   // U+0068 (h)
    { 0x0C, 0x00, 0x0E, 0x0C, 0x0C, 0x0C, 0x1E, 0x00},   // U+0069 (i)
    { 0x30, 0x00, 0x30, 0x30, 0x30, 0x33, 0x33, 0x1E},   // U+006A (j)
    { 0x07, 0x06, 0x66, 0x36, 0x1E, 0x36, 0x67, 0x00},   // U+006B (k)
    { 0x0E, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00},   // U+006C (l)
    { 0x00, 0x00, 0x33, 0x7F, 0x7F, 0x6B, 0x63, 0x00},   // U+006D (m)
    { 0x00, 0x00, 0x1F, 0x33, 0x33, 0x33, 0x33, 0x00},   // U+006E (n)
    { 0x00, 0x00, 0x1E, 0x33, 0x33, 0x33, 0x1E, 0x00},   // U+006F (o)
    { 0x00, 0x00, 0x3B, 0x66, 0x66, 0x3E, 0x06, 0x0F},   // U+0070 (p)
    { 0x00, 0x00, 0x6E, 0x33, 0x33, 0x3E, 0x30, 0x78},   // U+0071 (q)
    { 0x00, 0x00, 0x3B, 0x6E, 0x66, 0x06, 0x0F, 0x00},   // U+0072 (r)
    { 0x00, 0x00, 0x3E, 0x03, 0x1E, 0x30, 0x1F, 0x00},   // U+0073 (s)
    { 0x08, 0x0C, 0x3E, 0x0C, 0x0C, 0x2C, 0x18, 0x00},   // U+0074 (t)
    { 0x00, 0x00, 0x33, 0x33, 0x33, 0x33, 0x6E, 0x00},   // U+0075 (u)
    { 0x00, 0x00, 0x33, 0x33, 0x33, 0x1E, 0x0C, 0x00},   // U+0076 (v)
    { 0x00, 0x00, 0x63, 0x6B, 0x7F, 0x7F, 0x36, 0x00},   // U+0077 (w)
    { 0x00, 0x00, 0x63, 0x36, 0x1C, 0x36, 0x63, 0x00},   // U+0078 (x)
    { 0x00, 0x00, 0x33, 0x33, 0x33, 0x3E, 0x30, 0x1F},   // U+0079 (y)
    { 0x00, 0x00, 0x3F, 0x19, 0x0C, 0x26, 0x3F, 0x00},   // U+007A (z)
    { 0x38, 0x0C, 0x0C, 0x07, 0x0C, 0x0C, 0x38, 0x00},   // U+007B ({)
    { 0x18, 0x18, 0x18, 0x00, 0x18, 0x18, 0x18, 0x00},   // U+007C (|)
    { 0x07, 0x0C, 0x0C, 0x38, 0x0C, 0x0C, 0x07, 0x00},   // U+007D (})
    { 0x6E, 0x3B, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+007E (~)
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}    // U+007F
};

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

        devicePixelBuffer[((uint32_t)screenHeight - 1 - (uint32_t)y - (uint32_t)destination.y) * (uint32_t)screenWidth + ((uint32_t)x + (uint32_t)destination.x)] = deviceImagePixelBuffer[(uint32_t)imageY * (uint32_t)imageWidth + (uint32_t)imageX];
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


void Renderer::RenderText(const wchar_t* buffer, uint16_t length, uint16_t x, uint16_t y, uint32_t color, bool background) {
    cudaMemcpy(this->pixelBuffer, this->devicePixelBuffer, (uint32_t)this->width * (uint32_t)this->height * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    uint16_t currentX = x;
    uint16_t currentY = y;

    uint16_t characterIndex = 0;

    while (characterIndex < length && buffer[characterIndex] != '\0') {
        if (buffer[characterIndex] == '\n') {
            currentX = x;
            currentY += 8;
            characterIndex++;
            continue;
        }

        if (currentX + 8 >= this->width) {
            currentX = x;
            currentY += 8;
        }

        if (currentY + 8 >= this->height) {
            break;
        }

        uint8_t* character = textFont[(uint8_t)buffer[characterIndex]];

        for (uint8_t i = 0; i < 8; i++) {
            for (uint8_t j = 0; j < 8; j++) {
                if (character[i] & (1 << j)) {
                    this->pixelBuffer[((uint32_t)this->height - 1 - (uint32_t)currentY - (uint32_t)i) * (uint32_t)this->width + (uint32_t)currentX + (uint32_t)j] = color;
                }
                else if (background) {
                    uint32_t currentColor = this->pixelBuffer[((uint32_t)this->height - 1 - (uint32_t)currentY - (uint32_t)i) * (uint32_t)this->width + (uint32_t)currentX + (uint32_t)j];
                    uint8_t r = (currentColor >> 16) & 0xff;
                    uint8_t g = (currentColor >> 8) & 0xff;
                    uint8_t b = currentColor & 0xff;
                    this->pixelBuffer[((uint32_t)this->height - 1 - (uint32_t)currentY - (uint32_t)i) * (uint32_t)this->width + (uint32_t)currentX + (uint32_t)j] = ((uint32_t)(r / 2) << 16) | ((uint32_t)(g / 2) << 8) | ((uint32_t)(b / 2));
                }
            }
        }

        currentX += 8;
        characterIndex++;
    }

    cudaMemcpy(this->devicePixelBuffer, this->pixelBuffer, (uint32_t)this->width * (uint32_t)this->height * sizeof(uint32_t), cudaMemcpyHostToDevice);
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
