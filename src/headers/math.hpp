#ifndef MATH_HPP
#define MATH_HPP

#include <math.h>
#include <stdint.h>

#ifdef __host__
#define CUDA_LOCATION __host__ __device__
#else
#define CUDA_LOCATION
#endif

struct Vector2 {
    float x, y;

    CUDA_LOCATION Vector2() : x(0.0f), y(0.0f) {}
    CUDA_LOCATION Vector2(float _x, float _y) : x(_x), y(_y) {}

    CUDA_LOCATION Vector2 operator+(const Vector2& other) {
        return Vector2(x + other.x, y + other.y);
    }

    CUDA_LOCATION Vector2 operator-(const Vector2& other) {
        return Vector2(x - other.x, y - other.y);
    }

    CUDA_LOCATION Vector2 operator*(const Vector2& other) {
        return Vector2(x * other.x, y * other.y);
    }

    CUDA_LOCATION Vector2 operator/(const Vector2& other) {
        return Vector2(x / other.x, y / other.y);
    }

    CUDA_LOCATION Vector2 operator+=(const Vector2& other) {
        return Vector2(x += other.x, y += other.y);
    }

    CUDA_LOCATION Vector2 operator-=(const Vector2& other) {
        return Vector2(x -= other.x, y -= other.y);
    }

    CUDA_LOCATION Vector2 operator*=(const Vector2& other) {
        return Vector2(x *= other.x, y *= other.y);
    }

    CUDA_LOCATION Vector2 operator/=(const Vector2& other) {
        return Vector2(x /= other.x, y /= other.y);
    }

    CUDA_LOCATION Vector2 operator+(const float& other) {
        return Vector2(x + other, y + other);
    }

    CUDA_LOCATION Vector2 operator-(const float& other) {
        return Vector2(x - other, y - other);
    }

    CUDA_LOCATION Vector2 operator*(const float& other) {
        return Vector2(x * other, y * other);
    }

    CUDA_LOCATION Vector2 operator/(const float& other) {
        return Vector2(x / other, y / other);
    }

    CUDA_LOCATION Vector2 operator+=(const float& other) {
        return Vector2(x += other, y += other);
    }

    CUDA_LOCATION Vector2 operator-=(const float& other) {
        return Vector2(x -= other, y -= other);
    }

    CUDA_LOCATION Vector2 operator*=(const float& other) {
        return Vector2(x *= other, y *= other);
    }

    CUDA_LOCATION Vector2 operator/=(const float& other) {
        return Vector2(x /= other, y /= other);
    }

    CUDA_LOCATION float Length() {
        return sqrtf(x * x + y * y);
    }

    CUDA_LOCATION Vector2 Normalized() {
        float length = Length();
        return Vector2(x / length, y / length);
    }
};

struct Vector3 {
    float x, y, z;

    CUDA_LOCATION Vector3() : x(0.0f), y(0.0f), z(0.0f) {}
    CUDA_LOCATION Vector3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}

    CUDA_LOCATION Vector3 operator+(const Vector3& other) {
        return Vector3(x + other.x, y + other.y, z + other.z);
    }

    CUDA_LOCATION Vector3 operator-(const Vector3& other) {
        return Vector3(x - other.x, y - other.y, z - other.z);
    }

    CUDA_LOCATION Vector3 operator*(const Vector3& other) {
        return Vector3(x * other.x, y * other.y, z * other.z);
    }

    CUDA_LOCATION Vector3 operator/(const Vector3& other) {
        return Vector3(x / other.x, y / other.y, z / other.z);
    }

    CUDA_LOCATION Vector3 operator+=(const Vector3& other) {
        return Vector3(x += other.x, y += other.y, z += other.z);
    }

    CUDA_LOCATION Vector3 operator-=(const Vector3& other) {
        return Vector3(x -= other.x, y -= other.y, z -= other.z);
    }

    CUDA_LOCATION Vector3 operator*=(const Vector3& other) {
        return Vector3(x *= other.x, y *= other.y, z *= other.z);
    }

    CUDA_LOCATION Vector3 operator/=(const Vector3& other) {
        return Vector3(x /= other.x, y /= other.y, z /= other.z);
    }

    CUDA_LOCATION Vector3 operator+(const float& other) {
        return Vector3(x + other, y + other, z + other);
    }

    CUDA_LOCATION Vector3 operator-(const float& other) {
        return Vector3(x - other, y - other, z - other);
    }

    CUDA_LOCATION Vector3 operator*(const float& other) {
        return Vector3(x * other, y * other, z * other);
    }

    CUDA_LOCATION Vector3 operator/(const float& other) {
        return Vector3(x / other, y / other, z / other);
    }

    CUDA_LOCATION Vector3 operator+=(const float& other) {
        return Vector3(x += other, y += other, z += other);
    }

    CUDA_LOCATION Vector3 operator-=(const float& other) {
        return Vector3(x -= other, y -= other, z -= other);
    }

    CUDA_LOCATION Vector3 operator*=(const float& other) {
        return Vector3(x *= other, y *= other, z *= other);
    }

    CUDA_LOCATION Vector3 operator/=(const float& other) {
        return Vector3(x /= other, y /= other, z /= other);
    }

    CUDA_LOCATION float Length() {
        return sqrtf(x * x + y * y + z * z);
    }

    CUDA_LOCATION Vector3 Normalized() {
        float length = Length();
        return Vector3(x / length, y / length, z / length);
    }
};

struct Vector2i {
    int32_t x, y;

    CUDA_LOCATION Vector2i() : x(0), y(0) {}
    CUDA_LOCATION Vector2i(int32_t _x, int32_t _y) : x(_x), y(_y) {}

    CUDA_LOCATION Vector2i operator+(const Vector2i& other) {
        return Vector2i(x + other.x, y + other.y);
    }

    CUDA_LOCATION Vector2i operator-(const Vector2i& other) {
        return Vector2i(x - other.x, y - other.y);
    }

    CUDA_LOCATION Vector2i operator*(const Vector2i& other) {
        return Vector2i(x * other.x, y * other.y);
    }

    CUDA_LOCATION Vector2i operator/(const Vector2i& other) {
        return Vector2i(x / other.x, y / other.y);
    }

    CUDA_LOCATION Vector2i operator+=(const Vector2i& other) {
        return Vector2i(x += other.x, y += other.y);
    }

    CUDA_LOCATION Vector2i operator-=(const Vector2i& other) {
        return Vector2i(x -= other.x, y -= other.y);
    }

    CUDA_LOCATION Vector2i operator*=(const Vector2i& other) {
        return Vector2i(x *= other.x, y *= other.y);
    }

    CUDA_LOCATION Vector2i operator/=(const Vector2i& other) {
        return Vector2i(x /= other.x, y /= other.y);
    }

    CUDA_LOCATION Vector2i operator+(const int32_t& other) {
        return Vector2i(x + other, y + other);
    }

    CUDA_LOCATION Vector2i operator-(const int32_t& other) {
        return Vector2i(x - other, y - other);
    }

    CUDA_LOCATION Vector2i operator*(const int32_t& other) {
        return Vector2i(x * other, y * other);
    }

    CUDA_LOCATION Vector2i operator/(const int32_t& other) {
        return Vector2i(x / other, y / other);
    }

    CUDA_LOCATION Vector2i operator+=(const int32_t& other) {
        return Vector2i(x += other, y += other);
    }

    CUDA_LOCATION Vector2i operator-=(const int32_t& other) {
        return Vector2i(x -= other, y -= other);
    }

    CUDA_LOCATION Vector2i operator*=(const int32_t& other) {
        return Vector2i(x *= other, y *= other);
    }

    CUDA_LOCATION Vector2i operator/=(const int32_t& other) {
        return Vector2i(x /= other, y /= other);
    }

    CUDA_LOCATION float Length() {
        return sqrtf((float)(this->x * this->x + this->y * this->y));
    }

    CUDA_LOCATION Vector2i Normalized() {
        float length = this->Length();
        return { (int32_t)((float)this->x / length), (int32_t)((float)this->y / length) };
    }
};

struct Vector3i {
    int32_t x, y, z;

    CUDA_LOCATION Vector3i() : x(0), y(0), z(0) {}
    CUDA_LOCATION Vector3i(int32_t _x, int32_t _y, int32_t _z) : x(_x), y(_y), z(_z) {}

    CUDA_LOCATION Vector3i operator+(const Vector3i& other) {
        return Vector3i(x + other.x, y + other.y, z + other.z);
    }

    CUDA_LOCATION Vector3i operator-(const Vector3i& other) {
        return Vector3i(x - other.x, y - other.y, z - other.z);
    }

    CUDA_LOCATION Vector3i operator*(const Vector3i& other) {
        return Vector3i(x * other.x, y * other.y, z * other.z);
    }

    CUDA_LOCATION Vector3i operator/(const Vector3i& other) {
        return Vector3i(x / other.x, y / other.y, z / other.z);
    }

    CUDA_LOCATION Vector3i operator+=(const Vector3i& other) {
        return Vector3i(x += other.x, y += other.y, z += other.z);
    }

    CUDA_LOCATION Vector3i operator-=(const Vector3i& other) {
        return Vector3i(x -= other.x, y -= other.y, z -= other.z);
    }

    CUDA_LOCATION Vector3i operator*=(const Vector3i& other) {
        return Vector3i(x *= other.x, y *= other.y, z *= other.z);
    }

    CUDA_LOCATION Vector3i operator/=(const Vector3i& other) {
        return Vector3i(x /= other.x, y /= other.y, z /= other.z);
    }

    CUDA_LOCATION Vector3i operator+(const int32_t& other) {
        return Vector3i(x + other, y + other, z + other);
    }

    CUDA_LOCATION Vector3i operator-(const int32_t& other) {
        return Vector3i(x - other, y - other, z - other);
    }

    CUDA_LOCATION Vector3i operator*(const int32_t& other) {
        return Vector3i(x * other, y * other, z * other);
    }

    CUDA_LOCATION Vector3i operator/(const int32_t& other) {
        return Vector3i(x / other, y / other, z / other);
    }

    CUDA_LOCATION Vector3i operator+=(const int32_t& other) {
        return Vector3i(x += other, y += other, z += other);
    }

    CUDA_LOCATION Vector3i operator-=(const int32_t& other) {
        return Vector3i(x -= other, y -= other, z -= other);
    }

    CUDA_LOCATION Vector3i operator*=(const int32_t& other) {
        return Vector3i(x *= other, y *= other, z *= other);
    }

    CUDA_LOCATION Vector3i operator/=(const int32_t& other) {
        return Vector3i(x /= other, y /= other, z /= other);
    }

    CUDA_LOCATION float Length() {
        return sqrtf((float)(this->x * this->x + this->y * this->y + this->z * this->z));
    }

    CUDA_LOCATION Vector3i Normalized() {
        float length = this->Length();
        return { (int32_t)((float)this->x / length), (int32_t)((float)this->y / length), (int32_t)((float)this->z / length) };
    }
};

#endif
