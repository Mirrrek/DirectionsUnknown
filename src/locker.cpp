#include "headers/locker.hpp"
#include "headers/log.hpp"
#include <thread>

static const wchar_t* tag = L"Locker";

Locker::Mutex Locker::rendererBuffers;

void Locker::Mutex::Lock() {
    uint16_t yieldCount = 0;
    while (locked.test_and_set(std::memory_order_acquire)) {
        std::this_thread::yield();
        yieldCount++;
        if (yieldCount > 4096) {
            Log::Warn(tag, L"Thread is locked for over 4096 yields, sleeping for 1ms");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            yieldCount = 0;
        }
    }
}

void Locker::Mutex::Unlock() {
    locked.clear(std::memory_order_release);
}
