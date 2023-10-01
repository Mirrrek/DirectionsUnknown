#ifndef LOCKER_HPP
#define LOCKER_HPP

#include <atomic>

class Locker {
private:
    class Mutex {
    public:
        void Lock();
        void Unlock();
    private:
        std::atomic_flag locked;
    };

public:
    static Mutex rendererBuffers;
};

#endif
