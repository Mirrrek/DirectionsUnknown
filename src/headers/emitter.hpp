#ifndef EMITTER_HPP
#define EMITTER_HPP

#include <functional>
#include <string>

class Emitter {
public:
    void On(std::wstring event, std::function<bool(void*)> callback);

protected:
    void Emit(std::wstring event, void* data);

private:
    struct Listener {
        std::wstring event;
        std::function<bool(void*)> callback;
    };
    std::vector<Listener> listeners;
};

#endif
