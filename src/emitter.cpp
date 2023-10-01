#include "headers/emitter.hpp"

#include <stdarg.h>

void Emitter::On(std::wstring event, std::function<bool(void*)> callback) {
    Listener listener;
    listener.event = event;
    listener.callback = callback;
    this->listeners.push_back(listener);
}

void Emitter::Emit(std::wstring event, void* data) {
    for (int i = 0; i < this->listeners.size(); i++) {
        if (this->listeners[i].event == event && !this->listeners[i].callback(data)) {
            this->listeners.erase(this->listeners.begin() + i);
        }
    }
}
