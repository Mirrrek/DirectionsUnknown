#ifndef KEYS_HPP
#define KEYS_HPP

#include "headers/emitter.hpp"
#include <stdint.h>

class Keys : public Emitter {
public:
    void KeyDown(uint16_t scancode);
    void KeyUp(uint16_t scancode);

    struct Key {
        uint8_t code;
        std::wstring name;
        bool isDown;
    };

    static const uint8_t CODE_ESC = 0x00;
    static const uint8_t CODE_F1 = 0x01;
    static const uint8_t CODE_F2 = 0x02;
    static const uint8_t CODE_F3 = 0x03;
    static const uint8_t CODE_F4 = 0x04;
    static const uint8_t CODE_F5 = 0x05;
    static const uint8_t CODE_F6 = 0x06;
    static const uint8_t CODE_F7 = 0x07;
    static const uint8_t CODE_F8 = 0x08;
    static const uint8_t CODE_F9 = 0x09;
    static const uint8_t CODE_F10 = 0x0a;
    static const uint8_t CODE_F11 = 0x0b;
    static const uint8_t CODE_F12 = 0x0c;

    static const uint8_t CODE_BACKTICK = 0x10;
    static const uint8_t CODE_DIGIT_1 = 0x11;
    static const uint8_t CODE_DIGIT_2 = 0x12;
    static const uint8_t CODE_DIGIT_3 = 0x13;
    static const uint8_t CODE_DIGIT_4 = 0x14;
    static const uint8_t CODE_DIGIT_5 = 0x15;
    static const uint8_t CODE_DIGIT_6 = 0x16;
    static const uint8_t CODE_DIGIT_7 = 0x17;
    static const uint8_t CODE_DIGIT_8 = 0x18;
    static const uint8_t CODE_DIGIT_9 = 0x19;
    static const uint8_t CODE_DIGIT_0 = 0x1a;
    static const uint8_t CODE_MINUS = 0x1b;
    static const uint8_t CODE_EQUAL = 0x1c;
    static const uint8_t CODE_BACKSPACE = 0x1d;

    static const uint8_t CODE_TAB = 0x20;
    static const uint8_t CODE_Q = 0x21;
    static const uint8_t CODE_W = 0x22;
    static const uint8_t CODE_E = 0x23;
    static const uint8_t CODE_R = 0x24;
    static const uint8_t CODE_T = 0x25;
    static const uint8_t CODE_Y = 0x26;
    static const uint8_t CODE_U = 0x27;
    static const uint8_t CODE_I = 0x28;
    static const uint8_t CODE_O = 0x29;
    static const uint8_t CODE_P = 0x2a;
    static const uint8_t CODE_LBRACKET = 0x2b;
    static const uint8_t CODE_RBRACKET = 0x2c;
    static const uint8_t CODE_ENTER = 0x2d;

    static const uint8_t CODE_CAPSLOCK = 0x30;
    static const uint8_t CODE_A = 0x31;
    static const uint8_t CODE_S = 0x32;
    static const uint8_t CODE_D = 0x33;
    static const uint8_t CODE_F = 0x34;
    static const uint8_t CODE_G = 0x35;
    static const uint8_t CODE_H = 0x36;
    static const uint8_t CODE_J = 0x37;
    static const uint8_t CODE_K = 0x38;
    static const uint8_t CODE_L = 0x39;
    static const uint8_t CODE_SEMICOLON = 0x3a;
    static const uint8_t CODE_QUOTE = 0x3b;
    static const uint8_t CODE_BACKSLASH = 0x3c;

    static const uint8_t CODE_LSHIFT = 0x40;
    static const uint8_t CODE_Z = 0x41;
    static const uint8_t CODE_X = 0x42;
    static const uint8_t CODE_C = 0x43;
    static const uint8_t CODE_V = 0x44;
    static const uint8_t CODE_B = 0x45;
    static const uint8_t CODE_N = 0x46;
    static const uint8_t CODE_M = 0x47;
    static const uint8_t CODE_COMMA = 0x48;
    static const uint8_t CODE_PERIOD = 0x49;
    static const uint8_t CODE_SLASH = 0x4a;
    static const uint8_t CODE_RSHIFT = 0x4b;

    static const uint8_t CODE_LCTRL = 0x50;
    static const uint8_t CODE_LALT = 0x52;
    static const uint8_t CODE_SPACE = 0x53;
    static const uint8_t CODE_RALT = 0x54;
    static const uint8_t CODE_RCTRL = 0x55;

    static const uint8_t CODE_INSERT = 0x60;
    static const uint8_t CODE_HOME = 0x61;
    static const uint8_t CODE_PGUP = 0x62;
    static const uint8_t CODE_DEL = 0x63;
    static const uint8_t CODE_END = 0x64;
    static const uint8_t CODE_PGDN = 0x65;
    static const uint8_t CODE_UP = 0x66;
    static const uint8_t CODE_LEFT = 0x67;
    static const uint8_t CODE_DOWN = 0x68;
    static const uint8_t CODE_RIGHT = 0x69;

    static const uint8_t CODE_NUMLOCK = 0x70;
    static const uint8_t CODE_NUMPAD_DIV = 0x71;
    static const uint8_t CODE_NUMPAD_MUL = 0x72;
    static const uint8_t CODE_NUMPAD_MINUS = 0x73;
    static const uint8_t CODE_NUMPAD_PLUS = 0x74;
    static const uint8_t CODE_NUMPAD_DECIMAL = 0x76;

    static const uint8_t CODE_NUMPAD_0 = 0x80;
    static const uint8_t CODE_NUMPAD_1 = 0x81;
    static const uint8_t CODE_NUMPAD_2 = 0x82;
    static const uint8_t CODE_NUMPAD_3 = 0x83;
    static const uint8_t CODE_NUMPAD_4 = 0x84;
    static const uint8_t CODE_NUMPAD_5 = 0x85;
    static const uint8_t CODE_NUMPAD_6 = 0x86;
    static const uint8_t CODE_NUMPAD_7 = 0x87;
    static const uint8_t CODE_NUMPAD_8 = 0x88;
    static const uint8_t CODE_NUMPAD_9 = 0x89;

    Key ESC = Key{ CODE_ESC, L"Escape", false };
    Key F1 = Key{ CODE_F1, L"F1", false };
    Key F2 = Key{ CODE_F2, L"F2", false };
    Key F3 = Key{ CODE_F3, L"F3", false };
    Key F4 = Key{ CODE_F4, L"F4", false };
    Key F5 = Key{ CODE_F5, L"F5", false };
    Key F6 = Key{ CODE_F6, L"F6", false };
    Key F7 = Key{ CODE_F7, L"F7", false };
    Key F8 = Key{ CODE_F8, L"F8", false };
    Key F9 = Key{ CODE_F9, L"F9", false };
    Key F10 = Key{ CODE_F10, L"F10", false };
    Key F11 = Key{ CODE_F11, L"F11", false };
    Key F12 = Key{ CODE_F12, L"F12", false };

    Key BACKTICK = Key{ CODE_BACKTICK, L"Backtick", false };
    Key DIGIT_1 = Key{ CODE_DIGIT_1, L"1", false };
    Key DIGIT_2 = Key{ CODE_DIGIT_2, L"2", false };
    Key DIGIT_3 = Key{ CODE_DIGIT_3, L"3", false };
    Key DIGIT_4 = Key{ CODE_DIGIT_4, L"4", false };
    Key DIGIT_5 = Key{ CODE_DIGIT_5, L"5", false };
    Key DIGIT_6 = Key{ CODE_DIGIT_6, L"6", false };
    Key DIGIT_7 = Key{ CODE_DIGIT_7, L"7", false };
    Key DIGIT_8 = Key{ CODE_DIGIT_8, L"8", false };
    Key DIGIT_9 = Key{ CODE_DIGIT_9, L"9", false };
    Key DIGIT_0 = Key{ CODE_DIGIT_0, L"0", false };
    Key MINUS = Key{ CODE_MINUS, L"Minus", false };
    Key EQUAL = Key{ CODE_EQUAL, L"Equal", false };
    Key BACKSPACE = Key{ CODE_BACKSPACE, L"Backspace", false };

    Key TAB = Key{ CODE_TAB, L"Tab", false };
    Key Q = Key{ CODE_Q, L"Q", false };
    Key W = Key{ CODE_W, L"W", false };
    Key E = Key{ CODE_E, L"E", false };
    Key R = Key{ CODE_R, L"R", false };
    Key T = Key{ CODE_T, L"T", false };
    Key Y = Key{ CODE_Y, L"Y", false };
    Key U = Key{ CODE_U, L"U", false };
    Key I = Key{ CODE_I, L"I", false };
    Key O = Key{ CODE_O, L"O", false };
    Key P = Key{ CODE_P, L"P", false };
    Key LBRACKET = Key{ CODE_LBRACKET, L"Left Bracket", false };
    Key RBRACKET = Key{ CODE_RBRACKET, L"Right Bracket", false };
    Key ENTER = Key{ CODE_ENTER, L"Enter", false };

    Key CAPSLOCK = Key{ CODE_CAPSLOCK, L"CapsLock", false };
    Key A = Key{ CODE_A, L"A", false };
    Key S = Key{ CODE_S, L"S", false };
    Key D = Key{ CODE_D, L"D", false };
    Key F = Key{ CODE_F, L"F", false };
    Key G = Key{ CODE_G, L"G", false };
    Key H = Key{ CODE_H, L"H", false };
    Key J = Key{ CODE_J, L"J", false };
    Key K = Key{ CODE_K, L"K", false };
    Key L = Key{ CODE_L, L"L", false };
    Key SEMICOLON = Key{ CODE_SEMICOLON, L"Semicolon", false };
    Key QUOTE = Key{ CODE_QUOTE, L"Quote", false };
    Key BACKSLASH = Key{ CODE_BACKSLASH, L"Backslash", false };

    Key LSHIFT = Key{ CODE_LSHIFT, L"Left Shift", false };
    Key Z = Key{ CODE_Z, L"Z", false };
    Key X = Key{ CODE_X, L"X", false };
    Key C = Key{ CODE_C, L"C", false };
    Key V = Key{ CODE_V, L"V", false };
    Key B = Key{ CODE_B, L"B", false };
    Key N = Key{ CODE_N, L"N", false };
    Key M = Key{ CODE_M, L"M", false };
    Key COMMA = Key{ CODE_COMMA, L"Comma", false };
    Key PERIOD = Key{ CODE_PERIOD, L"Period", false };
    Key SLASH = Key{ CODE_SLASH, L"Slash", false };
    Key RSHIFT = Key{ CODE_RSHIFT, L"Right Shift", false };

    Key LCTRL = Key{ CODE_LCTRL, L"Left Control", false };
    Key LALT = Key{ CODE_LALT, L"Left Alt", false };
    Key SPACE = Key{ CODE_SPACE, L"Space", false };
    Key RALT = Key{ CODE_RALT, L"Right Alt", false };
    Key RCTRL = Key{ CODE_RCTRL, L"Right Control", false };

    Key INSERT = Key{ CODE_INSERT, L"Insert", false };
    Key HOME = Key{ CODE_HOME, L"Home", false };
    Key PGUP = Key{ CODE_PGUP, L"Page Up", false };
    Key DEL = Key{ CODE_DEL, L"Delete", false };
    Key END = Key{ CODE_END, L"End", false };
    Key PGDN = Key{ CODE_PGDN, L"Page Down", false };
    Key UP = Key{ CODE_UP, L"Up Arrow", false };
    Key LEFT = Key{ CODE_LEFT, L"Left Arrow", false };
    Key DOWN = Key{ CODE_DOWN, L"Down Arrow", false };
    Key RIGHT = Key{ CODE_RIGHT, L"Right Arrow", false };

    Key NUMLOCK = Key{ CODE_NUMLOCK, L"NumLock", false };
    Key NUMPAD_DIV = Key{ CODE_NUMPAD_DIV, L"Numpad Divide", false };
    Key NUMPAD_MUL = Key{ CODE_NUMPAD_MUL, L"Numpad Multiply", false };
    Key NUMPAD_MINUS = Key{ CODE_NUMPAD_MINUS, L"Numpad Minus", false };
    Key NUMPAD_PLUS = Key{ CODE_NUMPAD_PLUS, L"Numpad Plus", false };
    Key NUMPAD_DECIMAL = Key{ CODE_NUMPAD_DECIMAL, L"Numpad Decimal", false };

    Key NUMPAD_0 = Key{ CODE_NUMPAD_0, L"Numpad 0", false };
    Key NUMPAD_1 = Key{ CODE_NUMPAD_1, L"Numpad 1", false };
    Key NUMPAD_2 = Key{ CODE_NUMPAD_2, L"Numpad 2", false };
    Key NUMPAD_3 = Key{ CODE_NUMPAD_3, L"Numpad 3", false };
    Key NUMPAD_4 = Key{ CODE_NUMPAD_4, L"Numpad 4", false };
    Key NUMPAD_5 = Key{ CODE_NUMPAD_5, L"Numpad 5", false };
    Key NUMPAD_6 = Key{ CODE_NUMPAD_6, L"Numpad 6", false };
    Key NUMPAD_7 = Key{ CODE_NUMPAD_7, L"Numpad 7", false };
    Key NUMPAD_8 = Key{ CODE_NUMPAD_8, L"Numpad 8", false };
    Key NUMPAD_9 = Key{ CODE_NUMPAD_9, L"Numpad 9", false };
};

#endif
