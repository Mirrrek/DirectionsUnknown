#include "headers/keys.hpp"

void Keys::KeyDown(uint16_t scancode) {
    switch (scancode) {
    case 0x01:
        this->ESC.isDown = true;
        this->Emit(L"keydown", &this->ESC);
        break;
    case 0x3b:
        this->F1.isDown = true;
        this->Emit(L"keydown", &this->F1);
        break;
    case 0x3c:
        this->F2.isDown = true;
        this->Emit(L"keydown", &this->F2);
        break;
    case 0x3d:
        this->F3.isDown = true;
        this->Emit(L"keydown", &this->F3);
        break;
    case 0x3e:
        this->F4.isDown = true;
        this->Emit(L"keydown", &this->F4);
        break;
    case 0x3f:
        this->F5.isDown = true;
        this->Emit(L"keydown", &this->F5);
        break;
    case 0x40:
        this->F6.isDown = true;
        this->Emit(L"keydown", &this->F6);
        break;
    case 0x41:
        this->F7.isDown = true;
        this->Emit(L"keydown", &this->F7);
        break;
    case 0x42:
        this->F8.isDown = true;
        this->Emit(L"keydown", &this->F8);
        break;
    case 0x43:
        this->F9.isDown = true;
        this->Emit(L"keydown", &this->F9);
        break;
    case 0x44:
        this->F10.isDown = true;
        this->Emit(L"keydown", &this->F10);
        break;
    case 0x57:
        this->F11.isDown = true;
        this->Emit(L"keydown", &this->F11);
        break;
    case 0x58:
        this->F12.isDown = true;
        this->Emit(L"keydown", &this->F12);
        break;
    case 0x29:
        this->BACKTICK.isDown = true;
        this->Emit(L"keydown", &this->BACKTICK);
        break;
    case 0x02:
        this->DIGIT_1.isDown = true;
        this->Emit(L"keydown", &this->DIGIT_1);
        break;
    case 0x03:
        this->DIGIT_2.isDown = true;
        this->Emit(L"keydown", &this->DIGIT_2);
        break;
    case 0x04:
        this->DIGIT_3.isDown = true;
        this->Emit(L"keydown", &this->DIGIT_3);
        break;
    case 0x05:
        this->DIGIT_4.isDown = true;
        this->Emit(L"keydown", &this->DIGIT_4);
        break;
    case 0x06:
        this->DIGIT_5.isDown = true;
        this->Emit(L"keydown", &this->DIGIT_5);
        break;
    case 0x07:
        this->DIGIT_6.isDown = true;
        this->Emit(L"keydown", &this->DIGIT_6);
        break;
    case 0x08:
        this->DIGIT_7.isDown = true;
        this->Emit(L"keydown", &this->DIGIT_7);
        break;
    case 0x09:
        this->DIGIT_8.isDown = true;
        this->Emit(L"keydown", &this->DIGIT_8);
        break;
    case 0x0a:
        this->DIGIT_9.isDown = true;
        this->Emit(L"keydown", &this->DIGIT_9);
        break;
    case 0x0b:
        this->DIGIT_0.isDown = true;
        this->Emit(L"keydown", &this->DIGIT_0);
        break;
    case 0x0c:
        this->MINUS.isDown = true;
        this->Emit(L"keydown", &this->MINUS);
        break;
    case 0x0d:
        this->EQUAL.isDown = true;
        this->Emit(L"keydown", &this->EQUAL);
        break;
    case 0x0e:
        this->BACKSPACE.isDown = true;
        this->Emit(L"keydown", &this->BACKSPACE);
        break;
    case 0x0f:
        this->TAB.isDown = true;
        this->Emit(L"keydown", &this->TAB);
        break;
    case 0x10:
        this->Q.isDown = true;
        this->Emit(L"keydown", &this->Q);
        break;
    case 0x11:
        this->W.isDown = true;
        this->Emit(L"keydown", &this->W);
        break;
    case 0x12:
        this->E.isDown = true;
        this->Emit(L"keydown", &this->E);
        break;
    case 0x13:
        this->R.isDown = true;
        this->Emit(L"keydown", &this->R);
        break;
    case 0x14:
        this->T.isDown = true;
        this->Emit(L"keydown", &this->T);
        break;
    case 0x15:
        this->Y.isDown = true;
        this->Emit(L"keydown", &this->Y);
        break;
    case 0x16:
        this->U.isDown = true;
        this->Emit(L"keydown", &this->U);
        break;
    case 0x17:
        this->I.isDown = true;
        this->Emit(L"keydown", &this->I);
        break;
    case 0x18:
        this->O.isDown = true;
        this->Emit(L"keydown", &this->O);
        break;
    case 0x19:
        this->P.isDown = true;
        this->Emit(L"keydown", &this->P);
        break;
    case 0x1a:
        this->LBRACKET.isDown = true;
        this->Emit(L"keydown", &this->LBRACKET);
        break;
    case 0x1b:
        this->RBRACKET.isDown = true;
        this->Emit(L"keydown", &this->RBRACKET);
        break;
    case 0x1c:
        this->ENTER.isDown = true;
        this->Emit(L"keydown", &this->ENTER);
        break;
    case 0x3a:
        this->CAPSLOCK.isDown = true;
        this->Emit(L"keydown", &this->CAPSLOCK);
        break;
    case 0x1e:
        this->A.isDown = true;
        this->Emit(L"keydown", &this->A);
        break;
    case 0x1f:
        this->S.isDown = true;
        this->Emit(L"keydown", &this->S);
        break;
    case 0x20:
        this->D.isDown = true;
        this->Emit(L"keydown", &this->D);
        break;
    case 0x21:
        this->F.isDown = true;
        this->Emit(L"keydown", &this->F);
        break;
    case 0x22:
        this->G.isDown = true;
        this->Emit(L"keydown", &this->G);
        break;
    case 0x23:
        this->H.isDown = true;
        this->Emit(L"keydown", &this->H);
        break;
    case 0x24:
        this->J.isDown = true;
        this->Emit(L"keydown", &this->J);
        break;
    case 0x25:
        this->K.isDown = true;
        this->Emit(L"keydown", &this->K);
        break;
    case 0x26:
        this->L.isDown = true;
        this->Emit(L"keydown", &this->L);
        break;
    case 0x27:
        this->SEMICOLON.isDown = true;
        this->Emit(L"keydown", &this->SEMICOLON);
        break;
    case 0x28:
        this->QUOTE.isDown = true;
        this->Emit(L"keydown", &this->QUOTE);
        break;
    case 0x2b:
        this->BACKSLASH.isDown = true;
        this->Emit(L"keydown", &this->BACKSLASH);
        break;
    case 0x2a:
        this->LSHIFT.isDown = true;
        this->Emit(L"keydown", &this->LSHIFT);
        break;
    case 0x2c:
        this->Z.isDown = true;
        this->Emit(L"keydown", &this->Z);
        break;
    case 0x2d:
        this->X.isDown = true;
        this->Emit(L"keydown", &this->X);
        break;
    case 0x2e:
        this->C.isDown = true;
        this->Emit(L"keydown", &this->C);
        break;
    case 0x2f:
        this->V.isDown = true;
        this->Emit(L"keydown", &this->V);
        break;
    case 0x30:
        this->B.isDown = true;
        this->Emit(L"keydown", &this->B);
        break;
    case 0x31:
        this->N.isDown = true;
        this->Emit(L"keydown", &this->N);
        break;
    case 0x32:
        this->M.isDown = true;
        this->Emit(L"keydown", &this->M);
        break;
    case 0x33:
        this->COMMA.isDown = true;
        this->Emit(L"keydown", &this->COMMA);
        break;
    case 0x34:
        this->PERIOD.isDown = true;
        this->Emit(L"keydown", &this->PERIOD);
        break;
    case 0x35:
        this->SLASH.isDown = true;
        this->Emit(L"keydown", &this->SLASH);
        break;
    case 0x36:
        this->RSHIFT.isDown = true;
        this->Emit(L"keydown", &this->RSHIFT);
        break;
    case 0x1d:
        this->LCTRL.isDown = true;
        this->Emit(L"keydown", &this->LCTRL);
        break;
    case 0x38:
        this->LALT.isDown = true;
        this->Emit(L"keydown", &this->LALT);
        break;
    case 0x39:
        this->SPACE.isDown = true;
        this->Emit(L"keydown", &this->SPACE);
        break;
    case 0xe038:
        this->RALT.isDown = true;
        this->Emit(L"keydown", &this->RALT);
        break;
    case 0xe01d:
        this->RCTRL.isDown = true;
        this->Emit(L"keydown", &this->RCTRL);
        break;
    case 0xe052:
        this->INSERT.isDown = true;
        this->Emit(L"keydown", &this->INSERT);
        break;
    case 0xe047:
        this->HOME.isDown = true;
        this->Emit(L"keydown", &this->HOME);
        break;
    case 0xe049:
        this->PGUP.isDown = true;
        this->Emit(L"keydown", &this->PGUP);
        break;
    case 0xe053:
        this->DEL.isDown = true;
        this->Emit(L"keydown", &this->DEL);
        break;
    case 0xe04f:
        this->END.isDown = true;
        this->Emit(L"keydown", &this->END);
        break;
    case 0xe051:
        this->PGDN.isDown = true;
        this->Emit(L"keydown", &this->PGDN);
        break;
    case 0xe048:
        this->UP.isDown = true;
        this->Emit(L"keydown", &this->UP);
        break;
    case 0xe04b:
        this->LEFT.isDown = true;
        this->Emit(L"keydown", &this->LEFT);
        break;
    case 0xe050:
        this->DOWN.isDown = true;
        this->Emit(L"keydown", &this->DOWN);
        break;
    case 0xe04d:
        this->RIGHT.isDown = true;
        this->Emit(L"keydown", &this->RIGHT);
        break;
    case 0x45:
        this->NUMLOCK.isDown = true;
        this->Emit(L"keydown", &this->NUMLOCK);
        break;
    case 0xe035:
        this->NUMPAD_DIV.isDown = true;
        this->Emit(L"keydown", &this->NUMPAD_DIV);
        break;
    case 0x37:
        this->NUMPAD_MUL.isDown = true;
        this->Emit(L"keydown", &this->NUMPAD_MUL);
        break;
    case 0x4a:
        this->NUMPAD_MINUS.isDown = true;
        this->Emit(L"keydown", &this->NUMPAD_MINUS);
        break;
    case 0x4e:
        this->NUMPAD_PLUS.isDown = true;
        this->Emit(L"keydown", &this->NUMPAD_PLUS);
        break;
    case 0x53:
        this->NUMPAD_DECIMAL.isDown = true;
        this->Emit(L"keydown", &this->NUMPAD_DECIMAL);
        break;
    case 0x52:
        this->NUMPAD_0.isDown = true;
        this->Emit(L"keydown", &this->NUMPAD_0);
        break;
    case 0x4f:
        this->NUMPAD_1.isDown = true;
        this->Emit(L"keydown", &this->NUMPAD_1);
        break;
    case 0x50:
        this->NUMPAD_2.isDown = true;
        this->Emit(L"keydown", &this->NUMPAD_2);
        break;
    case 0x51:
        this->NUMPAD_3.isDown = true;
        this->Emit(L"keydown", &this->NUMPAD_3);
        break;
    case 0x4b:
        this->NUMPAD_4.isDown = true;
        this->Emit(L"keydown", &this->NUMPAD_4);
        break;
    case 0x4c:
        this->NUMPAD_5.isDown = true;
        this->Emit(L"keydown", &this->NUMPAD_5);
        break;
    case 0x4d:
        this->NUMPAD_6.isDown = true;
        this->Emit(L"keydown", &this->NUMPAD_6);
        break;
    case 0x47:
        this->NUMPAD_7.isDown = true;
        this->Emit(L"keydown", &this->NUMPAD_7);
        break;
    case 0x48:
        this->NUMPAD_8.isDown = true;
        this->Emit(L"keydown", &this->NUMPAD_8);
        break;
    case 0x49:
        this->NUMPAD_9.isDown = true;
        this->Emit(L"keydown", &this->NUMPAD_9);
        break;
    }
}

void Keys::KeyUp(uint16_t scancode) {
    switch (scancode) {
    case 0x01:
        this->ESC.isDown = false;
        this->Emit(L"keyup", &this->ESC);
        break;
    case 0x3b:
        this->F1.isDown = false;
        this->Emit(L"keyup", &this->F1);
        break;
    case 0x3c:
        this->F2.isDown = false;
        this->Emit(L"keyup", &this->F2);
        break;
    case 0x3d:
        this->F3.isDown = false;
        this->Emit(L"keyup", &this->F3);
        break;
    case 0x3e:
        this->F4.isDown = false;
        this->Emit(L"keyup", &this->F4);
        break;
    case 0x3f:
        this->F5.isDown = false;
        this->Emit(L"keyup", &this->F5);
        break;
    case 0x40:
        this->F6.isDown = false;
        this->Emit(L"keyup", &this->F6);
        break;
    case 0x41:
        this->F7.isDown = false;
        this->Emit(L"keyup", &this->F7);
        break;
    case 0x42:
        this->F8.isDown = false;
        this->Emit(L"keyup", &this->F8);
        break;
    case 0x43:
        this->F9.isDown = false;
        this->Emit(L"keyup", &this->F9);
        break;
    case 0x44:
        this->F10.isDown = false;
        this->Emit(L"keyup", &this->F10);
        break;
    case 0x57:
        this->F11.isDown = false;
        this->Emit(L"keyup", &this->F11);
        break;
    case 0x58:
        this->F12.isDown = false;
        this->Emit(L"keyup", &this->F12);
        break;
    case 0x29:
        this->BACKTICK.isDown = false;
        this->Emit(L"keyup", &this->BACKTICK);
        break;
    case 0x02:
        this->DIGIT_1.isDown = false;
        this->Emit(L"keyup", &this->DIGIT_1);
        break;
    case 0x03:
        this->DIGIT_2.isDown = false;
        this->Emit(L"keyup", &this->DIGIT_2);
        break;
    case 0x04:
        this->DIGIT_3.isDown = false;
        this->Emit(L"keyup", &this->DIGIT_3);
        break;
    case 0x05:
        this->DIGIT_4.isDown = false;
        this->Emit(L"keyup", &this->DIGIT_4);
        break;
    case 0x06:
        this->DIGIT_5.isDown = false;
        this->Emit(L"keyup", &this->DIGIT_5);
        break;
    case 0x07:
        this->DIGIT_6.isDown = false;
        this->Emit(L"keyup", &this->DIGIT_6);
        break;
    case 0x08:
        this->DIGIT_7.isDown = false;
        this->Emit(L"keyup", &this->DIGIT_7);
        break;
    case 0x09:
        this->DIGIT_8.isDown = false;
        this->Emit(L"keyup", &this->DIGIT_8);
        break;
    case 0x0a:
        this->DIGIT_9.isDown = false;
        this->Emit(L"keyup", &this->DIGIT_9);
        break;
    case 0x0b:
        this->DIGIT_0.isDown = false;
        this->Emit(L"keyup", &this->DIGIT_0);
        break;
    case 0x0c:
        this->MINUS.isDown = false;
        this->Emit(L"keyup", &this->MINUS);
        break;
    case 0x0d:
        this->EQUAL.isDown = false;
        this->Emit(L"keyup", &this->EQUAL);
        break;
    case 0x0e:
        this->BACKSPACE.isDown = false;
        this->Emit(L"keyup", &this->BACKSPACE);
        break;
    case 0x0f:
        this->TAB.isDown = false;
        this->Emit(L"keyup", &this->TAB);
        break;
    case 0x10:
        this->Q.isDown = false;
        this->Emit(L"keyup", &this->Q);
        break;
    case 0x11:
        this->W.isDown = false;
        this->Emit(L"keyup", &this->W);
        break;
    case 0x12:
        this->E.isDown = false;
        this->Emit(L"keyup", &this->E);
        break;
    case 0x13:
        this->R.isDown = false;
        this->Emit(L"keyup", &this->R);
        break;
    case 0x14:
        this->T.isDown = false;
        this->Emit(L"keyup", &this->T);
        break;
    case 0x15:
        this->Y.isDown = false;
        this->Emit(L"keyup", &this->Y);
        break;
    case 0x16:
        this->U.isDown = false;
        this->Emit(L"keyup", &this->U);
        break;
    case 0x17:
        this->I.isDown = false;
        this->Emit(L"keyup", &this->I);
        break;
    case 0x18:
        this->O.isDown = false;
        this->Emit(L"keyup", &this->O);
        break;
    case 0x19:
        this->P.isDown = false;
        this->Emit(L"keyup", &this->P);
        break;
    case 0x1a:
        this->LBRACKET.isDown = false;
        this->Emit(L"keyup", &this->LBRACKET);
        break;
    case 0x1b:
        this->RBRACKET.isDown = false;
        this->Emit(L"keyup", &this->RBRACKET);
        break;
    case 0x1c:
        this->ENTER.isDown = false;
        this->Emit(L"keyup", &this->ENTER);
        break;
    case 0x3a:
        this->CAPSLOCK.isDown = false;
        this->Emit(L"keyup", &this->CAPSLOCK);
        break;
    case 0x1e:
        this->A.isDown = false;
        this->Emit(L"keyup", &this->A);
        break;
    case 0x1f:
        this->S.isDown = false;
        this->Emit(L"keyup", &this->S);
        break;
    case 0x20:
        this->D.isDown = false;
        this->Emit(L"keyup", &this->D);
        break;
    case 0x21:
        this->F.isDown = false;
        this->Emit(L"keyup", &this->F);
        break;
    case 0x22:
        this->G.isDown = false;
        this->Emit(L"keyup", &this->G);
        break;
    case 0x23:
        this->H.isDown = false;
        this->Emit(L"keyup", &this->H);
        break;
    case 0x24:
        this->J.isDown = false;
        this->Emit(L"keyup", &this->J);
        break;
    case 0x25:
        this->K.isDown = false;
        this->Emit(L"keyup", &this->K);
        break;
    case 0x26:
        this->L.isDown = false;
        this->Emit(L"keyup", &this->L);
        break;
    case 0x27:
        this->SEMICOLON.isDown = false;
        this->Emit(L"keyup", &this->SEMICOLON);
        break;
    case 0x28:
        this->QUOTE.isDown = false;
        this->Emit(L"keyup", &this->QUOTE);
        break;
    case 0x2b:
        this->BACKSLASH.isDown = false;
        this->Emit(L"keyup", &this->BACKSLASH);
        break;
    case 0x2a:
        this->LSHIFT.isDown = false;
        this->Emit(L"keyup", &this->LSHIFT);
        break;
    case 0x2c:
        this->Z.isDown = false;
        this->Emit(L"keyup", &this->Z);
        break;
    case 0x2d:
        this->X.isDown = false;
        this->Emit(L"keyup", &this->X);
        break;
    case 0x2e:
        this->C.isDown = false;
        this->Emit(L"keyup", &this->C);
        break;
    case 0x2f:
        this->V.isDown = false;
        this->Emit(L"keyup", &this->V);
        break;
    case 0x30:
        this->B.isDown = false;
        this->Emit(L"keyup", &this->B);
        break;
    case 0x31:
        this->N.isDown = false;
        this->Emit(L"keyup", &this->N);
        break;
    case 0x32:
        this->M.isDown = false;
        this->Emit(L"keyup", &this->M);
        break;
    case 0x33:
        this->COMMA.isDown = false;
        this->Emit(L"keyup", &this->COMMA);
        break;
    case 0x34:
        this->PERIOD.isDown = false;
        this->Emit(L"keyup", &this->PERIOD);
        break;
    case 0x35:
        this->SLASH.isDown = false;
        this->Emit(L"keyup", &this->SLASH);
        break;
    case 0x36:
        this->RSHIFT.isDown = false;
        this->Emit(L"keyup", &this->RSHIFT);
        break;
    case 0x1d:
        this->LCTRL.isDown = false;
        this->Emit(L"keyup", &this->LCTRL);
        break;
    case 0x38:
        this->LALT.isDown = false;
        this->Emit(L"keyup", &this->LALT);
        break;
    case 0x39:
        this->SPACE.isDown = false;
        this->Emit(L"keyup", &this->SPACE);
        break;
    case 0xe038:
        this->RALT.isDown = false;
        this->Emit(L"keyup", &this->RALT);
        break;
    case 0xe01d:
        this->RCTRL.isDown = false;
        this->Emit(L"keyup", &this->RCTRL);
        break;
    case 0xe052:
        this->INSERT.isDown = false;
        this->Emit(L"keyup", &this->INSERT);
        break;
    case 0xe047:
        this->HOME.isDown = false;
        this->Emit(L"keyup", &this->HOME);
        break;
    case 0xe049:
        this->PGUP.isDown = false;
        this->Emit(L"keyup", &this->PGUP);
        break;
    case 0xe053:
        this->DEL.isDown = false;
        this->Emit(L"keyup", &this->DEL);
        break;
    case 0xe04f:
        this->END.isDown = false;
        this->Emit(L"keyup", &this->END);
        break;
    case 0xe051:
        this->PGDN.isDown = false;
        this->Emit(L"keyup", &this->PGDN);
        break;
    case 0xe048:
        this->UP.isDown = false;
        this->Emit(L"keyup", &this->UP);
        break;
    case 0xe04b:
        this->LEFT.isDown = false;
        this->Emit(L"keyup", &this->LEFT);
        break;
    case 0xe050:
        this->DOWN.isDown = false;
        this->Emit(L"keyup", &this->DOWN);
        break;
    case 0xe04d:
        this->RIGHT.isDown = false;
        this->Emit(L"keyup", &this->RIGHT);
        break;
    case 0x45:
        this->NUMLOCK.isDown = false;
        this->Emit(L"keyup", &this->NUMLOCK);
        break;
    case 0xe035:
        this->NUMPAD_DIV.isDown = false;
        this->Emit(L"keyup", &this->NUMPAD_DIV);
        break;
    case 0x37:
        this->NUMPAD_MUL.isDown = false;
        this->Emit(L"keyup", &this->NUMPAD_MUL);
        break;
    case 0x4a:
        this->NUMPAD_MINUS.isDown = false;
        this->Emit(L"keyup", &this->NUMPAD_MINUS);
        break;
    case 0x4e:
        this->NUMPAD_PLUS.isDown = false;
        this->Emit(L"keyup", &this->NUMPAD_PLUS);
        break;
    case 0x53:
        this->NUMPAD_DECIMAL.isDown = false;
        this->Emit(L"keyup", &this->NUMPAD_DECIMAL);
        break;
    case 0x52:
        this->NUMPAD_0.isDown = false;
        this->Emit(L"keyup", &this->NUMPAD_0);
        break;
    case 0x4f:
        this->NUMPAD_1.isDown = false;
        this->Emit(L"keyup", &this->NUMPAD_1);
        break;
    case 0x50:
        this->NUMPAD_2.isDown = false;
        this->Emit(L"keyup", &this->NUMPAD_2);
        break;
    case 0x51:
        this->NUMPAD_3.isDown = false;
        this->Emit(L"keyup", &this->NUMPAD_3);
        break;
    case 0x4b:
        this->NUMPAD_4.isDown = false;
        this->Emit(L"keyup", &this->NUMPAD_4);
        break;
    case 0x4c:
        this->NUMPAD_5.isDown = false;
        this->Emit(L"keyup", &this->NUMPAD_5);
        break;
    case 0x4d:
        this->NUMPAD_6.isDown = false;
        this->Emit(L"keyup", &this->NUMPAD_6);
        break;
    case 0x47:
        this->NUMPAD_7.isDown = false;
        this->Emit(L"keyup", &this->NUMPAD_7);
        break;
    case 0x48:
        this->NUMPAD_8.isDown = false;
        this->Emit(L"keyup", &this->NUMPAD_8);
        break;
    case 0x49:
        this->NUMPAD_9.isDown = false;
        this->Emit(L"keyup", &this->NUMPAD_9);
        break;
    }
}
