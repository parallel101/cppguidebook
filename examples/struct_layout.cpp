#include "memory_view.h"

struct C    // size=16 align=8
{
    // 0B
    double d; // size=8 align=8
    // 8B
    int i;    // size=4 align=4
    // 12B
    short s;  // size=2 align=2
    // 14B
    char c;   // size=1 align=1
    // 15B
    // char padding[1];
    // 16B
};

struct B  // size=4 align=4
{
    int i;
};

struct D : C, B   // size=24 align=8
{
    // C c;
    // 16B
    // B b;
    // 20B
    int j;  // size=4 align=4
    // 24B
};

int main()
{
    C c{1.0, 2, 3, 4};
    D d{{1.0, 2, 3, 4}, {5}, 6};
    memory_view(c);
    memory_view(d);
}
