#pragma once

#include <cstdint>
#include <cstddef>
#include <cstdio>

template <class T>
void memory_view(const T &obj)
{
    auto buf = reinterpret_cast<const uint8_t *>(&obj);
    auto len = sizeof(T);

    for (size_t i = 0; i < len; ++i) {
        printf(" %02X", buf[i]);
    }
    printf("\n");
}
