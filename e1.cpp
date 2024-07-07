#include <fmt/format.h>

int hello(int x, int y) {
    fmt::println("hello({}, {})", x, y);
    return x + y;
}

int main() {
    fmt::println("main 调用 hello 结果：{}", hello(2, 3));
    return 0;
}
