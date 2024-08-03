#include <map>
#include <cstdio>

int main() {
    struct MyClass {
        MyClass() { printf("MyClass()\n"); }
        MyClass(MyClass &&) noexcept { printf("MyClass(MyClass &&)\n"); }
        MyClass &operator=(MyClass &&) noexcept { printf("MyClass &operator=(MyClass &&)\n"); return *this; }
    };

    std::map<int, MyClass> tab;
    printf("insert+make_pair的开销:\n");
    tab.insert(std::make_pair(1, MyClass()));
    printf("insert的开销:\n");
    tab.insert({2, MyClass()});
    printf("try_emplace的开销:\n");
    tab.try_emplace(3);
    // try_emplace 只有一个 key 参数时，相当于调用无参构造函数 MyClass()
    return 0;
}
