#include <iostream>

int main() {
    setlocale(LC_ALL, "");
    std::wcout << L"我是 wcout!" << L'\n';
    std::cout << "我是 cout!" << '\n';
    return 0;
}
