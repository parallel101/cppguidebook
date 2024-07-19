#include <ctime>
#include <iomanip>
#include <iostream>
#include <locale>

int main() {
    time_t t = time(NULL);
    tm *tm = localtime(&t);

    auto locale_zh = std::locale("zh_CN.UTF-8");
    std::cout.imbue(locale_zh);
    std::cout << std::put_time(tm, "%c") << '\n';

    auto locale_en = std::locale("en_US.UTF-8");
    std::cout.imbue(locale_en);
    std::cout << std::put_time(tm, "%c") << '\n';
}
