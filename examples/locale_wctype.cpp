#include <fmt/format.h>
#include <boost/locale.hpp>

int main() {
    fmt::println("默认: {}", iswpunct(L'，'));

    setlocale(LC_ALL, "C");
    fmt::println("C: {}", iswpunct(L'，'));

    setlocale(LC_ALL, "C.UTF-8");
    fmt::println("C.UTF-8: {}", iswpunct(L'，'));

    setlocale(LC_ALL, "zh_CN.UTF-8");
    fmt::println("zh_CN.UTF-8: {}", iswpunct(L'，'));

    setlocale(LC_ALL, "zh_CN.GBK");
    fmt::println("zh_CN.GBK: {}", iswpunct(L'，'));

    setlocale(LC_ALL, "en_US.ISO-8859-1");
    fmt::println("en_US.ISO-8859-1: {}", iswpunct(L'，'));

    setlocale(LC_ALL, "POSIX");
    fmt::println("POSIX: {}", iswpunct(L'，'));

    setlocale(LC_ALL, "en_CA.ISO-8859-1");
    fmt::println("en_CA: {}", iswalpha(L'é'));
    setlocale(LC_ALL, "fr_CA.ISO-8859-1");
    fmt::println("fr_CA: {}", iswalpha(L'é'));

    return 0;
}
