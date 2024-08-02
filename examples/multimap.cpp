#include <fmt/core.h>
#include <fmt/ranges.h>
#include <map>
#include <string>

int main() {
    std::multimap<std::string, std::string> tab;
    tab.insert({"rust", "silly"});
    tab.insert({"rust", "trash"});
    tab.insert({"rust", "trash"});
    tab.insert({"cpp", "smart"});
    tab.insert({"rust", "lazy"});
    tab.insert({"cpp", "fast"});
    tab.insert({"java", "pig"});
    fmt::println("{}", tab);
    return 0;
}
